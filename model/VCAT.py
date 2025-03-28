import numpy as np
from torch.nn import functional as F

from .utils import *
from model.cross_fly_v2 import fly_1d as MFDA


class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill


class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        output, (hidden, _) = self.encoder(embed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=None)
        text_embedding = output

        question_embedding = torch.cat([hidden[0], hidden[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding, text_embedding


class InputUnitVisual(nn.Module):
    def __init__(self, vision_dim, module_dim=512):
        super(InputUnitVisual, self).__init__()

        self.clip_level_motion_cond = MFDA(5, module_dim)
        self.clip_level_question_cond = MFDA(5, module_dim)
        self.video_level_motion_cond = MFDA(5, module_dim)
        self.video_level_question_cond = MFDA(5, module_dim)

        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, text_embedding):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        batch_size = appearance_video_feat.size(0)
        clip_level_mfda_outputs = []
        question_embedding_proj = self.question_embedding_proj(question_embedding)
        clip_level_motion = self.clip_level_motion_proj(motion_video_feat)
        clip_question_loss = 0
        clip_motion_loss = 0
        for i in range(appearance_video_feat.size(1)):

            clip_level_appearance = appearance_video_feat[:, i, :, :]  # (bz, 16, 2048)
            clip_level_appearance_proj = self.appearance_feat_proj(clip_level_appearance)  # (bz, 16, 512)

            clip_level_mfda_motion, loss_clip_level_motion \
                = self.clip_level_motion_cond(clip_level_appearance_proj,
                                                                clip_level_motion, text_embedding)
            clip_level_mfda_question, loss_clip_level_question \
                = self.clip_level_question_cond(clip_level_mfda_motion,
                                                                    question_embedding_proj, text_embedding)


            clip_level_mfda_outputs.append(clip_level_mfda_question.sum(1))

            clip_motion_loss =  clip_motion_loss + loss_clip_level_motion
            clip_question_loss = clip_question_loss + loss_clip_level_question
        clip_question_loss = clip_question_loss / appearance_video_feat.size(1)
        clip_motion_loss = clip_motion_loss / appearance_video_feat.size(1)

        clip_level_mfda_outputs = torch.cat(
            [frame_relation.unsqueeze(1) for frame_relation in clip_level_mfda_outputs],
            dim=1)


        video_level_mfda_motion, loss_video_level_motion  \
            = self.video_level_motion_cond(clip_level_mfda_outputs, clip_level_motion, text_embedding)
        video_level_mfda_question, loss_video_level_question \
            = self.video_level_question_cond(video_level_mfda_motion,
                                                                  question_embedding_proj, text_embedding)

        loss = clip_question_loss + clip_motion_loss + loss_video_level_motion + loss_video_level_question


        return video_level_mfda_question, loss


class OutputUnitOpenEnded(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out


class OutputUnitMultiChoices(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitMultiChoices, self).__init__()

        self.question_proj = nn.Linear(module_dim, module_dim)

        self.ans_candidates_proj = nn.Linear(module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 4, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, 1))

    def forward(self, question_embedding, q_visual_embedding, ans_candidates_embedding,
                a_visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        ans_candidates_embedding = self.ans_candidates_proj(ans_candidates_embedding)
        out = torch.cat([q_visual_embedding, question_embedding, a_visual_embedding,
                         ans_candidates_embedding], 1)
        out = self.classifier(out)

        return out




class VCATNetwork(nn.Module):
    def __init__(self, vision_dim, module_dim, word_dim, k_max_frame_level,
                 k_max_clip_level, spl_resolution, vocab, question_type, calculate_loss = False):
        super(VCATNetwork, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation(module_dim)

        encoder_vocab_size = len(vocab['question_token_to_idx'])
        self.num_classes = len(vocab['answer_token_to_idx'])
        self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                         module_dim=module_dim, rnn_dim=module_dim)
        self.linguistic_input_video_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,
                                                               module_dim=module_dim, rnn_dim=module_dim)
        self.visual_input_unit = InputUnitVisual(vision_dim=vision_dim,
                                                 module_dim=module_dim)
        self.output_unit = OutputUnitOpenEnded(num_answers=self.num_classes)


        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat, question,
                question_len, video_questions, video_questions_len, _):

        question_video_embedding, text_video_embedding = self.linguistic_input_video_unit(video_questions,
                                                                                          video_questions_len.cpu())
        question_embedding, text_embedding = self.linguistic_input_unit(question, question_len.cpu())
        visual_embedding, loss = self.visual_input_unit(video_appearance_feat, video_motion_feat, text_embedding,
                                                        text_video_embedding)

        visual_embedding = self.feature_aggregation(question_embedding, visual_embedding)

        out = self.output_unit(question_embedding, visual_embedding)
        return out
