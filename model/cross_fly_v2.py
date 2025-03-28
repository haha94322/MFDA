import torch.nn as nn
import torch
from torch.nn import functional as F





class pearson_channel(torch.nn.Module):
    def __init__(self, moduledim=512, channels=None, e_lambda=1e-4):
        super(pearson_channel, self).__init__()

        self.activaton = nn.LeakyReLU()
        self.e_lambda = e_lambda


    def pearson_distance(self, x, y):
        mean_x = x.mean(dim=[-2], keepdim=True)
        mean_y = y.mean(dim=[-2], keepdim=True)
        c = torch.tensor([True]).cuda()
        d = torch.tensor([0]).cuda()
        # 计算 Pearson 相关系数的分子部分
        cov_xy = (x - mean_x) * (y - mean_y)

        # 计算 Pearson 相关系数的分母部分
        a = (x - mean_x).pow(2).sum(dim=[-2], keepdim=True)
        b = (y - mean_y).pow(2).sum(dim=[-2], keepdim=True)

        # 计算 Pearson 相关系数
        rho = cov_xy / (torch.sqrt(a * b + 0.0001))

        return rho

    def forward(self, x, y):

        matrix_tmp = self.pearson_distance(x, y)  # ** 2
        value_imag = torch.atan2(matrix_tmp.imag, (matrix_tmp.real + 0.0001))
        weight_matrix = 2 / (1 + torch.exp(-1 * (value_imag) * 0.5)) - 1
        weight_matrix[torch.isnan(weight_matrix)] = 0.0
        weight_matrix = weight_matrix.clone().detach()

        # print(weight_matrix.max(),weight_matrix.min())

        freq_distance = (torch.abs(x - y)) ** 2
        weight = torch.exp(-(torch.abs(freq_distance)) * 0.5)
        weight = weight * weight_matrix

        return weight


class channel_crossmodel(torch.nn.Module):
    def __init__(self, moduledim, kernel_size, dsl_init_sigma, dsl_init_center, channels=None, e_lambda=1e-4):
        super(channel_crossmodel, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        self.Norm = nn.BatchNorm2d(moduledim)
        self.pearson = pearson_channel()

        self.kernel_num = 4

        self.sigmas = nn.Parameter(torch.empty(self.kernel_num, 1), requires_grad=True)
        self.center_shift = nn.Parameter(torch.empty(self.kernel_num, 1),
            requires_grad=True)

        nn.init.uniform_(self.sigmas)
        nn.init.uniform_(self.center_shift)

        self.kernel_size = kernel_size
        self.padding = [self.kernel_size // 2, self.kernel_size // 2]


    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)

        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def _get_gaussian_kernel1d(self, kernel_size, sigma, center_shift):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).cuda()
        pdf = torch.exp(-0.5 * ((x - center_shift) / (sigma + 0.001)).pow(2))
        kernel1d = pdf / (pdf.sum() + 0.01)
        return kernel1d

    def _gaussian_blur(self, x_org, i):
        x = F.pad(x_org, self.padding, mode="replicate")
        self.gauss_kernel = self._get_gaussian_kernel1d(self.kernel_size, self.sigmas[i], self.center_shift[i])
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-2], 1, 1)
        feat = F.conv1d(x, self.gauss_kernel, groups=x.shape[-2])
        return feat

    def forward(self, x, q):
        # q = self.q_porj(q)
        q_bak = q
        outputs = []
        x_shift_bak = []
        for i in range(self.kernel_num):
            x_shift = self._gaussian_blur(x, i)
            x_shift_bak.append(x_shift)

        x_shift = torch.cat(
            [value_fft.unsqueeze(1) for value_fft in x_shift_bak],
            dim=1)

        query_fft_shift = torch.fft.fft(q, dim=-2).unsqueeze(1)

        value_fft_shift = torch.fft.fft(x_shift, n=q.size(-2), dim=-2)


        weight = self.pearson(value_fft_shift, query_fft_shift)  # ** 2

        value_filter = query_fft_shift * (1 - weight) + value_fft_shift * weight

        outputs = torch.abs(torch.fft.ifft(value_filter, n=x.size(-2), dim=-2))



        w_b_2_a = F.softmax(torch.matmul(q_bak.mean(-2).unsqueeze(-2).repeat(1, outputs.size(2), 1).unsqueeze(-2),
                                         outputs.permute(0, 2, 3, 1)), dim=-1)

        outputs = torch.matmul(w_b_2_a, outputs.permute(0, 2, 1, 3)).squeeze()


        return outputs

class crossmodel(torch.nn.Module):
    def __init__(self, moduledim, kernel_size, dsl_init_sigma, dsl_init_center, channels=None, e_lambda=1e-4):
        super(crossmodel, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

        self.Norm = nn.BatchNorm2d(moduledim)
        self.pearson = pearson_channel()

        self.kernel_num = 4

        self.sigmas = nn.Parameter(torch.empty(self.kernel_num, 1), requires_grad=True)
        self.center_shift = nn.Parameter(torch.empty(self.kernel_num, 1),
            requires_grad=True)

        nn.init.uniform_(self.sigmas)
        nn.init.uniform_(self.center_shift)

        self.kernel_size = kernel_size
        self.padding = [self.kernel_size // 2, self.kernel_size // 2]


    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)

        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def _get_gaussian_kernel1d(self, kernel_size, sigma, center_shift):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).cuda()
        pdf = torch.exp(-0.5 * ((x - center_shift) / (sigma + 0.001)).pow(2))
        kernel1d = pdf / (pdf.sum() + 0.01)
        return kernel1d

    def _gaussian_blur(self, x_org, i):
        x = F.pad(x_org, self.padding, mode="replicate")
        self.gauss_kernel = self._get_gaussian_kernel1d(self.kernel_size, self.sigmas[i], self.center_shift[i])
        self.gauss_kernel = self.gauss_kernel.repeat(x.shape[-2], 1, 1)
        feat = F.conv1d(x, self.gauss_kernel, groups=x.shape[-2])
        return feat

    def forward(self, x, q):
        # q = self.q_porj(q)
        q_bak = q
        outputs = []
        x_shift_bak = []
        for i in range(self.kernel_num):
            x_shift = self._gaussian_blur(x, i)
            x_shift_bak.append(x_shift)

        x_shift = torch.cat(
            [value_fft.unsqueeze(1) for value_fft in x_shift_bak],
            dim=1)

        query_fft_shift = torch.fft.fft(q, dim=-2).unsqueeze(1)

        value_fft_shift = torch.fft.fft(x_shift, n=q.size(-2), dim=-2)


        weight = self.pearson(value_fft_shift, query_fft_shift)

        value_filter = query_fft_shift * (1 - weight) + value_fft_shift * weight


        outputs = torch.abs(torch.fft.ifft(value_filter, n=x.size(-2), dim=-2))



        w_b_2_a = F.softmax(torch.matmul(q_bak.mean(-2).unsqueeze(-2).repeat(1, outputs.size(2), 1).unsqueeze(-2),
                                         outputs.permute(0, 2, 3, 1)), dim=-1)

        outputs = torch.matmul(w_b_2_a, outputs.permute(0, 2, 1, 3)).squeeze()


        return outputs


class fly_1d(nn.Module):
    """feature fusion"""
    def __init__(self, kernel_size = 3, moduledim = 512, dsl_init_sigma = 3.0, dsl_init_center = 0.0):
        super(fly_1d, self).__init__()

        self.channel_cross = crossmodel(moduledim, kernel_size, dsl_init_sigma, dsl_init_center)
        self.cross = channel_crossmodel(moduledim, kernel_size, dsl_init_sigma, dsl_init_center)

    def forward(self, value, query, video_query):

        value_channel = self.cross(value, video_query)

        value_channel = self.channel_cross(value_channel, query)


        return value_channel, 0