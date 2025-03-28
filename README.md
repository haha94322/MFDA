# Multi-granularity Frequency Difference-Aware Attention for Video Question Answering

Release the code for ICME 2025

MFDA Unit             
-------------------------
![](MFDA.png)  


## evaluate on MSVD-QA
download our pre-extracted features and Pretrained model for msvd_qa from [here](https://pan.baidu.com/s/1jQ5HxITAeY7jiq9yKWcpTg?pwd=ge1c) with password (ge1c) and save them in `data` folder.
   


## Acknowledgement
- As for motion feature extraction, we adapt ResNeXt-101 model from this [repo](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to our code. Thank @kenshohara for releasing the code and the pretrained models. 
- We refer to this [repo](https://github.com/facebookresearch/clevr-iep) for preprocessing.
- Our implementation of dataloader is based on this [repo](https://github.com/shijx12/XNM-Net).
- We adapt HCRN-videoQA from this [repo](https://github.com/thaolmk54/hcrn-videoqa) to our code. Thank @thaolmk54 for releasing the code and the pretrained models. 
