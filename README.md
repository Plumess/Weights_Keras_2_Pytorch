# Weights_Keras_2_Pytorch

最近想在Pytorch项目里使用一下谷歌的NIMA，但是发现没有预训练好的pytorch权重，于是整理了一下将Keras预训练权重转为Pytorch的代码，目前是支持Keras的Conv2D, Dense, DepthwiseConv2D, BatchNormlization的转换。需要注意的是在**Pytorch模型内需要给每一层命名为与Keras每一层相同的名字**，才能对应转换。

I recently wanted to use Google's NIMA in the Pytorch project, but found that there were no pre-trained pytorch weights, so I organized the code to convert Keras pre-trained weights to Pytorch, which currently supports Keras' Conv2D, Dense, DepthwiseConv2D, BatchNormlization conversion. **Note that you need to name each layer within the Pytorch model with the same name as each layer of Keras** in order to correspond to the conversion.

## 文件介绍：

**weights_keras2pytorch.py** 是Keras预训练权重转为Pytorch的代码；

weights_keras2pytorch.py is the code for converting Keras pre-trained weights to Pytorch.



**model_keras_NIMA.py** 是谷歌NR-IQA NIMA的Keras版模型代码；

model_keras_NIMA.py is the Keras version of the Google NR-IQA NIMA model code.



**model_pytorch_NIMA.py** 是NIMA的Pytorch版模型代码；

model_pytorch_NIMA.py is the Pytorch version of the model code for NIMA.



**mobilenet_weights.h5** 是用mobilenet实现NIMA的预训练权重；

mobilenet_weights.h5 is a pre-trained weights for implementing NIMA with mobilenet.



**NIMA_pytorch_model.pth** 是用转换代码转换出来的权重；

NIMA_pytorch_model.pth is the weights converted with the conversion code.



## Requirements：

h5py==3.1.0

###### Keras

keras==2.6.0
Keras-Preprocessing==1.1.2

###### Tensorflow

tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow-estimator==2.7.0
tensorflow-gpu==2.6.0

###### Pytorch

torch==1.10.0
torchaudio==0.10.0
torchvision==0.11.1



上述只是我用的环境，并不是一定需要这么高的版本。

The above is just the environment I use, and does not necessarily require such a high version.
