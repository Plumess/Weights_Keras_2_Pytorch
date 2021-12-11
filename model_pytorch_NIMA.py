import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.autograd import Variable

def DepthwiseConv2D(in_channels,stride,padding):
    return nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=stride,padding=padding,groups=in_channels,bias=False)

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        channels = 32
        self.input_1 = nn.Sequential()

        # conv_block
        stride = 2
        self.conv1_pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=channels,kernel_size=3,stride=stride,padding=0,bias=False)
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv1_relu = nn.ReLU()

        # DepthwiseConv_block1
        mul = 2
        stride = 1
        padding = 1
        self.conv_dw_1 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_1_bn = nn.BatchNorm2d(channels)
        self.conv_dw_1_relu  = nn.ReLU()
        self.conv_pw_1 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels*=mul # 64
        self.conv_pw_1_bn = nn.BatchNorm2d(channels)
        self.conv_pw_1_relu = nn.ReLU()

        # DepthwiseConv_block2
        mul = 2
        stride = 2
        padding = 0
        self.conv_pad_2 = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.conv_dw_2 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_2_bn = nn.BatchNorm2d(channels)
        self.conv_dw_2_relu = nn.ReLU()
        self.conv_pw_2 = nn.Conv2d(in_channels=channels, out_channels=channels * mul, kernel_size=1, stride=1,padding=0, bias=False)
        channels *= mul  # 128
        self.conv_pw_2_bn = nn.BatchNorm2d(channels)
        self.conv_pw_2_relu = nn.ReLU()

        # DepthwiseConv_block3
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_3 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_3_bn = nn.BatchNorm2d(channels)
        self.conv_dw_3_relu  = nn.ReLU()
        self.conv_pw_3 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 128
        self.conv_pw_3_bn = nn.BatchNorm2d(channels)
        self.conv_pw_3_relu = nn.ReLU()

        # DepthwiseConv_block4
        mul = 2
        stride = 2
        padding = 0
        self.conv_pad_4 = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.conv_dw_4 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_4_bn = nn.BatchNorm2d(channels)
        self.conv_dw_4_relu = nn.ReLU()
        self.conv_pw_4 = nn.Conv2d(in_channels=channels, out_channels=channels * mul, kernel_size=1, stride=1,padding=0, bias=False)
        channels *= mul  # 256
        self.conv_pw_4_bn = nn.BatchNorm2d(channels)
        self.conv_pw_4_relu = nn.ReLU()

        # DepthwiseConv_block5
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_5 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_5_bn = nn.BatchNorm2d(channels)
        self.conv_dw_5_relu  = nn.ReLU()
        self.conv_pw_5 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 258
        self.conv_pw_5_bn = nn.BatchNorm2d(channels)
        self.conv_pw_5_relu = nn.ReLU()

        # DepthwiseConv_block6
        mul = 2
        stride = 2
        padding = 0
        self.conv_pad_6 = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.conv_dw_6 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_6_bn = nn.BatchNorm2d(channels)
        self.conv_dw_6_relu = nn.ReLU()
        self.conv_pw_6 = nn.Conv2d(in_channels=channels, out_channels=channels * mul, kernel_size=1, stride=1,padding=0, bias=False)
        channels *= mul  # 512
        self.conv_pw_6_bn = nn.BatchNorm2d(channels)
        self.conv_pw_6_relu = nn.ReLU()

        # DepthwiseConv_block7
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_7 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_7_bn = nn.BatchNorm2d(channels)
        self.conv_dw_7_relu  = nn.ReLU()
        self.conv_pw_7 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 512
        self.conv_pw_7_bn = nn.BatchNorm2d(channels)
        self.conv_pw_7_relu = nn.ReLU()

        # DepthwiseConv_block8
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_8 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_8_bn = nn.BatchNorm2d(channels)
        self.conv_dw_8_relu  = nn.ReLU()
        self.conv_pw_8 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 512
        self.conv_pw_8_bn = nn.BatchNorm2d(channels)
        self.conv_pw_8_relu = nn.ReLU()

        # DepthwiseConv_block9
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_9 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_9_bn = nn.BatchNorm2d(channels)
        self.conv_dw_9_relu  = nn.ReLU()
        self.conv_pw_9 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 512
        self.conv_pw_9_bn = nn.BatchNorm2d(channels)
        self.conv_pw_9_relu = nn.ReLU()

        # DepthwiseConv_block10
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_10 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_10_bn = nn.BatchNorm2d(channels)
        self.conv_dw_10_relu  = nn.ReLU()
        self.conv_pw_10 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 512
        self.conv_pw_10_bn = nn.BatchNorm2d(channels)
        self.conv_pw_10_relu = nn.ReLU()

        # DepthwiseConv_block11
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_11 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_11_bn = nn.BatchNorm2d(channels)
        self.conv_dw_11_relu  = nn.ReLU()
        self.conv_pw_11 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 512
        self.conv_pw_11_bn = nn.BatchNorm2d(channels)
        self.conv_pw_11_relu = nn.ReLU()

        # DepthwiseConv_block12
        mul = 2
        stride = 2
        padding = 0
        self.conv_pad_12 = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.conv_dw_12 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_12_bn = nn.BatchNorm2d(channels)
        self.conv_dw_12_relu = nn.ReLU()
        self.conv_pw_12 = nn.Conv2d(in_channels=channels, out_channels=channels * mul, kernel_size=1, stride=1,padding=0, bias=False)
        channels *= mul  # 1024
        self.conv_pw_12_bn = nn.BatchNorm2d(channels)
        self.conv_pw_12_relu = nn.ReLU()

        # DepthwiseConv_block13
        mul = 1
        stride = 1
        padding = 1
        self.conv_dw_13 = DepthwiseConv2D(channels,stride,padding)
        self.conv_dw_13_bn = nn.BatchNorm2d(channels)
        self.conv_dw_13_relu  = nn.ReLU()
        self.conv_pw_13 = nn.Conv2d(in_channels=channels,out_channels=channels*mul,kernel_size=1,stride=1,padding=0,bias=False)
        channels *= mul # 1024
        self.conv_pw_13_bn = nn.BatchNorm2d(channels)
        self.conv_pw_13_relu = nn.ReLU()


        self.global_average_pooling2d_1 = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.75)
        self.dense = nn.Linear(channels,10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # conv_block
        # print("conv_block:")
        input_1 = self.input_1(x)
        conv1_pad = self.conv1_pad(x)
        conv1 = self.conv1(conv1_pad)
        conv1_bn = self.conv1_bn(conv1)
        conv1_relu = self.conv1_relu(conv1_bn)
        lastout = conv1_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 1
        # print("DWBlockID:", 1)
        # lastout = self.conv_pad_1(lastout)
        # # print(lastout.shape)
        conv_dw_1 = self.conv_dw_1(lastout)
        # print(conv_dw_1.shape)
        conv_dw_1_bn = self.conv_dw_1_bn(conv_dw_1)
        conv_dw_1_relu = self.conv_dw_1_relu(conv_dw_1_bn)
        conv_pw_1 = self.conv_pw_1(conv_dw_1_relu)
        conv_pw_1_bn = self.conv_pw_1_bn(conv_pw_1)
        conv_pw_1_relu = self.conv_pw_1_relu(conv_pw_1_bn)
        lastout = conv_pw_1_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 2
        # print("DWBlockID:",2)
        lastout = self.conv_pad_2(lastout)
        # print(lastout.shape)
        conv_dw_2 = self.conv_dw_2(lastout)
        # print(conv_dw_2.shape)
        conv_dw_2_bn = self.conv_dw_2_bn(conv_dw_2)
        conv_dw_2_relu = self.conv_dw_2_relu(conv_dw_2_bn)
        conv_pw_2 = self.conv_pw_2(conv_dw_2_relu)
        conv_pw_2_bn = self.conv_pw_2_bn(conv_pw_2)
        conv_pw_2_relu = self.conv_pw_2_relu(conv_pw_2_bn)
        lastout = conv_pw_2_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 3
        # print("DWBlockID:",3)
        # lastout = self.conv_pad_3(lastout)
        # # print(lastout.shape)
        conv_dw_3 = self.conv_dw_3(lastout)
        # print(conv_dw_3.shape)
        conv_dw_3_bn = self.conv_dw_3_bn(conv_dw_3)
        conv_dw_3_relu = self.conv_dw_3_relu(conv_dw_3_bn)
        conv_pw_3 = self.conv_pw_3(conv_dw_3_relu)
        conv_pw_3_bn = self.conv_pw_3_bn(conv_pw_3)
        conv_pw_3_relu = self.conv_pw_3_relu(conv_pw_3_bn)
        lastout = conv_pw_3_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 4
        # print("DWBlockID:",4)
        lastout = self.conv_pad_4(lastout)
        # print(lastout.shape)
        conv_dw_4 = self.conv_dw_4(lastout)
        # print(conv_dw_4.shape)
        conv_dw_4_bn = self.conv_dw_4_bn(conv_dw_4)
        conv_dw_4_relu = self.conv_dw_4_relu(conv_dw_4_bn)
        conv_pw_4 = self.conv_pw_4(conv_dw_4_relu)
        conv_pw_4_bn = self.conv_pw_4_bn(conv_pw_4)
        conv_pw_4_relu = self.conv_pw_4_relu(conv_pw_4_bn)
        lastout = conv_pw_4_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 5
        # print("DWBlockID:",5)
        # lastout = self.conv_pad_5(lastout)
        # # print(lastout.shape)
        conv_dw_5 = self.conv_dw_5(lastout)
        # print(conv_dw_5.shape)
        conv_dw_5_bn = self.conv_dw_5_bn(conv_dw_5)
        conv_dw_5_relu = self.conv_dw_5_relu(conv_dw_5_bn)
        conv_pw_5 = self.conv_pw_5(conv_dw_5_relu)
        conv_pw_5_bn = self.conv_pw_5_bn(conv_pw_5)
        conv_pw_5_relu = self.conv_pw_5_relu(conv_pw_5_bn)
        lastout = conv_pw_5_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 6
        # print("DWBlockID:",6)
        lastout = self.conv_pad_6(lastout)
        # print(lastout.shape)
        conv_dw_6 = self.conv_dw_6(lastout)
        # print(conv_dw_6.shape)
        conv_dw_6_bn = self.conv_dw_6_bn(conv_dw_6)
        conv_dw_6_relu = self.conv_dw_6_relu(conv_dw_6_bn)
        conv_pw_6 = self.conv_pw_6(conv_dw_6_relu)
        conv_pw_6_bn = self.conv_pw_6_bn(conv_pw_6)
        conv_pw_6_relu = self.conv_pw_6_relu(conv_pw_6_bn)
        lastout = conv_pw_6_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 7
        # print("DWBlockID:",7)
        # lastout = self.conv_pad_7(lastout)
        # # print(lastout.shape)
        conv_dw_7 = self.conv_dw_7(lastout)
        # print(conv_dw_7.shape)
        conv_dw_7_bn = self.conv_dw_7_bn(conv_dw_7)
        conv_dw_7_relu = self.conv_dw_7_relu(conv_dw_7_bn)
        conv_pw_7 = self.conv_pw_7(conv_dw_7_relu)
        conv_pw_7_bn = self.conv_pw_7_bn(conv_pw_7)
        conv_pw_7_relu = self.conv_pw_7_relu(conv_pw_7_bn)
        lastout = conv_pw_7_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 8
        # print("DWBlockID:",8)
        # lastout = self.conv_pad_8(lastout)
        # # print(lastout.shape)
        conv_dw_8 = self.conv_dw_8(lastout)
        # print(conv_dw_8.shape)
        conv_dw_8_bn = self.conv_dw_8_bn(conv_dw_8)
        conv_dw_8_relu = self.conv_dw_8_relu(conv_dw_8_bn)
        conv_pw_8 = self.conv_pw_8(conv_dw_8_relu)
        conv_pw_8_bn = self.conv_pw_8_bn(conv_pw_8)
        conv_pw_8_relu = self.conv_pw_8_relu(conv_pw_8_bn)
        lastout = conv_pw_8_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 9
        # print("DWBlockID:",9)
        # lastout = self.conv_pad_9(lastout)
        # # print(lastout.shape)
        conv_dw_9 = self.conv_dw_9(lastout)
        # print(conv_dw_9.shape)
        conv_dw_9_bn = self.conv_dw_9_bn(conv_dw_9)
        conv_dw_9_relu = self.conv_dw_9_relu(conv_dw_9_bn)
        conv_pw_9 = self.conv_pw_9(conv_dw_9_relu)
        conv_pw_9_bn = self.conv_pw_9_bn(conv_pw_9)
        conv_pw_9_relu = self.conv_pw_9_relu(conv_pw_9_bn)
        lastout = conv_pw_9_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 10
        # print("DWBlockID:",10)
        # lastout = self.conv_pad_10(lastout)
        # # print(lastout.shape)
        conv_dw_10 = self.conv_dw_10(lastout)
        # print(conv_dw_10.shape)
        conv_dw_10_bn = self.conv_dw_10_bn(conv_dw_10)
        conv_dw_10_relu = self.conv_dw_10_relu(conv_dw_10_bn)
        conv_pw_10 = self.conv_pw_10(conv_dw_10_relu)
        conv_pw_10_bn = self.conv_pw_10_bn(conv_pw_10)
        conv_pw_10_relu = self.conv_pw_10_relu(conv_pw_10_bn)
        lastout = conv_pw_10_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 11
        # print("DWBlockID:",11)
        # lastout = self.conv_pad_11(lastout)
        # # print(lastout.shape)
        conv_dw_11 = self.conv_dw_11(lastout)
        # print(conv_dw_11.shape)
        conv_dw_11_bn = self.conv_dw_11_bn(conv_dw_11)
        conv_dw_11_relu = self.conv_dw_11_relu(conv_dw_11_bn)
        conv_pw_11 = self.conv_pw_11(conv_dw_11_relu)
        conv_pw_11_bn = self.conv_pw_11_bn(conv_pw_11)
        conv_pw_11_relu = self.conv_pw_11_relu(conv_pw_11_bn)
        lastout = conv_pw_11_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 12
        # print("DWBlockID:",12)
        lastout = self.conv_pad_12(lastout)
        # print(lastout.shape)
        conv_dw_12 = self.conv_dw_12(lastout)
        # print(conv_dw_12.shape)
        conv_dw_12_bn = self.conv_dw_12_bn(conv_dw_12)
        conv_dw_12_relu = self.conv_dw_12_relu(conv_dw_12_bn)
        conv_pw_12 = self.conv_pw_12(conv_dw_12_relu)
        conv_pw_12_bn = self.conv_pw_12_bn(conv_pw_12)
        conv_pw_12_relu = self.conv_pw_12_relu(conv_pw_12_bn)
        lastout = conv_pw_12_relu
        # print(lastout.shape)
        # print()

        # DepthwiseConv_block 13
        # print("DWBlockID:",13)
        # lastout = self.conv_pad_13(lastout)
        # # print(lastout.shape)
        conv_dw_13 = self.conv_dw_13(lastout)
        # print(conv_dw_13.shape)
        conv_dw_13_bn = self.conv_dw_13_bn(conv_dw_13)
        conv_dw_13_relu = self.conv_dw_13_relu(conv_dw_13_bn)
        conv_pw_13 = self.conv_pw_13(conv_dw_13_relu)
        conv_pw_13_bn = self.conv_pw_13_bn(conv_pw_13)
        conv_pw_13_relu = self.conv_pw_13_relu(conv_pw_13_bn)
        lastout = conv_pw_13_relu
        # print(lastout.shape)
        # print()

        global_average_pooling2d_1 = self.global_average_pooling2d_1(lastout)
        global_average_pooling2d_1 = torch.squeeze(global_average_pooling2d_1, 3)
        global_average_pooling2d_1 = torch.squeeze(global_average_pooling2d_1, 2)
        # print("global_average_pooling2d:")
        # print(global_average_pooling2d_1.shape)

        dropout = self.dropout(global_average_pooling2d_1)
        # print("dropout:")
        # print(dropout.shape)

        dense = self.dense(dropout)
        dense = self.softmax(dense)
        # print("dense:")
        # print(dense.shape)
        lastout = dense

        out = lastout
        return out


if __name__ == '__main__':
    model = NIMA()
    x = torch.randn((1, 3, 224, 224))
    x = model(x)
    # # print(x.shape)