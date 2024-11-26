'''
Author: shengyu
Date: 2024-04-08 15:24:09
LastEditors: shengyu
LastEditTime: 2024-07-23 09:11:09
Description: encoder and decoder v1
AUTOENCODER FOR VIBROTACTILE SIGNAL COMPRESSION 复现
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch import nn, optim
from .gdn import GDN
import math
class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        blocks=[nn.Conv1d(1,input_size, kernel_size=5,dilation=3, stride=1, padding=3),
                GDN(input_size), 
                nn.Conv1d(input_size, int(input_size), kernel_size=3,dilation=2,  stride=2, padding=2),
                GDN(int(input_size)),
                nn.Conv1d(input_size, int(input_size), kernel_size=3, dilation=2, stride=2, padding=2),
                GDN(input_size),
                nn.Conv1d(input_size, int(input_size), kernel_size=3, dilation=2, stride=2, padding=2),
                GDN(input_size),
                nn.Conv1d(int(input_size), input_size, kernel_size=3,dilation=2,  stride=2,padding=2,bias=False),
                ]
        self.blocks = nn.Sequential(*blocks)
        self.initialize()
    def forward(self, x):
        x=self.blocks(x)
        return x
    def initialize(self):  # 初始化模型参数
        for m in self.blocks.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight.data, math.sqrt(3))
                if m.bias!= None:
                    torch.nn.init.constant_(m.bias.data, 0.01)
class Decoder(nn.Module):
    def __init__(self, out_channel_N):
        super(Decoder, self).__init__()
        blocks=[nn.ConvTranspose1d(out_channel_N, int(out_channel_N), 3, dilation=2,stride=2, padding=1, output_padding=0),
                GDN(int(out_channel_N), inverse=True),
                nn.ConvTranspose1d(int(out_channel_N), out_channel_N, 3,dilation=2, stride=2, padding=1, output_padding=0),
                GDN(out_channel_N, inverse=True),
                nn.ConvTranspose1d(int(out_channel_N), out_channel_N, 3, dilation=2,stride=2, padding=2, output_padding=0),
                GDN(out_channel_N, inverse=True),
                nn.ConvTranspose1d(int(out_channel_N), out_channel_N, 3, dilation=2,stride=2, padding=2, output_padding=1),
                GDN(out_channel_N, inverse=True),
                nn.Conv1d( out_channel_N,1, kernel_size=5, dilation=3,stride=1, padding=1)]
        self.blocks = nn.Sequential(*blocks)
        self.initialize()
    def forward(self, z):
        x=self.blocks(z)
        #print(x.shape)
        return x
    def initialize(self):  # 初始化模型参数
        for m in self.blocks.modules():
            if isinstance(m, nn.ConvTranspose1d):
                torch.nn.init.xavier_normal_(m.weight.data, math.sqrt(3))
                if m.bias!= None:
                    torch.nn.init.constant_(m.bias.data, 0.01)
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight.data, math.sqrt(3))
                if m.bias!= None:
                    torch.nn.init.constant_(m.bias.data, 0.01)