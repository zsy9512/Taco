'''
Author: shengyu
Date: 2024-03-29 09:11:02
LastEditors: shengyu
LastEditTime: 2024-06-26 10:14:48
Description: 请填写简介
'''
import math
import random
import torch
import numpy as np
import torch.nn as nn
from src.modules.codec import Encoder,Decoder
import torch.nn.functional as F
# Generator
class TouchStream(nn.Module):
    def __init__(self,
                 input_dim,
                 quantizer):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.quantizer =quantizer
        self.decoder = Decoder(input_dim) 
    def forward(self, x,is_quan_on_val=1):
        e = self.encoder(x)
        if is_quan_on_val:
            #print(e.shape)
            quantized_st,quantized,commit_loss= self.quantizer(e)
            o = self.decoder(quantized_st)
            commit_loss=0.5*self.compute_loss(quantized,e)+0.5*commit_loss
        else:
            quantized=e
            o = self.decoder(e)
            commit_loss=torch.tensor(0.0)
        return o, commit_loss
    def encode(self, x, st=None):
        e = self.encoder(x)
        if st is None:
            st = 0
        codes = self.quantizer.encode(e,st)
        return codes
    def decode(self, codes):
        quantized = self.quantizer.decode(codes)
        o = self.decoder(quantized)
        return o
    def compute_loss(self,z_q_x, z_e_x,beta=0.2):
        #print(z_q_x.shape,z_e_x.shape)
        loss_vq = F.smooth_l1_loss(z_q_x, z_e_x.detach(),beta=0.1)
        # Commitment objective
        loss_commit = F.smooth_l1_loss(z_e_x, z_q_x.detach(),beta=0.1)
        loss=loss_commit  + beta* loss_vq
        return loss