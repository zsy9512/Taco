'''
Author: shengyu
Date: 2024-04-19 16:09:39
LastEditors: shengyu
LastEditTime: 2024-07-23 22:37:22
Description: loss functions
'''
import torch
import torch.nn.functional as F 
from pathlib import Path
from random import random
import numpy as np
from einops import rearrange
from vibromaf.metrics import stsim,spqi
def reconstruction_loss(x, G_x, args=None, eps=1e-7):
    L=F.mse_loss(x,G_x)
    return L
def criterion( x, G_x,args=None):
    rec_loss =reconstruction_loss(x, G_x)
    psnr=10*torch.log10(4/rec_loss)
    deg=G_x.detach().cpu().squeeze(1).numpy()
    ref=x.detach().cpu().squeeze(1).numpy()
    st=torch.tensor(stsim.st_sim(deg,ref)).to("cuda")
    return rec_loss,psnr,torch.mean(st),torch.std(st)
def loss(inputs,#输入信号
         reconstructions,#输出信号
        args=None):
    #rec_loss =reconstruction_loss(inputs.contiguous(),reconstructions.contiguous())
    rec_loss = F.smooth_l1_loss(inputs.contiguous(),reconstructions.contiguous() ,beta=0.1)
    psnr =10*torch.log10(4/reconstruction_loss(inputs.contiguous(),reconstructions.contiguous()))
    return rec_loss,psnr
def criterion_uq(z_e_x,z_q_x, x, G_x,args):
    codebook_loss=z_q_x
    rec_loss = reconstruction_loss(x.contiguous(), G_x.contiguous(), args)
    psnr=10*torch.log10(((x.max()-x.min()).abs())**2/rec_loss)
    total_loss = args.LAMBDA_REC *rec_loss+args.LAMBDA_COM * codebook_loss
    return total_loss, args.LAMBDA_REC *rec_loss, args.LAMBDA_COM * codebook_loss,psnr
def loss_uq(z_e_x,
           z_q_x,
           inputs,#输入信号
           reconstructions,#输出信号
           beta=0.25,
           args=None):
    codebook_loss=z_q_x
    rec_loss = reconstruction_loss(inputs.contiguous(),reconstructions.contiguous(), args)
    total_loss = args.LAMBDA_REC *rec_loss+args.LAMBDA_COM * codebook_loss
    return total_loss, args.LAMBDA_REC *rec_loss, args.LAMBDA_COM * codebook_loss,torch.tensor(0.0)
def Wfft_loss(x, G_x,weight):
    mag_x=abs(1 / torch.sqrt(torch.tensor(512)) * torch.fft.fft(x,1024))[:,:,0:512]
    x_dB=torch.log10(mag_x)
    mag_Gx=abs(1 / torch.sqrt(torch.tensor(512)) * torch.fft.fft(G_x,1024))[:,:,0:512]
    Gx_dB=torch.log10(mag_Gx)
    #print(mag_x.shape,mag_Gx.shape)
    L=torch.mean(F.l1_loss(x_dB,Gx_dB,reduce="none")*weight)
    #L=F.l1_loss(x_dB,Gx_dB,reduce="mean")
    #print(L)
    return L
def fft_loss(x, G_x,weight):
    mag_x=abs(1 / torch.sqrt(torch.tensor(512)) * torch.fft.fft(x,1024))[:,:,0:512]
    x_dB=torch.log10(mag_x)
    mag_Gx=abs(1 / torch.sqrt(torch.tensor(512)) * torch.fft.fft(G_x,1024))[:,:,0:512]
    Gx_dB=torch.log10(mag_Gx)
    #print(mag_x.shape,mag_Gx.shape)
    L=F.l1_loss(x_dB,Gx_dB,reduce="mean")
    #print(L)
    return L