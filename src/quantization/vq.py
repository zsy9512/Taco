# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Residual vector quantizer implementation."""
import math
import typing as tp
from dataclasses import dataclass
from dataclasses import field

import torch
from torch import nn
from src.quantization.core_vq import ResidualVectorQuantization


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)



class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
            self,
            dimension: int=256,
            codebook_dim: tp.Optional[int]=None,
            n_q: int=8,
            bins: int=1024):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.codebook_dim=codebook_dim
        
        self.vq= ResidualVectorQuantization(
            dim=self.dimension,
            codebook_dim=self.codebook_dim,
            codebook_size=self.bins,
            num_quantizers=self.n_q)

    def forward(self,x: torch.Tensor) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            sample_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        quantized_st, quantized,loss= self.vq(x, n_q=self.n_q)
        return quantized_st, quantized,loss


    def encode(self,
               x: torch.Tensor,   
               st: tp.Optional[int]=None) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        st = st or 0
        codes= self.vq.encode(x, n_q=self.n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.vq.decode(codes)
        return quantized

    
    
class GroupVectorQuantizer(torch.nn.Module):
    def __init__(self,             
                dimension: int=256,
                codebook_dim: tp.Optional[int]=None,
                n_q: int=8,
                bins: int=256,
                n_codes_group: int=2):
        super(GroupVectorQuantizer, self).__init__()
        assert codebook_dim % n_codes_group == 0
        self.n_codes_group= n_codes_group
        dimension=dimension//n_codes_group
        
        codebook_dim=codebook_dim//n_codes_group
        self.codebook_dim=codebook_dim
        
        self.codebook_loss_lambda=1
        self.commitment_loss_lambda=0.2
        if n_q%2==0:
            nq=[n_q//2,n_q//2]
        else:
            nq=[1+(n_q//2),n_q//2]
        
        for i in range(self.n_codes_group):
            #print(dimension, codebook_dim, n_q,bins)
            
            setattr(self, f"quantizer_{i}", ResidualVectorQuantizer(dimension, codebook_dim, nq[i],bins))

    def for_one_step(self, xin, idx):
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, 128)
        x = torch.split(x, 128 // self.h.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        if idx == 0:
            for _x, m in zip(x, self.quantizer_modules):
                _z_q, _min_indicies = m(_x)
                #_z_q_st = _z_q.clone()
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies
        else:
            for _x, m in zip(x, self.quantizer_modules2):
                _z_q, _min_indicies = m(_x)
                z_q.append(_z_q)
                min_indicies.append(_min_indicies)  #B * T,
            z_q = torch.cat(z_q, -1).reshape(xin.shape)
            # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
            loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
                + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
            z_q = xin + (z_q - xin).detach()
            z_q = z_q.transpose(1, 2)
            return z_q, loss, min_indicies

    def forward(self, xin):
        #B, C, T
        x = xin
        x = torch.split(x, 64,dim=1)
        z_q_st = []
        z_q = []
        vqloss=0
        for i in range(self.n_codes_group):
            e=x[i]
            z_q_st_,z_q_,loss=getattr(self, f"quantizer_{i}")(e)
            z_q_st.append(z_q_st_)
            z_q.append(z_q_)
            vqloss+=loss
        z_q = torch.cat(z_q, 1)
        z_q_st = torch.cat(z_q_st, 1)
        return z_q_st,z_q,vqloss
    def embed(self, x):
        #idx: N, T, 4
        #print('x ', x.shape)
        quantized_out = torch.tensor(0.0, device=x.device)
        x = torch.split(x, 1, 2)  # split, 将最后一个维度分开, 每个属于一个index group
        #print('x.shape ', len(x),x[0].shape)
        for i in range(self.residul_layer):
            ret = []
            if i == 0:
                for j in range(self.n_code_groups):
                    q = x[j]
                    embed = self.quantizer_modules[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                #print(ret.shape)
                quantized_out = quantized_out + ret
            else:
                for j in range(self.n_code_groups):
                    q = x[j + self.n_code_groups]
                    embed = self.quantizer_modules2[j]
                    q = embed.embedding(q.squeeze(-1))
                    ret.append(q)
                ret = torch.cat(ret, -1)
                quantized_out = quantized_out + ret
        return quantized_out.transpose(1, 2)  #N, C, T
