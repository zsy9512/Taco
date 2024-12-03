# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Core vector quantization implementation."""
import typing as tp

import torch
import torch.nn.functional as F
from einops import rearrange,repeat
from torch import nn
import numpy as np
import vqtorch
def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d

class VQEmbedding(nn.Module):
    def __init__(self,dim,codebook_size,codebook_dim):
        super().__init__()
        self.nu=0.0
        self.codebook_size=codebook_size
        self._codebook_dim: int = default(codebook_dim, dim)
        requires_projection = self._codebook_dim != dim
        self.project_in = (nn.Linear(dim, self._codebook_dim)
                           if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(self._codebook_dim, dim)
                            if requires_projection else nn.Identity())
        self.embedding = nn.Embedding(self.codebook_size, self._codebook_dim)
        self.embedding.weight.data.uniform_(-1./self.codebook_size, 1./self.codebook_size)    
    def forward(self, input):
        x = rearrange(input, "b d n -> b n d")
        x = self.project_in(x)        
        latents = vq(x, self.embedding.weight)
        return latents
    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x
    def straight_through(self, z_e_x):

        z_e_x_ =rearrange(z_e_x, "b d n -> b n d")
        z_e_x_ = self.project_in(z_e_x_)     
        # xshape = z_e_x_.shape
        # z_e_x_ = self.preprocess(z_e_x_) 
        # self.init_embed_(z_e_x_)
        # z_e_x_=z_e_x_.view(xshape)
        #调用量化器
        
        
        #这里原始是用索引重新创造一个向量用于反向回归，训练码本，上面的旧变量z_q_x_做为st用于传递到后面的解码器，因为其使用了一个ctx.save_for_backward
        #其它版本代码里st是在生成用于回归的向量之后，用z_q_st=z+(z_q-z_e).detach代替
        z_q_x_, indices = vq_st(z_e_x_,self.embedding.weight.detach())
        #indices = vq(z_e_x_, self.embedding.weight).view(-1)
        # z_q_x_=self.project_out(z_q_x_)
        # z_q_x = rearrange(z_q_x_, "b n d -> b d n")
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar_=self.project_out(z_q_x_bar_)
        z_q_x_bar = rearrange(z_q_x_bar_, "b n d -> b d n")
        #所谓量化同步
        z_q_x=z_e_x+(z_q_x_bar-z_e_x).detach() +(self.nu * z_q_x_bar) + (-self.nu * z_q_x_bar).detach()
        #loss=self.compute_loss(z_q_x_bar,z_e_x)
        return z_q_x,z_q_x_bar
    def compute_loss(self,z_q_x, z_e_x,beta=0.25):
        #print(z_q_x.shape,z_e_x.shape)
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
        loss=loss_vq + beta* loss_commit
        return loss
    def decode(self,indices):
        #z_e_x_ = self.project_in(z_e_x_) 
        z_q_x_flatten = torch.index_select(self.embedding.weight,
        dim=0, index=indices.view(-1))
        z_q_x_ = z_q_x_flatten.view([indices.shape[0],indices.shape[1],self._codebook_dim])
        z_q_x_=self.project_out(z_q_x_)
        z_q_x = rearrange(z_q_x_, "b d n -> b n d")
        return z_q_x
        

from torch.autograd import Function
class VectorQuantization(Function):
    
    @staticmethod
    def forward(ctx, inputs, codebook,k=2):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            #将batch合并在一起计算
            inputs_flatten = inputs.reshape(-1, embedding_size)       
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            #Compute the distances to the codebook,分别计算每个输入向量与codebook中每个向量的距离
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
            # print(inputs_flatten.unsqueeze(1).shape,codebook.unsqueeze(1).shape)
            # F.cosine_similarity(codebook.unsqueeze(1),inputs_flatten.unsqueeze(1))
            #找到距离最小的码本索引
            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            #print(indices_flatten.shape,indices.shape)
            #topk-beamsearch
           # _, indices_flatten =distances.topk(k)
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):

        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        # affine_transform.update_running_statistics(indices_flatten, codebook)
        # codebook = affine_transform(codebook)       
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)
        #根据索引把对应的向量取出来，也就是量化后的值
        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)
        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)
            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

__all__ = [vq, vq_st]
class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers,shared=False, **kwargs):
        super().__init__() 
        if shared==True:
            self.emb=VQEmbedding(**kwargs)
            self.layers = nn.ModuleList([self.emb for _ in range(num_quantizers)])
        else:     
            self.layers = nn.ModuleList(
            [VQEmbedding(**kwargs) for _ in range(num_quantizers)])
        # self.layers = nn.ModuleList([VQEmbedding(dim=128,codebook_size=512,codebook_dim=128),
        #                              VQEmbedding(dim=128,codebook_size=256,codebook_dim=128),
        #                              VQEmbedding(dim=128,codebook_size=256,codebook_dim=128),
        #                              VQEmbedding(dim=128,codebook_size=128,codebook_dim=128)])
    def forward(self, x, n_q: tp.Optional[int]=None):
        quantized_out = 0.0
        quantized_out_st = 0.0
        loss=0.0
        residual = x
        n_q = n_q or len(self.layers)
        i=0
        for layer in self.layers[:n_q]:
            #z_e_x = layer(residual)            
            z_q_x_st, z_q_x=layer.straight_through(residual)
            #print(z_q_x_st,residual)
            res_loss=self.compute_loss(z_q_x,residual)
            #print(res_loss)
            loss+=res_loss
            residual = residual - z_q_x_st
            quantized_out = quantized_out + z_q_x
            quantized_out_st = quantized_out_st +z_q_x_st
        return  quantized_out_st,quantized_out,loss
    def encode(self,
               x: torch.Tensor,
               n_q: tp.Optional[int]=None,
               st: tp.Optional[int]=None) -> torch.Tensor:
        residual = x
        quantized_out = 0.0
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:  # 设置解码的起止layer
            indices = layer(residual)
            z_q_x = layer.decode(indices)    
            #print(z_q_x,residual)        
            residual = residual - z_q_x         
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)  
        #print("one test")    
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
    def compute_loss(self,z_q_x, z_e_x,beta=0.2):
        #print(z_q_x.shape,z_e_x.shape)
        loss_vq = F.smooth_l1_loss(z_e_x.detach(),z_q_x,beta=0.1)
        # Commitment objective
        loss_commit = F.smooth_l1_loss(z_e_x, z_q_x.detach(),beta=0.1)
        loss=loss_commit  +beta*loss_vq
        return loss