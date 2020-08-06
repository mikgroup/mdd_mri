#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import util.complex as cp

'''
Contains defintions for models
'''

def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def torch_fftshift(im):
    t = len(im.shape)
    n = int(np.floor(im.shape[t-3]/2))
    m = int(np.floor(im.shape[t-2]/2))
    P_torch1 = roll(roll(im,m,t-2),n,t-3)
    return P_torch1

def torch_ifftshift(im):
    t = len(im.shape)
    n = int(np.ceil(im.shape[t-3]/2))
    m = int(np.ceil(im.shape[t-2]/2))
    P_torch1 = roll(roll(im,m,t-2),n,t-3)
    return P_torch1

def maps_forw(img, maps):
    return cp.zmul(img[:,None,:,:,:], maps)

def maps_adj(cimg, maps):
    return torch.sum(cp.zmul(cp.zconj(maps), cimg), 1, keepdim=False)

def fft_forw(x, ndim=2):
    return torch_fftshift(torch.fft(torch_ifftshift(x), signal_ndim=ndim, normalized=True))

def fft_adj(x, ndim=2):
    return torch_fftshift(torch.ifft(torch_ifftshift(x), signal_ndim=ndim, normalized=True))

def mask_forw(y, mask):
    return y * mask[:,None,:,:,None]

def sense_forw(img, maps, mask):
    return mask_forw(fft_forw(maps_forw(img, maps)), mask)

def shuffle_for_fft(x):
    dims = x.size()
    x = x.reshape((1, dims[1]//2, 2, dims[2], dims[3])) # Reshaping to seperate real and imaginary for each coil
    x = x.permute((0, 1, 3, 4, 2)) # Permuting so last channel is real/imaginary for pytorch fft
    return x

def shuffle_after_fft(x):
    x = x.permute((0, 1, 4, 2, 3)) # Permuting so that real/imaginary dimension is next to coil dimension
    dims = x.size()
    x = x.reshape((1, dims[1]*2, dims[3], dims[4])) #Reshaping to combine real/imaginary dimension and coil dimension
    return x

class FFTInterpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.up_size = size
        
    def forward(self, x):
        input_size = tuple(x.size()[2:4])
        total_padding = [goal-current for goal, current in zip(self.up_size, input_size)]
        padding_tup = (0, 0) + sum([(dim//2,dim//2+dim%2) for dim in total_padding[::-1]], tuple()) #padding: 0, 0, left, right, top, bottom
        
        x = shuffle_for_fft(x)
        x = fft_forw(x)
        x = nn.functional.pad(x, padding_tup)
        x = fft_adj(x)
        x = shuffle_after_fft(x)
        
        return x

class SimpleSenseModel(torch.nn.Module):
    def __init__(self, mask):
        super(SimpleSenseModel, self).__init__()
        self.mask = mask
        
    def forward(self, x):
        return mask_forw(fft_forw(x), self.mask)

    def adjoint(self, y):
        return fft_adj(mask_forw(y, self.mask))
        
class DecoderForward(torch.nn.Module):
    def __init__(self, decoder, mask):
        super(DecoderForward, self).__init__()
        self.decoder = decoder
        self.A = SimpleSenseModel(mask)

    def forward(self, x):
        out = self.decoder(x)
        out = shuffle_for_fft(out)
        out = self.A(out)
        out = shuffle_after_fft(out)
        return out

class MDD(torch.nn.Module):
    def __init__(self, depth=5, k_out=3, k=256, filter_size=1, upsample_mode='bilinear', up_size=2, net_input=None):
        super(MDD, self).__init__()
        
        # Initialize parameters
        self.depth = depth
        self.k_out = k_out
        self.k = k.append(k[-1]) if type(k) is list else [k]*(self.depth+1) # Extend by one to account for extra final layer
        self.filter_size = filter_size.append(filter_size[-1]) if type(filter_size) is list else [filter_size]*len(self.k)
        self.upsample_mode = upsample_mode
        
        # Initialize upsampling sizes
        if type(up_size) is not list: # Scale factor given
            one_dim_up_size = [16*(up_size**i) for i in range(1, self.depth+1)]
            self.up_size = list(zip(one_dim_up_size, one_dim_up_size))
            print(self.up_size)
        else:
            self.up_size = up_size
        
        # Assertions to make sure inputted parameters match expectations
        assert self.depth+1 == len(self.k) == len(self.filter_size)
        assert self.depth == len(self.up_size)
             
        # Create layers according to the following:
        # Conv -> ReLU -> Batchnorm -> Upsample (x Depth)
        # Conv -> ReLU -> BatchNorm -> Conv (x 1)
        
        layer_list = []
        for i in range(self.depth):
            if self.upsample_mode == "fft_pad":
                upsample_layer = nn.FFTInterpolate(self.up_size[i])
            else:
                upsample_layer = nn.Upsample(size=self.up_size[i], mode=self.upsample_mode)
            to_pad = int((self.filter_size[i] - 1) / 2)
            padder = nn.ReflectionPad2d(to_pad) # Padding so that Conv layer output has expected spatial output
            layer_list.append(nn.Sequential(
                                   padder,
                                   nn.Conv2d(self.k[i], self.k[i+1], self.filter_size[i], bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(self.k[i+1], affine=True),
                                   upsample_layer))
        
        to_pad = int((self.filter_size[self.depth] - 1) / 2)
        layer_list.append(nn.Sequential(
                               nn.ReflectionPad2d(to_pad),
                               nn.Conv2d(self.k[self.depth], self.k[self.depth], self.filter_size[self.depth], bias=False),
                               nn.ReLU(),
                               nn.BatchNorm2d(self.k[self.depth], affine=True),
                               nn.ReflectionPad2d(to_pad),
                               nn.Conv2d(self.k[self.depth], self.k_out, self.filter_size[self.depth], bias=False)))
        
        self.layers = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.layers(x)
    

