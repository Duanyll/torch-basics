import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv2d3x3s1p1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

def conv2d1x1s1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channel)
        self.act1 = nn.SiLU()
        self.conv1 = conv2d3x3s1p1(in_channel, out_channel)
        self.proj_temb = nn.Linear(temb_dim, out_channel)
        self.norm2 = nn.GroupNorm(32, out_channel)
        self.act2 = nn.SiLU()
        self.conv2 = conv2d3x3s1p1(out_channel, out_channel)

        if in_channel != out_channel:
            self.skip = conv2d1x1s1(in_channel, out_channel)

    def forward(self, x, temb):
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out += self.proj_temb(temb)[:, :, None, None]
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)

        if x.shape[1] != out.shape[1]:
            x = self.skip(x)
        out += x
        return out


class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_size, dim):
        super().__init__()
        self.emb_size = emb_size
        self.dim = dim
        
        position = torch.arange(0, emb_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        emb = torch.zeros(emb_size, dim)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('emb', emb)
        
    def forward(self, t):
        return self.emb[t % self.emb_size]


class UNet(nn.Module):
    def __init__(self,
                 in_channel=3,
                 emb_dim=128,
                 channels_scales=[1, 1, 2, 2, 4, 4],
                 blocks_per_scale=[2, 2, 2, 2, 2, 2],
                 blocks_mid=1,
                 T=1000):
        super().__init__()
        temb_dim = emb_dim * 4
        self.t_encoder = nn.Sequential(
            SinusoidalEmbedding(T, emb_dim),
            nn.Linear(emb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
        )
        channel = emb_dim
        self.in_proj = conv2d3x3s1p1(in_channel, channel)
        channel_stack = [channel]
        self.downs = nn.ModuleList()
        for i, (blocks, scale) in enumerate(zip(blocks_per_scale, channels_scales)):
            for _ in range(blocks):
                self.downs.append(ResBlock(channel, emb_dim * scale, temb_dim))
                channel = emb_dim * scale
                channel_stack.append(channel)
            if i != len(channels_scales) - 1:
                self.downs.append(nn.AvgPool2d(2))
                channel_stack.append(channel)
        
        self.mid = nn.ModuleList([ResBlock(channel, channel, temb_dim) for _ in range(blocks_mid)])
        
        self.ups = nn.ModuleList()
        for i, (blocks, scale) in enumerate(zip(reversed(blocks_per_scale), reversed(channels_scales))):
            for _ in range(blocks + 1):
                self.ups.append(ResBlock(channel + channel_stack.pop(), emb_dim * scale, temb_dim))
                channel = emb_dim * scale
            if i != len(channels_scales) - 1:
                self.ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
                
        self.out_proj = nn.Sequential(
            nn.GroupNorm(32, channel),
            nn.SiLU(),
            conv2d3x3s1p1(channel, in_channel),
        )
        
    def forward(self, x, t):
        temb = self.t_encoder(t)
        x = self.in_proj(x)
        
        xs = [x]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, temb)
            else:
                x = layer(x)
            xs.append(x)
            
        for layer in self.mid:
            x = layer(x, temb)
        
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = torch.cat([x, xs.pop()], dim=1)
                x = layer(x, temb)
            else:
                x = layer(x)
        
        return self.out_proj(x)