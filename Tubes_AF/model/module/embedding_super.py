import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
from torch import Tensor
import numpy as np
# todo to modify and add temporal component
class PatchembedSuper(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False):
        super(PatchembedSuper, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = embed_dim
        self.scale = scale

    # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        if self.scale:
            return x * self.sampled_scale
        return x
    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops


class SparseTubesTokenizerSuper(nn.Module):
    def __init__(self, embed_dim, kernel_sizes, strides, offsets, scale=False):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.offsets = offsets
        # self.patch_size = kernel_sizes
        self.super_embed_dim = embed_dim
        self.scale = scale
        self.num_patches = 0 #todo devo capire ce numero restitutie e se e' fondamentale restituirlo

        self.conv_proj_weight = nn.Parameter(
            torch.empty((embed_dim, 3, *self.kernel_sizes[0])).normal_(), requires_grad=True
        )

        self.register_parameter("conv_proj_weight", self.conv_proj_weight)

        self.conv_proj_bias = nn.Parameter(torch.zeros(len(self.kernel_sizes), embed_dim), requires_grad=True)
        self.register_parameter("conv_proj_bias", self.conv_proj_bias)

        # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.conv_proj_weight[:sample_embed_dim, ...]
        self.sampled_bias = self.conv_proj_bias[..., :self.sample_embed_dim]
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim

    def forward(self, x: Tensor) -> Tensor:
        n, c, t, h, w = x.shape  # CTHW
        tubes = []
        if t !=1:
            for i in range(len(self.kernel_sizes)):
                if i == 0:
                    weight = self.sampled_weight#self.conv_proj_weight
                else:
                    weight = F.interpolate(self.sampled_weight, self.kernel_sizes[i], mode="trilinear")#F.interpolate(self.conv_proj_weight, self.kernel_sizes[i], mode="trilinear")

                tube = F.conv3d(
                    x[:, :, self.offsets[i][0]:, self.offsets[i][1]:, self.offsets[i][2]:],
                    weight,
                    bias=self.sampled_bias[i], #self.conv_proj_bias[i],
                    stride=self.strides[i],
                )

                tube = tube.reshape((n, self.sample_embed_dim, -1))

                tubes.append(tube)
        else:
            weight = F.interpolate(self.sampled_weight, self.kernel_sizes[-1], mode="trilinear")#
            tube = F.conv3d(
                x[:, :, self.offsets[-1][0]:, self.offsets[-1][1]:, self.offsets[-1][2]:],
                weight,
                bias=self.sampled_bias[-1],  # self.conv_proj_bias[i],
                stride=self.strides[-1],
            )

            tube = tube.reshape((n, self.sample_embed_dim, -1))

            tubes.append(tube)

        x = torch.cat(tubes, dim=-1) #io ho 20 tokens in meno... 539 vs 559. Come mai? le dimensioni kernel stride e inpu sono giuste
        # forse hanno riportato loro il numero sbagliato
        x = x.permute(0, 2, 1).contiguous()

        if self.scale:
            return x * self.sampled_scale

        return x

    def calc_sampled_param_num(self):
        return self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
            total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops