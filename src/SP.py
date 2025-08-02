from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from torch import Tensor
from typing import Union
from functools import partial
from timm.models.layers import DropPath
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.convolutions import Convolution

from timm.layers.helpers import to_3tuple

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class Enc(nn.Module):
    def __init__(self, dim, drop_path=0.1, exp=2):
        super().__init__()
        in_channels = dim
        out_channels = exp * dim
        self.dwconv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, groups=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = input + self.drop_path(x)
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 kernel_size=3,
                 drop_rate=0.):
        super(MLP, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)  
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size, 1, (kernel_size-1)//2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size, 1, (kernel_size-1)//2)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class AttentionConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 kernel=7,
                 num_heads=8):
        super(AttentionConv, self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.BatchNorm3d(in_channels)

        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, kernel, 1, 1))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, kernel, 1))
        self.padding = kernel // 2

    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,d,h,w
        d, h, w = x_shape[2], x_shape[3], x_shape[4]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])  # n,c_inter,d,h,w -> n,heads,c_inner//heads,dhw
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, d, h, w])
        return x

    def forward(self, x):
        x = self.norm(x)
        x1 = F.conv3d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=(self.padding, 0, 0))
        x1 = self._act_dn(x1)
        x1 = F.conv3d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=(self.padding, 0, 0))
        x3 = F.conv3d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=(0, self.padding, 0))
        x3 = self._act_dn(x3)
        x3 = F.conv3d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1, padding=(0, self.padding, 0))
        x = x1 + x3
        return x

class MCA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=7,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.):
        super(MCA, self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = AttentionConv(
            in_channels_l,
            out_channels_l,
            inter_channels=64,
            kernel=kernel,
            num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x))
        return x

class conv_atten(nn.Module):
    def __init__(self, in_channels=4, depths=[2,2,1,1], dims =[32, 64, 128, 256],
                   out_indices=[0,1,2,3], kernel=7, stride=1):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        if stride == 1:
            stem = nn.Sequential(
                    nn.Conv3d(in_channels=in_channels, out_channels=dims[0], kernel_size=3, stride=1, padding=1),
                    LayerNorm(normalized_shape=dims[0], eps=1e-6, data_format='channels_first'))
        else:
            stem = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=dims[0], kernel_size=7, stride=2, padding=3),
                LayerNorm(normalized_shape=dims[0], eps=1e-6, data_format='channels_first'))

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(normalized_shape=dims[i], eps=1e-6, data_format='channels_first'),
                                             nn.Conv3d(in_channels=dims[i],out_channels=dims[i+1], kernel_size=3, stride=2, padding=1))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            if i <= 2 :
                stage = nn.Sequential(*[Enc(dim=dims[i], drop_path=0.1 * j) for j in range(depths[i])])

            else:
                stage = nn.Sequential(*[MCA(in_channels=dims[i], out_channels=dims[i], kernel=kernel, num_heads=8, drop_rate=0., drop_path_rate=0.1) for j in range(depths[i])])
            self.stages.append(stage)
        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)


class ASA(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=2, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels * exp_r, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(in_channels=exp_r * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.res_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)


    def forward(self, x_residual):
        input_x = x_residual
        input_x = self.dwconv1(input_x)
        input_x = self.act(self.conv2(self.norm(input_x)))
        input_x = self.conv3(input_x)
        input_x = input_x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            input_x = self.gamma * input_x
        input_x = input_x.permute(0, 4, 1, 2, 3)
        x_residual = self.res_conv(x_residual)
        return input_x + x_residual


class SP(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, depths=[2,2,1,1], dims=[32,64,128,256], norm_name: Union[Tuple,str]="instance"):
        super().__init__()
        self.out_indice = []
        for i in range(len(dims)):
            self.out_indice.append(i)
        self.enc = conv_atten(in_channels=in_channels, depths=depths, dims=dims, out_indices=self.out_indice,
                                   kernel=7)

        self.stem = nn.Conv3d(in_channels=in_channels, out_channels=dims[0], kernel_size=3, stride=1, padding=1)

        self.skip1 = ASA(in_channels=dims[0], out_channels=dims[1])
        self.skip2 = ASA(in_channels=dims[1], out_channels=dims[2])
        self.skip3 = ASA(in_channels=dims[2], out_channels=dims[3])
        self.skip4 = ASA(in_channels=dims[3], out_channels=2 * dims[3])

        self.dec4 = UnetrUpBlock(spatial_dims=3, in_channels=2 * dims[3], out_channels=dims[3],
                                 kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
        self.dec3 = UnetrUpBlock(spatial_dims=3, in_channels=dims[3], out_channels=dims[2],
                                 kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
        self.dec2 = UnetrUpBlock(spatial_dims=3, in_channels=dims[2], out_channels=dims[1],
                                 kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
        self.dec1 = UnetrUpBlock(spatial_dims=3, in_channels=dims[1], out_channels=dims[0],
                                 kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)

        self.out = UnetOutBlock(spatial_dims=3, in_channels=dims[0], out_channels=out_channels)

    def forward(self, x):
        outs = self.enc(x)
        enc1 = self.stem(x)
        enc2 = self.skip1(outs[0])
        enc3 = self.skip2(outs[1])
        enc4 = self.skip3(outs[2])
        bottle = self.skip4(outs[3])

        dec3 = self.dec4(bottle, enc4)
        dec2 = self.dec3(dec3, enc3)
        dec1 = self.dec2(dec2, enc2)
        dec0 = self.dec1(dec1, enc1)
        out = self.out(dec0)

        return out
