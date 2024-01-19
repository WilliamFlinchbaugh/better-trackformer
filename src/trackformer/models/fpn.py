import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import operator
from functools import reduce

# Adapted from https://github.com/ShuLiu1993/PANet/blob/master/lib/modeling/FPN.py
class FPN(nn.Module):
    def __init__(self, fpn_lateral_dims: list, fpn_dim: int, use_panet: bool):
        super().__init__()
        self.use_panet = use_panet
        self.num_modules = len(fpn_lateral_dims) - 1 # number of topdown and bottomup modules
        self.num_stages = len(fpn_lateral_dims) # number of backbone stages
        self.fpn_lateral_dims = fpn_lateral_dims
        self.fpn_dim = fpn_dim
        
        # Top-down
        self.conv_top = nn.Conv2d(self.fpn_lateral_dims[0], self.fpn_dim, 1, 1, 0)
        self.topdown_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()
        
        for i in range(self.num_modules):
            self.topdown_lateral_modules.append(
                topdown_lateral_module(fpn_dim, fpn_lateral_dims[i+1])
            )
        
        for i in range(self.num_stages):
            self.posthoc_modules.append(
                nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
            )
        
        # Bottom-up
        if self.use_panet:
            self.bottomup_conv1_modules = nn.ModuleList()
            self.bottomup_conv2_modules = nn.ModuleList()
            
            for i in range(self.num_modules):
                self.bottomup_conv1_modules.append(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1)
                )
                self.bottomup_conv2_modules.append(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
                )
            
        self._init_weights()
        
    def forward(self, body_out):
        
        # convert to list
        conv_body_blobs = []
        for i in range(self.num_stages):
            conv_body_blobs.append(body_out[str(i)])
        
        # topdown lateral
        fpn_inner_blobs = [self.conv_top(conv_body_blobs[-1])] 
        for i in range(self.num_modules):
            fpn_inner_blobs.append(
                self.topdown_lateral_modules[i](fpn_inner_blobs[-1], conv_body_blobs[-(i+2)])
            )
        
        # topdown posthoc
        fpn_output_blobs = []
        for i in range(self.num_stages):
            fpn_output_blobs.append(
                self.posthoc_modules[i](fpn_inner_blobs[i])
            )
        
        # bottomup if using panet
        if self.use_panet:
            panet_output_blobs = [fpn_output_blobs[-1]]
            for i in range(2, self.num_stages + 1):
                tmp = self.bottomup_conv1_modules[i-2](panet_output_blobs[0])
                tmp = tmp + fpn_output_blobs[self.num_stages - i]
                tmp = self.bottomup_conv2_modules[i-2](tmp)
                panet_output_blobs.insert(0, tmp)
            
            out = {str(i): panet_output_blobs[i] for i in range(len(panet_output_blobs))}
            return out
        
        out = {str(i): fpn_output_blobs[i] for i in range(len(fpn_output_blobs))}
        return out
    
    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                XavierFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for child_m in self.children():
            if (not isinstance(child_m, nn.ModuleList) or
                not isinstance(child_m[0], topdown_lateral_module)):
                # topdown_lateral_module has its own init method
                child_m.apply(init_func)


class topdown_lateral_module(nn.Module):
    """Add a top-down lateral module."""
    def __init__(self, dim_in_top, dim_in_lateral):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        conv = self.conv_lateral
        XavierFill(conv.weight)
        if conv.bias is not None:
            init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        # Lateral 1x1 conv                                                   
        lat = self.conv_lateral(lateral_blob)
        # Top-down 2x upsampling
        td = F.upsample(top_blob, size=lat.size()[2:], mode='bilinear', align_corners=False)
        # Sum lateral and top-down
        return lat + td
    
def XavierFill(tensor):
    """Caffe2 XavierFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_in = size / tensor.shape[0]
    scale = math.sqrt(3 / fan_in)
    return init.uniform_(tensor, -scale, scale)