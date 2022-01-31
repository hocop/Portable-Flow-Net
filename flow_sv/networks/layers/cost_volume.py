
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CostVolume(nn.Module):
    '''
    Cost volume layer used in PWCNet and FlowNetC.
    Implementation uses only reshape and transpose operations
    Can be converted to onnx and openvino
    '''
    def __init__(
        self,
        input_dim,
        output_dim,
        window_h,
        window_w=None,
        export_mode=True,
        dilation=1,
        **kwargs,
    ):
        '''
        
        '''
        super().__init__()
        assert window_h % 2 == 1, 'window_h must be not divisible by 2'
        assert window_w is None or window_w % 2 == 1, 'window_w must be not divisible by 2'
        self.window_h = window_h
        self.window_w = window_w or window_h
        self.output_dim = output_dim
        self.export_mode = export_mode
        if isinstance(input_dim, int):
            input_dim = (input_dim, input_dim)
        assert len(input_dim) == 2
        self.conv = nn.Conv2d(
            input_dim[0],
            input_dim[1] * output_dim,
            kernel_size=(self.window_h, self.window_w),
            padding=(self.window_w // 2 * dilation, self.window_h // 2 * dilation),
            dilation=dilation,
            **kwargs,
        )

    def forward(self, feats_target, feats_source):
        lambda_p = self.conv(feats_source)

        if not self.export_mode:
            lambda_p = rearrange(lambda_p, 'n (u c) h w -> n c u h w', u=self.output_dim)
            output = torch.einsum('n c h w, n c u h w -> n u h w', feats_target, lambda_p)
        else:
            lambda_p = rearrange(lambda_p, 'n (u c) h w -> (n h w) c u', u=self.output_dim)
            feats_target = rearrange(feats_target, 'n c h w -> (n h w) 1 c')
            output = feats_target @ lambda_p
            output = rearrange(
                output, '(n h w) 1 u -> n u h w',
                n=feats_source.shape[0],
                h=feats_source.shape[2],
                w=feats_source.shape[3],
            )

        return output
