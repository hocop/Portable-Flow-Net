import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange

from .layers import cost_volume, packnet, heads


class MidasFlow(nn.Module):
    def __init__(
        self,
        n_inputs,
    ):
        super().__init__()

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        # self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

        # Modify input convolution layer
        if n_inputs > 1:
            self.midas.pretrained.layer1[0].n_inputs = 3 * n_inputs
            # Remember weights of the input layer
            input_sd = self.midas.pretrained.layer1[0].state_dict()
            # Set the correct weights to the input layer
            input_sd['weight'] = torch.cat(
                [input_sd['weight']] * n_inputs,
                dim=1,
            ) / n_inputs
            noise = torch.randn_like(input_sd['weight']) * input_sd['weight'].std()
            input_sd['weight'] = (input_sd['weight'] + noise) / np.sqrt(2)
            self.midas.pretrained.layer1[0].weight = nn.Parameter(input_sd['weight'])

        self.midas.scratch.output_conv = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # self.midas.scratch.output_conv = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.midas.scratch.output_conv.register_forward_hook(self.hook)
        self.hook_result = None

        # Decoder
        self.flow_decoder = heads.PredictFlow(32)
        # self.mask_decoder = heads.PredictMask(32)

    def forward(self, images):
        '''
        input:
        - images: torch.tensor of shape [batch, n * 3, H, W]
            RGB images from multiple cameras concatenated together
        returns: tuple
        - flow: torch.tensor of shape [batch, 2, H, W]
        - motion: torch.tensor of shape [batch, 1, H, W]
            3d motion of each pixel relative to scene
        '''
        # Features
        self.midas(images)
        features = self.hook_result
        self.hook_result = None

        # Decoder
        flow = self.flow_decoder(features)
        # mask = self.mask_decoder(features)
        mask = None

        return flow, mask

    def hook(self, module, input, output):
        self.hook_result = output