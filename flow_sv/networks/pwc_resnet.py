import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision

from .layers import heads, mobilenetv2
from .layers.cost_volume import CostVolume
from . import backends



class PWCResNet(nn.Module):
    CVOL_CHANNELS_IN = 16
    CVOL_CHANNELS_OUT = 64

    def __init__(self, max_displacement=4):
        """
        input: max_displacement --- maximum displacement in correlation layer after warpping
        """
        super().__init__()
        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # Load pretrained resnet
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.encoder.bn1.register_forward_hook(self.hook_layer0)
        self.encoder.layer1.register_forward_hook(self.hook_layer1)
        self.encoder.layer2.register_forward_hook(self.hook_layer2)
        self.encoder.layer3.register_forward_hook(self.hook_layer3)
        self.encoder.layer4.register_forward_hook(self.hook_layer4)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        # Decoder layer 4
        self.bottleneck_4 = nn.Conv2d(512, self.CVOL_CHANNELS_IN, 1)
        self.corr_4 = CostVolume(self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, max_displacement * 2 + 1)
        self.flow_4 = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(512 + self.CVOL_CHANNELS_OUT, self.CVOL_CHANNELS_OUT, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.CVOL_CHANNELS_OUT, 2, 1),
        )
        self.up_4 = nn.Sequential(
            self.upsample,
            nn.Conv2d(512 + self.CVOL_CHANNELS_OUT, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )
        # Decoder layer 3
        self.bottleneck_3 = nn.Conv2d(256, self.CVOL_CHANNELS_IN, 1)
        self.corr_3 = CostVolume(self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, max_displacement * 2 + 1)
        self.flow_3 = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(256 + 256 + self.CVOL_CHANNELS_OUT, self.CVOL_CHANNELS_OUT, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.CVOL_CHANNELS_OUT, 2, 1),
        )
        self.up_3 = nn.Sequential(
            self.upsample,
            nn.Conv2d(256 + 256 + self.CVOL_CHANNELS_OUT, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        # Decoder layer 2
        self.bottleneck_2 = nn.Conv2d(128, self.CVOL_CHANNELS_IN, 1)
        self.corr_2 = CostVolume(self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, max_displacement * 2 + 1)
        self.flow_2 = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(128 + 128 + self.CVOL_CHANNELS_OUT, self.CVOL_CHANNELS_OUT, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.CVOL_CHANNELS_OUT, 2, 1),
        )
        self.up_2 = nn.Sequential(
            self.upsample,
            nn.Conv2d(128 + 128 + self.CVOL_CHANNELS_OUT, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        # Decoder layer 1
        self.bottleneck_1 = nn.Conv2d(64, self.CVOL_CHANNELS_IN, 1)
        self.corr_1 = CostVolume(self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, max_displacement * 2 + 1)
        self.flow_1 = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Conv2d(64 + 64 + self.CVOL_CHANNELS_OUT, self.CVOL_CHANNELS_OUT, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(self.CVOL_CHANNELS_OUT, 2, 1),
        )
        self.up_1 = nn.Sequential(
            self.upsample,
            nn.Conv2d(64 + 64 + self.CVOL_CHANNELS_OUT, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        # Decoder layer 0
        self.bottleneck_0 = nn.Conv2d(64, 2, 1)
        self.corr_0 = CostVolume(2, 32, max_displacement * 2 + 1)

        # Final decoder
        self.flow_final = heads.PredictFlow(64 + 64 + 32)

    def forward(self, images):
        '''
        input:
        - images: torch.tensor of shape [batch, 2 * 3, H, W]
            Two RGB images concatenated together
        returns:
        - flow: torch.tensor of shape [batch, 2, H, W]
        '''
        image_t = images[:, :3] # target image
        image_s = images[:, 3:] # source image

        # Compute features from encoder
        self.encoder(image_t)
        feats_t0, feats_t1, feats_t2, feats_t3, feats_t4 = self.result_layer0, self.result_layer1, self.result_layer2, self.result_layer3, self.result_layer4
        self.reset_hooks()
        self.encoder(image_s)
        feats_s0, feats_s1, feats_s2, feats_s3, feats_s4 = self.result_layer0, self.result_layer1, self.result_layer2, self.result_layer3, self.result_layer4
        self.reset_hooks()

        # Decoder Layer 4
        cost_volume = self.corr_4(self.bottleneck_4(feats_t4), self.bottleneck_4(feats_s4))
        x = torch.cat([feats_t4, cost_volume], 1)
        flow_4 = self.flow_4(x) * 4
        up_4 = self.up_4(x)
        # Decoder Layer 3
        up_flow_4 = self.upsample(flow_4 * 2)
        warped_source = self.warp(self.bottleneck_3(feats_s3), up_flow_4)
        cost_volume = self.corr_3(self.bottleneck_3(feats_t3), warped_source)
        x = torch.cat([up_4, feats_t3, cost_volume], 1)
        flow_3 = up_flow_4 + self.flow_3(x) * 0.1
        up_3 = self.up_3(x)
        # Decoder Layer 2
        up_flow_3 = self.upsample(flow_3 * 2)
        warped_source = self.warp(self.bottleneck_2(feats_s2), up_flow_3)
        cost_volume = self.corr_2(self.bottleneck_2(feats_t2), warped_source)
        x = torch.cat([up_3, feats_t2, cost_volume], 1)
        flow_2 = up_flow_3 + self.flow_2(x) * 0.1
        up_2 = self.up_2(x)
        # Decoder Layer 1
        up_flow_2 = self.upsample(flow_2 * 2)
        warped_source = self.warp(self.bottleneck_1(feats_s1), up_flow_2)
        cost_volume = self.corr_1(self.bottleneck_1(feats_t1), warped_source)
        x = torch.cat([up_2, feats_t1, cost_volume], 1)
        flow_1 = up_flow_2 + self.flow_1(x) * 0.1
        up_1 = self.up_1(x)

        # Decoder layer 0
        up_flow_1 = self.upsample(flow_1 * 2)
        warped_source = self.warp(self.bottleneck_0(feats_s0), up_flow_1)
        cost_volume = self.corr_0(self.bottleneck_0(feats_t0), warped_source)

        # Final decoder layer
        x = torch.cat([up_1, feats_t0, cost_volume], 1)
        x = self.upsample(x)
        flow_final = self.upsample(up_flow_1 * 2) + self.flow_final(x) * 0.1

        return [flow_final, flow_1, flow_2, flow_3, flow_4], None

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, _, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0 * vgrid[:, 0, :, :].clone() / (W - 1) - 1.0
        vgrid[:,1,:,:] = 2.0 * vgrid[:, 1, :, :].clone() / (H - 1) - 1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = torch.ones(x.size())
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask


    def hook_layer0(self, module, input, output):
        self.result_layer0 = output
    def hook_layer1(self, module, input, output):
        self.result_layer1 = output
    def hook_layer2(self, module, input, output):
        self.result_layer2 = output
    def hook_layer3(self, module, input, output):
        self.result_layer3 = output
    def hook_layer4(self, module, input, output):
        self.result_layer4 = output
    def reset_hooks(self):
        self.result_layer0 = None
        self.result_layer1 = None
        self.result_layer2 = None
        self.result_layer3 = None
        self.result_layer4 = None

