import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision

from .layers import mobilenetv2
from .layers.cost_volume import CostVolume
from . import backends


def FlowPred(n_inputs, hidden_dim, expand_ratio=1):
    return nn.Sequential(
        nn.Conv2d(n_inputs, hidden_dim, 1),
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.ELU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1),
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.ELU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1),
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.ELU(inplace=True),
        nn.Conv2d(hidden_dim, 2, 1),
    )
    # return nn.Sequential(
    #     mobilenetv2.InvertedResidual(n_inputs, hidden_dim, 1, expand_ratio),
    #     mobilenetv2.InvertedResidual(hidden_dim, hidden_dim, 1, expand_ratio),
    #     mobilenetv2.InvertedResidual(hidden_dim, hidden_dim, 1, expand_ratio),
    #     mobilenetv2.InvertedResidual(hidden_dim, 2, 1, expand_ratio),
    # )


def UpConv(n_inputs, hidden_dim, expand_ratio=1):
    return nn.Sequential(
        nn.PixelShuffle(2),
        nn.Conv2d(n_inputs // 4, hidden_dim, 1),
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.ELU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1),
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.ELU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1),
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.ELU(inplace=True),
    )
    # return nn.Sequential(
    #     nn.PixelShuffle(2),
    #     mobilenetv2.InvertedResidual(n_inputs // 4, hidden_dim, 1, expand_ratio),
    #     mobilenetv2.InvertedResidual(hidden_dim, hidden_dim, 1, expand_ratio),
    #     mobilenetv2.InvertedResidual(hidden_dim, hidden_dim, 1, expand_ratio),
    #     mobilenetv2.InvertedResidual(hidden_dim, hidden_dim, 1, expand_ratio),
    # )


class PWCOne(nn.Module):
    CVOL_CHANNELS_IN = 16
    CVOL_CHANNELS_OUT = 48
    CVOL_GROUPS = 2
    CVOL_KERNEL = 5
    CVOL_DILATION = 2

    def __init__(self, backend='mobilenet'):
        """
        """
        super().__init__()
        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if backend == 'resnet':
            # Load pretrained resnet
            resnet = backends.ResNet18MultiInput(2)
            self.encoder = backends.FeatureExtractor(
                resnet,
                ['bn1', 'layer1', 'layer2', 'layer3', 'layer4']
            )
            feature_sizes = [64, 64, 128, 256, 512]
        elif backend == 'mobilenet':
            # Load pretrained mobilenet
            mobilenet = backends.MobileNetV2MultiInput(2).features
            mobilenet[18] = nn.Identity()
            self.encoder = backends.FeatureExtractor(
                mobilenet,
                ['1', '3', '6', '13', '17']
            )
            feature_sizes = [16, 24, 32, 96, 320]
        elif backend == 'shufflenet':
            # Load pretrained shufflenet
            shufflenet = backends.ShuffleNetV2MultiInput(2)
            self.encoder = backends.FeatureExtractor(
                shufflenet,
                ['conv1', 'maxpool', 'stage2', 'stage3', 'stage4']
            )
            feature_sizes = [24, 24, 48, 96, 192]

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        # Decoder layer 4
        self.bottleneck_t4 = nn.Conv2d(feature_sizes[4], self.CVOL_CHANNELS_IN, 1)
        self.bottleneck_s4 = nn.Conv2d(feature_sizes[4], self.CVOL_CHANNELS_IN, 1)
        self.corr_4 = CostVolume(
            self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, self.CVOL_KERNEL, dilation=self.CVOL_DILATION, groups=self.CVOL_GROUPS)
        self.flow_4 = FlowPred(feature_sizes[4] + self.CVOL_CHANNELS_OUT, 64)
        self.up_4 = UpConv(feature_sizes[4] + self.CVOL_CHANNELS_OUT, 128)
        # Decoder layer 3
        self.bottleneck_t3 = nn.Conv2d(feature_sizes[3], self.CVOL_CHANNELS_IN, 1)
        self.bottleneck_s3 = nn.Conv2d(feature_sizes[3], self.CVOL_CHANNELS_IN, 1)
        self.corr_3 = CostVolume(
            self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, self.CVOL_KERNEL, dilation=self.CVOL_DILATION, groups=self.CVOL_GROUPS)
        self.flow_3 = FlowPred(feature_sizes[3] + 128 + self.CVOL_CHANNELS_OUT, 64)
        self.up_3 = UpConv(feature_sizes[3] + 128 + self.CVOL_CHANNELS_OUT, 128)
        # Decoder layer 2
        self.bottleneck_t2 = nn.Conv2d(feature_sizes[2], self.CVOL_CHANNELS_IN, 1)
        self.bottleneck_s2 = nn.Conv2d(feature_sizes[2], self.CVOL_CHANNELS_IN, 1)
        self.corr_2 = CostVolume(
            self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, self.CVOL_KERNEL, dilation=self.CVOL_DILATION, groups=self.CVOL_GROUPS)
        self.flow_2 = FlowPred(feature_sizes[2] + 128 + self.CVOL_CHANNELS_OUT, 64)
        self.up_2 = UpConv(feature_sizes[2] + 128 + self.CVOL_CHANNELS_OUT, 64)
        # Decoder layer 1
        self.bottleneck_t1 = nn.Conv2d(feature_sizes[1], self.CVOL_CHANNELS_IN, 1)
        self.bottleneck_s1 = nn.Conv2d(feature_sizes[1], self.CVOL_CHANNELS_IN, 1)
        self.corr_1 = CostVolume(
            self.CVOL_CHANNELS_IN, self.CVOL_CHANNELS_OUT, self.CVOL_KERNEL, dilation=self.CVOL_DILATION, groups=self.CVOL_GROUPS)
        self.flow_1 = FlowPred(feature_sizes[1] + 64 + self.CVOL_CHANNELS_OUT, 64)
        self.up_1 = UpConv(feature_sizes[1] + 64 + self.CVOL_CHANNELS_OUT, 32)

        # Decoder layer 0
        self.bottleneck_t0 = nn.Conv2d(feature_sizes[0], self.CVOL_CHANNELS_IN, 1)
        self.bottleneck_s0 = nn.Conv2d(feature_sizes[0], self.CVOL_CHANNELS_IN, 1)
        self.corr_0 = CostVolume(
            self.CVOL_CHANNELS_IN, 32, self.CVOL_KERNEL, dilation=self.CVOL_DILATION, groups=self.CVOL_GROUPS)

        # Final decoder
        in_sz = feature_sizes[0] + 32 + 32
        self.flow_final = FlowPred(in_sz, 32)
        self.dc_conv = nn.Sequential(
            nn.Conv2d(in_sz, 128, 3, stride=1, padding=1,  dilation=1, groups=16),  nn.LeakyReLU(0.1),
            nn.Conv2d(128,   128, 3, stride=1, padding=2,  dilation=2, groups=1),  nn.LeakyReLU(0.1),
            nn.Conv2d(128,   128, 3, stride=1, padding=4,  dilation=4, groups=32),  nn.LeakyReLU(0.1),
            nn.Conv2d(128,   96,  3, stride=1, padding=8,  dilation=8, groups=1),  nn.LeakyReLU(0.1),
            nn.Conv2d(96,    64,  3, stride=1, padding=16, dilation=16, groups=32), nn.LeakyReLU(0.1),
            nn.Conv2d(64,    32,  3, stride=1, padding=1,  dilation=1, groups=1),  nn.LeakyReLU(0.1),
            nn.Conv2d(32,    2,   1),
        )


    def forward(self, images):
        '''
        input:
        - images: torch.tensor of shape [batch, 2 * 3, H, W]
            Two RGB images concatenated together
        returns:
        - flow: torch.tensor of shape [batch, 2, H, W]
        '''

        # Normalize input
        images = self.norm_input(images.view(-1, 3, images.shape[2], images.shape[3])).view(images.shape)

        # Compute features from encoder
        feats_0, feats_1, feats_2, feats_3, feats_4 = self.encoder(images)

        # Decoder Layer 4
        cost_volume = self.corr_4(self.bottleneck_t4(feats_4), self.bottleneck_s4(feats_4))
        x = torch.cat([feats_4, cost_volume], 1)
        flow_4 = self.flow_4(x)
        up_4 = self.up_4(x)
        # Decoder Layer 3
        up_flow_4 = self.upsample(flow_4 * 2)
        warped_source = self.warp(self.bottleneck_s3(feats_3), up_flow_4.detach())
        cost_volume = self.corr_3(self.bottleneck_t3(feats_3), warped_source)
        x = torch.cat([up_4, feats_3, cost_volume], 1)
        flow_3 = self.flow_3(x)
        up_3 = self.up_3(x)
        # Decoder Layer 2
        up_flow_3 = self.upsample(flow_3 * 2)
        warped_source = self.warp(self.bottleneck_s2(feats_2), up_flow_3.detach())
        cost_volume = self.corr_2(self.bottleneck_t2(feats_2), warped_source)
        x = torch.cat([up_3, feats_2, cost_volume], 1)
        flow_2 = self.flow_2(x)
        up_2 = self.up_2(x)
        # Decoder Layer 1
        up_flow_2 = self.upsample(flow_2 * 2)
        warped_source = self.warp(self.bottleneck_s1(feats_1), up_flow_2.detach())
        cost_volume = self.corr_1(self.bottleneck_t1(feats_1), warped_source)
        x = torch.cat([up_2, feats_1, cost_volume], 1)
        flow_1 = self.flow_1(x)
        up_1 = self.up_1(x)

        # Decoder layer 0
        up_flow_1 = self.upsample(flow_1 * 2)
        warped_source = self.warp(self.bottleneck_s0(feats_0), up_flow_1.detach())
        cost_volume = self.corr_0(self.bottleneck_t0(feats_0), warped_source)

        # Final decoder layer
        x = torch.cat([up_1, feats_0, cost_volume], 1)
        flow_final = self.flow_final(x) + self.dc_conv(x)

        return [flow_final, flow_1, flow_2, flow_3, flow_4]

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, _, H, W = x.size()
        # Mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float() # [B, 2, H, W]

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        output, mask = bilinear_sample_noloop(x, vgrid)
        return output * mask

def bilinear_sample_noloop(image, grid):
    """
    Bilinear sampling with no loops.
    from: https://forums.developer.nvidia.com/t/how-to-optimize-the-custom-bilinear-sampling-alternative-to-grid-sample-for-tensorrt-inference/178920/5

    :param image: sampling source of shape [N, C, H, W]
    :param grid: integer sampling pixel coordinates of shape [N, 2, grid_H, grid_W]
    :return: sampling result of shape [N, C, grid_H, grid_W]
    """
    Nt, C, H, W = image.shape
    grid_H = grid.shape[2]
    grid_W = grid.shape[3]
    xgrid, ygrid = grid.split([1, 1], dim=1)
    mask = ((xgrid >= 0) & (ygrid >= 0) & (xgrid < W - 1) & (ygrid < H - 1)).float()
    x0 = torch.floor(xgrid)
    x1 = x0 + 1
    y0 = torch.floor(ygrid)
    y1 = y0 + 1
    wa = ((x1 - xgrid) * (y1 - ygrid)).permute(1, 0, 2, 3) #.permute(3, 0, 1, 2)
    wb = ((x1 - xgrid) * (ygrid - y0)).permute(1, 0, 2, 3) #.permute(3, 0, 1, 2)
    wc = ((xgrid - x0) * (y1 - ygrid)).permute(1, 0, 2, 3) #.permute(3, 0, 1, 2)
    wd = ((xgrid - x0) * (ygrid - y0)).permute(1, 0, 2, 3) #.permute(3, 0, 1, 2)
    x0 = (x0 * mask).view(Nt, grid_H, grid_W).long()
    y0 = (y0 * mask).view(Nt, grid_H, grid_W).long()
    x1 = (x1 * mask).view(Nt, grid_H, grid_W).long()
    y1 = (y1 * mask).view(Nt, grid_H, grid_W).long()
    ind = torch.arange(Nt, device=image.device)
    ind = ind.view(Nt, 1).expand(-1, grid_H).view(Nt, grid_H, 1).expand(-1, -1, grid_W).long()
    image = image.permute(1, 0, 2, 3)
    output_tensor = (image[:, ind, y0, x0] * wa + image[:, ind, y1, x0] * wb + image[:, ind, y0, x1] * wc + \
                    image[:, ind, y1, x1] * wd).permute(1, 0, 2, 3)
    return output_tensor, mask
