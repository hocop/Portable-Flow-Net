import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision

from .layers import mobilenetv2
from .layers.cost_volume import CostVolume
from . import backends


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool6 = nn.AdaptiveAvgPool2d(6)
        inter_channels = in_channels // 4
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels + inter_channels * 4, out_channels, 1)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool1(x)), size)
        feat2 = self.upsample(self.conv2(self.pool2(x)), size)
        feat3 = self.upsample(self.conv3(self.pool3(x)), size)
        feat4 = self.upsample(self.conv4(self.pool6(x)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class FastPWC(nn.Module):

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
            resnet.layer4 = nn.Identity()
            self.encoder = backends.FeatureExtractor(
                resnet,
                ['bn1', 'layer3']
            )
            feature_sizes = [64, 256]
        elif backend == 'mobilenet':
            # Load pretrained mobilenet
            mobilenet = backends.MobileNetV2MultiInput(2).features
            for i in range(14, 19):
                mobilenet[i] = nn.Identity()
            self.encoder = backends.FeatureExtractor(
                mobilenet,
                ['1', '13'] # 1/2, 1/16
            )
            feature_sizes = [16, 96]
        elif backend == 'shufflenet':
            # Load pretrained shufflenet
            shufflenet = backends.ShuffleNetV2MultiInput(2)
            shufflenet.stage4 = nn.Identity()
            self.encoder = backends.FeatureExtractor(
                shufflenet,
                ['conv1', 'stage3']
            )
            feature_sizes = [24, 96]

        # self.ppm = PyramidPooling(feature_sizes[1], 128)
        self.ppm = _ConvBNReLU(feature_sizes[1], 128, 1)

        self.upsample = nn.Upsample(
            scale_factor=8,
            mode='bilinear',
            align_corners=False
        )

        # Decoder layers
        self.flow_4 = FlowPred(128)
        self.flow_1 = FlowPred(feature_sizes[0] + 128)


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
        feats_1, feats_4 = self.encoder(images)

        # Predict at size 1/16
        x_4 = self.ppm(feats_4)
        flow_4 = self.flow_4(x_4)

        # Predict at size 1/2
        x_1 = self.upsample(x_4)
        x_1 = torch.cat([feats_1, x_1], 1)
        flow_1 = self.flow_1(x_1, self.upsample(flow_4) * 8)

        return [flow_1, flow_4]


class FlowPred(nn.Module):
    def __init__(self, in_channels, inner_channels=16, cvol_out=48, hidden_dim=64):
        super().__init__()
        self.bottleneck_target = nn.Conv2d(in_channels, inner_channels, 1)
        self.bottleneck_source = nn.Conv2d(in_channels, inner_channels, 1)
        self.corr = CostVolume(inner_channels, cvol_out, 7, dilation=2, groups=2)
        self.predict = nn.Sequential(
            nn.Conv2d(in_channels + cvol_out, in_channels + cvol_out, 3, padding=1, groups=in_channels + cvol_out),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ELU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),
        )

    def forward(self, feats, flow=None):
        # Warp source
        if flow is not None:
            source = self.warp(self.bottleneck_source(feats), flow.detach())
        else:
            source = self.bottleneck_source(feats)
        # Compute cost volume
        cost_volume = self.corr(self.bottleneck_target(feats), source)
        x = torch.cat([feats, cost_volume], 1)
        flow = self.predict(x)
        return flow

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


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)