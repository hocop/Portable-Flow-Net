import torch
import torch.nn as nn


class PredictFlow(nn.Module):
    '''
    This layer predicts optical flow
    '''
    def __init__(
        self,
        in_channels,
        inner_dim=32,
    ):
        '''
        in_channels - number of input channels
        '''
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, 2, kernel_size=3, stride=1, padding=1, dilation=1),
        )

    def forward(self, x):
        flow = self.conv(x)
        return flow

# def PredictFlow:
# return nn.Sequential(
#     nn.Conv2d(feature_sizes[0] + 32 + 32, 32, 1),
#     nn.Conv2d(32, 32, 3, padding=2, dilation=2, groups=32),
#     nn.BatchNorm2d(32),
#     nn.ELU(inplace=True),
#     nn.Conv2d(32, 32, 1),
#     nn.Conv2d(32, 32, 3, padding=4, dilation=4, groups=32),
#     nn.BatchNorm2d(32),
#     nn.ELU(inplace=True),
#     nn.Conv2d(32, 32, 1),
#     nn.Conv2d(32, 32, 3, padding=8, dilation=8, groups=32),
#     nn.BatchNorm2d(32),
#     nn.ELU(inplace=True),
#     nn.Conv2d(32, 32, 1),
#     nn.Conv2d(32, 32, 3, padding=16, dilation=16, groups=32),
#     nn.BatchNorm2d(32),
#     nn.ELU(inplace=True),
#     nn.Conv2d(32, 32, 1),
#     nn.Conv2d(32, 32, 3, padding=1, dilation=1, groups=32),
#     nn.BatchNorm2d(32),
#     nn.ELU(inplace=True),
#     nn.Conv2d(32, 2, 1),
# )


class PredictMask(nn.Module):
    '''
    This layer predicts binary mask
    '''
    def __init__(
        self,
        in_channels,
        inner_dim=32,
    ):
        '''
        in_channels - number of input channels
        '''
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_dim, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, 1, kernel_size=3, stride=1, padding=1, dilation=1),
        )

    def forward(self, x):
        logits = self.conv(x)
        return torch.sigmoid(logits)
