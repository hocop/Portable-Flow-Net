'''
Reprojection functions.
'''

import time
import torch
import torch.nn as nn


def warp(x, flo, return_mask=False):
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

    # Scale grid to [-1,1]
    # vgrid[:,0,:,:] = 2.0 * vgrid[:, 0, :, :].clone() / (W - 1) - 1.0
    # vgrid[:,1,:,:] = 2.0 * vgrid[:, 1, :, :].clone() / (H - 1) - 1.0

    # vgrid = vgrid.permute(0,2,3,1) # [B, H, W, 2]
    # output = nn.functional.grid_sample(x, vgrid, align_corners=False)
    # mask = torch.ones(x.size())
    # if x.is_cuda:
    #     mask = mask.cuda()
    # mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1

    # return output*mask

    output, mask = bilinear_sample_noloop(x, vgrid)

    if return_mask:
        return output, mask
    else:
        return output


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
