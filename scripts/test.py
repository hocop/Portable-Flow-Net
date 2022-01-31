'''
Test model
'''

import os
import matplotlib.pyplot as plt
import tez
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import datasets
from utils import flow_utils, losses, trainer, geometry
import load_config
from models import pwcnet

config, _ = load_config.initialize(use_wandb=False)

# Create model
flownet = pwcnet.MaskPWCDCNet(config.original_h, config.original_w, config.camera_matrix_l)
pose_net = pwcnet.PoseNetMobileNetV2()
model = trainer.OdometryTrainer(flownet, pose_net, config.distance_btw_cameras, lr=config.learning_rate)

model.load('model.bin', device='cpu')

transform_k = datasets.TransformK(
    config.camera_matrix_l,
    config.camera_matrix_r,
    config.camera_matrix_virtual,
)

# Create validation dataset
valid_sequence_paths = [
    os.path.join(config.data_path, 'sequences', seq, 'image_2')
    for seq in config.dev_sequences
]
valid_sequence_paths_right = [
    os.path.join(config.data_path, 'sequences', seq, 'image_3')
    for seq in config.dev_sequences
]
valid_dataset = datasets.OdometryDataset(
    valid_sequence_paths,
    # valid_sequence_paths_right,
    resize=(config.image_h, config.image_w),
    channel_first=True,
    initial_transforms=transform_k
)

flownet = model.flownet

for i in range(100, len(valid_dataset)):
    imgs = valid_dataset[i]
    image1 = imgs['image'][None]
    image2 = imgs['image_prev'][None]

    out = flownet(image1, image2)

    fig, axes = plt.subplots(len(out['flows']), 3, figsize=(14,14))

    img1 = np.transpose(image1[0].numpy(), (1, 2, 0))
    img2 = np.transpose(image2[0].numpy(), (1, 2, 0))
    axes[0, 0].set_title('Left image')
    axes[0, 0].imshow(img1)

    for j in range(len(out['flows'])):
        flow = out['flows'][j]

        flow_image = flow_utils.flow_to_rgb(
            np.transpose(flow.detach().numpy()[0], (1, 2, 0))
        )

        axes[j, 1].set_title('Flow ' + str(j + 2))
        axes[j, 1].imshow(flow_image)

        axes[j, 2].set_title('Depth' + str(j + 2))
        axes[j, 2].imshow(1 / out['depths'][j][0, 0].detach().cpu(), vmin=0, vmax=0.5)

    plt.show()
