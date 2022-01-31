'''
Model class using pytorch lightning
'''

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl

from . import networks, losses, geometry, evaluation
from .utils import visualization


class LitFlow(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Make flow net
        self.flow_net = networks.pwc_one.PWCOne()
        # self.flow_net = networks.fast_scnn.FastSCNN(6, 2)
        # self.flow_net = networks.fast_pwc.FastPWC()
        self.loss_fn = nn.MSELoss()
        self.image_h = self.hparams.image_h
        self.image_w = self.hparams.image_w
        self.sequence_names = {
            'valid': self.hparams.dev_sequences,
            'test': self.hparams.test_sequences,
        }

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model Parameters")
        parser.add_argument('--image_h', type=int, default=None)
        parser.add_argument('--image_w', type=int, default=None)
        parser.add_argument('--learning_rate', type=float, default=None)
        parser.add_argument('--log_flow_freq', type=int, default=None)
        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def resize(image, size):
        if size[0] == image.shape[2] and size[1] == image.shape[3]:
            return image
        return F.interpolate(image, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def resize_flow(flow, size):
        if size[0] == flow.shape[2] and size[1] == flow.shape[3]:
            return flow
        scale_x = size[1] / flow.shape[3]
        scale_y = size[0] / flow.shape[2]
        new_flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=False)
        # pylint: disable=not-callable
        scale = torch.tensor([scale_x, scale_y], device=flow.device)
        scale = scale[None, :, None, None] # [1, 2, 1, 1]
        return new_flow * scale

    def grad_x(self, flow):
        return flow[:, :, 1:, 1:] - flow[:, :, 1:, :-1]
    def grad_xx(self, flow):
        return 2 * flow[:, :, 1:-1, 1:-1] - flow[:, :, 1:-1, :-2] - flow[:, :, 1:-1, 2:]
    def grad_y(self, flow):
        return flow[:, :, 1:, 1:] - flow[:, :, :-1, 1:]
    def grad_yy(self, flow):
        return 2 * flow[:, :, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1] - flow[:, :, 2:, 1:-1]


    def forward(self, batch):
        '''
        Inputs:
            batch: tensor of 2 images concatenated together; shape: [batch_size, 2 * 3, height, width]
        '''
        # Predict flow
        flows = self.flow_net(batch)

        return flows[0]

    def training_step(self, batch, batch_idx):
        # Freeze batch normalization after 2 epochs
        # if self.current_epoch <= 1:
        #     self.flow_net.train()
        # else:
        #     self.flow_net.eval()

        # Prepare input tensors
        model_input_bwd = torch.cat([batch['image_tgt'], batch['image_src']], 1)
        model_input_fwd = torch.cat([batch['image_src'], batch['image_tgt']], 1)
        model_input = torch.cat([model_input_bwd, model_input_fwd], 0)
        flow_gt = torch.cat([batch['flow_bwd'], batch['flow_fwd']], 0)

        # Predict flow
        flows = self.flow_net(model_input)

        loss = 0
        # alphas = [0.005, 0.01, 0.02, 0.08, 0.32]
        alphas = np.array([1 / flow.shape[2] for flow in flows])
        alphas = alphas / alphas.sum()
        for i, flow in enumerate(flows):
            # Resize ground truth
            flow_gt_scaled = self.resize_flow(flow_gt, (flow.shape[2], flow.shape[3]))

            # Flow loss
            tolerance = 0.05
            flow_mse = ((flow - flow_gt_scaled)**2).sum(1)
            flow_loss = torch.sqrt(flow_mse + tolerance**2) - tolerance

            # Edge flow loss
            weight = torch.abs(self.grad_xx(flow_gt_scaled) + self.grad_yy(flow_gt_scaled))
            weight = weight.sum(1).clamp(0.1, 1)
            flow_loss = flow_loss[:, 1:-1, 1:-1] * weight

            # Sum of losses
            loss_i = flow_loss.mean()
            loss = loss + loss_i * alphas[i]
            # Log loss
            self.log(
                f'loss{i}',
                loss_i,
                on_step=True, on_epoch=False, prog_bar=True, logger=False
            )

        if loss != loss:
            print('ERROR: NaN loss!')

        self.log(
            'train/loss',
            loss,
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0, mode='valid'):
        import matplotlib.pyplot as plt

        # Prepare input tensors
        model_input = torch.cat([batch['image_tgt'], batch['image_src']], 1)

        # Predict flow
        flows = self.flow_net(model_input)
        flow = flows[0]

        # Resize ground truth
        flow_gt = self.resize_flow(batch['flow_bwd'], (flow.shape[2], flow.shape[3]))
        # Compute loss
        loss = (torch.abs(flow - flow_gt)).mean()
        self.log(
            f'{mode}/loss',
            loss,
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        # Compute flow under-estimation coefficient
        avg = torch.sqrt((flow**2).sum(1)).mean(1).mean(1)
        avg_gt = torch.sqrt((flow_gt**2).sum(1)).mean(1).mean(1)
        flow_scale_coef = (avg / (avg_gt + 1e-6)).mean()
        self.log(
            f'{mode}/scale_coef',
            flow_scale_coef,
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        if batch_idx % self.hparams.log_flow_freq == 0:
            # Flow to numpy
            flow_np = flow[0].detach().cpu().numpy().transpose(1, 2, 0)
            flow_gt_np = flow_gt[0].detach().cpu().numpy().transpose(1, 2, 0)

            # Get sequence name
            if self.sequence_names[mode] is not None:
                seq_name = self.sequence_names[mode][dataloader_idx]
            else:
                seq_name = f'{mode}_seq_{dataloader_idx}'

            # Plot optical flow
            fig, axes = plt.subplots(2, 2, figsize=(15, 14))
            axes[0, 0].set_title('Image target')
            axes[0, 0].imshow(batch['image_tgt'][0].detach().cpu().numpy().transpose(1, 2, 0))
            axes[1, 0].set_title('Image source')
            axes[1, 0].imshow(batch['image_src'][0].detach().cpu().numpy().transpose(1, 2, 0))
            axes[0, 1].set_title('Flow')
            axes[0, 1].imshow(visualization.flow_to_rgb(flow_np))
            axes[1, 1].set_title('Flow ground truth')
            axes[1, 1].imshow(visualization.flow_to_rgb(flow_gt_np))
            fig.tight_layout()
            tensorboard = self.logger.experiment[-1]
            tensorboard.add_figure(
                f'{seq_name}_batch{batch_idx}',
                fig,
                self.current_epoch
            )


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, mode='test')


    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict