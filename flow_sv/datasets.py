'''
Classes for loading optical flow datasets
'''

import os
import cv2
import numpy as np
from argparse import Namespace
from typing import Optional
import torch
from torch.utils.data import DataLoader
import albumentations as A
import pytorch_lightning as pl
import nnio

from flow_sv.utils import flow_io


class FlowDataModule(pl.LightningDataModule):
    def __init__(self, config: Namespace, data_config: dict):
        super().__init__()

        self.config = config
        self.data_config = data_config

        # Augmentations for training
        self.augmentations_parallel = A.Compose([
            # Crops
            A.RandomCrop(self.config.crop_h, self.config.crop_w),
            # Flips
            FlowFlip('x', p=0.5),
            FlowFlip('y', p=0.5),
            # Random conditions
            A.RandomRain(p=0.001),
            A.RandomSunFlare(p=0.001),
            A.RandomFog(p=0.001),
            # Noises
            A.GaussNoise(p=0.05),
            A.ImageCompression(quality_lower=1, p=0.01),
            # Color transforms
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.5),
            A.ChannelShuffle(p=0.1),
            A.ChannelDropout(p=0.02),
            A.ToGray(p=0.5),
            A.InvertImg(p=0.02),
        ], additional_targets={
            'image_2': 'image',
            'flow_fwd': 'mask',
            'flow_bwd': 'mask',
        })
        self.augmentations_indep = A.Compose([
            # Random conditions
            A.RandomRain(p=0.001),
            A.RandomSunFlare(p=0.001),
            A.RandomFog(p=0.001),
            # Noises
            A.GaussNoise(p=0.01),
            A.ImageCompression(quality_lower=1, p=0.01),
            A.ISONoise(color_shift=(0.1, 0.5), p=0.1),
            # Color transforms
            A.RandomBrightnessContrast(p=0.02),
            A.HueSaturationValue(p=0.05),
            A.HueSaturationValue(sat_shift_limit=100, val_shift_limit=100, p=0.05),
            A.ChannelShuffle(p=0.01),
            A.ChannelDropout(p=0.01),
            A.ToGray(p=0.01),
            A.InvertImg(p=0.01),
        ])

        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataloader parameters")
        parser.add_argument('--train_sequences', type=list, default=None)
        parser.add_argument('--dev_sequences', type=list, default=None)
        parser.add_argument('--test_sequences', type=list, default=None)
        parser.add_argument('--image_h', type=int, default=None)
        parser.add_argument('--image_w', type=int, default=None)
        parser.add_argument('--image_h_val', type=int, default=None)
        parser.add_argument('--image_w_val', type=int, default=None)
        parser.add_argument('--image_h_test', type=int, default=None)
        parser.add_argument('--image_w_test', type=int, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--cpus', type=int, default=None)
        return parent_parser

    def setup(self, stage: Optional[str] = None):
        # Create training dataset
        if self.train_dataset is None:
            self.train_dataset = torch.utils.data.ConcatDataset([
                get_dataset(
                    self.data_config[seq],
                    resize=(self.config.image_w, self.config.image_h),
                    channels_first=True,
                    augmentations_indep=self.augmentations_indep,
                    augmentations_parallel=self.augmentations_parallel,
                )
                for seq in self.config.train_sequences
            ])

        # Create validation datasets
        if self.val_datasets is None:
            self.val_datasets = [
                get_dataset(
                    self.data_config[seq],
                    resize=(self.config.image_w_val, self.config.image_h_val),
                    channels_first=True,
                )
                for seq in self.config.dev_sequences
            ]

        # Create testing datasets
        if self.test_datasets is None:
            self.test_datasets = [
                get_dataset(
                    self.data_config[seq],
                    resize=(self.config.image_w_test, self.config.image_h_test),
                    channels_first=True,
                )
                for seq in self.config.test_sequences
            ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.cpus,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.config.batch_size,
                num_workers=self.config.cpus,
            )
            for ds in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.config.batch_size,
                num_workers=self.config.cpus,
            )
            for ds in self.test_datasets
        ]


def get_dataset(cfg, **kwargs):
    if cfg['format'] == 'flyingthings':
        return FlyingThingsFormatDataset(
            cfg['images'],
            cfg['flows_fwd'],
            cfg['flows_bwd'],
            **kwargs,
        )
    if cfg['format'] == 'flyingchairs':
        return FlyingChairsFormatDataset(
            cfg['path'],
            **kwargs,
        )
    if cfg['format'] == 'mpi_sintel':
        return MPISintelDataset(
            cfg['path'],
            **kwargs,
        )


class FlowFlip(A.DualTransform):
    '''
    Horizontal flip changing left and right frames
    This augmentation must go before other augmentations but after camera matrix transform
    '''

    def __init__(
        self,
        direction,
        always_apply=False,
        p=0.5,
    ):
        'Standard initialization for Albumenations'
        super().__init__(always_apply, p)

        assert direction in ['x', 'y']

        if direction == 'x':
            self.flip = A.HorizontalFlip(always_apply=True, p=1.0)
        else:
            self.flip = A.VerticalFlip(always_apply=True, p=1.0)
        self.direction = direction

    def apply_with_params(self, _, **kwargs):
        '''
        This augmentation must go before other augmentations but after camera matrix transform
        '''
        flipped = self.flip(**kwargs)
        for key in flipped:
            if 'flow' in key:
                if self.direction == 'x':
                    flipped[key][:, :, 0] = -flipped[key][:, :, 0]
                else:
                    flipped[key][:, :, 1] = -flipped[key][:, :, 1]
        return flipped

    def add_targets(self, additional_targets):
        if additional_targets:
            self.flip.add_targets(additional_targets)


class FlyingThingsFormatDataset(torch.utils.data.Dataset):
    '''
    Loads images from folders with video sequences
    '''
    def  __init__(
        self,
        images_path,
        flow_fwd_path,
        flow_bwd_path,
        resize=None,
        augmentations_parallel=None,
        augmentations_indep=None,
        channels_first=True,
    ):
        """
        Parameters
        ----------
        TODO
        resize (tuple or None): size to which images must be scaled
        """

        self.resize = resize
        self.augmentations_parallel = augmentations_parallel
        self.augmentations_indep = augmentations_indep
        self.channels_first = channels_first

        self.preproc = nnio.Preprocessing(
            resize=resize,
            channels_first=channels_first,
            divide_by_255=True,
            dtype='float32',
        )

        # Find image paths
        image_files = sorted(os.listdir(images_path))
        fwd_files = sorted(os.listdir(flow_fwd_path))
        bwd_files = sorted(os.listdir(flow_bwd_path))

        pref = lambda fname: ''.join([c for c in fname if c in '0123456789'])
        fwd_prefixes = {pref(fname): fname for fname in fwd_files}
        bwd_prefixes = {pref(fname): fname for fname in bwd_files}

        self.pairs = []
        for i in range(1, len(image_files)):
            tgt_fname = image_files[i]
            src_fname = image_files[i - 1]
            if pref(src_fname) in fwd_prefixes and pref(tgt_fname) in bwd_prefixes:
                fwd_fname = fwd_prefixes[pref(src_fname)]
                bwd_fname = bwd_prefixes[pref(tgt_fname)]
                self.pairs.append({
                    'tgt': os.path.join(images_path, tgt_fname),
                    'src': os.path.join(images_path, src_fname),
                    'fwd': os.path.join(flow_fwd_path, fwd_fname),
                    'bwd': os.path.join(flow_bwd_path, bwd_fname),
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        # Load images
        image_tgt = cv2.imread(self.pairs[item]['tgt'])
        image_src = cv2.imread(self.pairs[item]['src'])
        if image_tgt is None:
            print(f'WARNING: cannot load image {self.pairs[item]["tgt"]}')
        if image_src is None:
            print(f'WARNING: cannot load image {self.pairs[item]["src"]}')
        image_tgt = image_tgt[:, :, ::-1]
        image_src = image_src[:, :, ::-1]

        # Load optical flow
        flow_fwd = flow_io.read(self.pairs[item]['fwd'])[:, :, :2].copy()
        flow_bwd = flow_io.read(self.pairs[item]['bwd'])[:, :, :2].copy()

        # Clear up NaN's
        nan_mask = flow_bwd != flow_bwd
        if nan_mask.any():
            flow_bwd[nan_mask] = 0
        nan_mask = flow_fwd != flow_fwd
        if nan_mask.any():
            flow_fwd[nan_mask] = 0

        # Augment images
        if self.augmentations_parallel is not None:
            images = self.augmentations_parallel(
                image=image_tgt,
                image_2=image_src,
                flow_bwd=flow_bwd,
                flow_fwd=flow_fwd,
            )
            image_tgt = images['image']
            image_src = images['image_2']
            flow_bwd = images['flow_bwd']
            flow_fwd = images['flow_fwd']
        if self.augmentations_indep is not None:
            image_tgt = self.augmentations_indep(
                image=image_tgt,
            )['image']
            image_src = self.augmentations_indep(
                image=image_src,
            )['image']

        # Resize and preprocess images
        image_tgt = self.preproc(image_tgt)
        image_src = self.preproc(image_src)

        if self.channels_first:
            flow_fwd = flow_fwd.transpose([2, 0, 1])
            flow_bwd = flow_bwd.transpose([2, 0, 1])

        return {
            'image_tgt': image_tgt,
            'image_src': image_src,
            'flow_fwd': flow_fwd,
            'flow_bwd': flow_bwd,
        }


class FlyingChairsFormatDataset(torch.utils.data.Dataset):
    '''
    Loads images from folders with video sequences
    '''
    def  __init__(
        self,
        path,
        resize=None,
        augmentations_parallel=None,
        augmentations_indep=None,
        channels_first=True,
    ):
        """
        Parameters
        ----------
        TODO
        resize (tuple or None): size to which images must be scaled
        """

        self.resize = resize
        self.augmentations_parallel = augmentations_parallel
        self.augmentations_indep = augmentations_indep
        self.channels_first = channels_first

        self.preproc = nnio.Preprocessing(
            resize=resize,
            channels_first=channels_first,
            divide_by_255=True,
            dtype='float32',
        )

        # Find image paths
        all_files = sorted(os.listdir(path))
        self.img1_paths = [os.path.join(path, f) for f in all_files if 'img1' in f]
        self.img2_paths = [os.path.join(path, f) for f in all_files if 'img2' in f]
        self.flow_paths = [os.path.join(path, f) for f in all_files if 'flow' in f]

    def __len__(self):
        return len(self.img1_paths)

    def __getitem__(self, item):
        # Load images
        image_tgt = cv2.imread(self.img1_paths[item])
        image_src = cv2.imread(self.img2_paths[item])
        if image_tgt is None:
            print(f'WARNING: cannot load image {self.pairs[item]["tgt"]}')
        if image_src is None:
            print(f'WARNING: cannot load image {self.pairs[item]["src"]}')
        image_tgt = image_tgt[:, :, ::-1]
        image_src = image_src[:, :, ::-1]

        # Load optical flow
        flow_bwd = flow_io.read(self.flow_paths[item])[:, :, :2].copy()

        # Clear up NaN's
        nan_mask = flow_bwd != flow_bwd
        if nan_mask.any():
            flow_bwd[nan_mask] = 0

        # Augment images
        if self.augmentations_parallel is not None:
            images = self.augmentations_parallel(
                image=image_tgt,
                image_2=image_src,
                flow_bwd=flow_bwd,
            )
            image_tgt = images['image']
            image_src = images['image_2']
            flow_bwd = images['flow_bwd']
        if self.augmentations_indep is not None:
            image_tgt = self.augmentations_indep(
                image=image_tgt,
            )['image']
            image_src = self.augmentations_indep(
                image=image_src,
            )['image']

        # Resize and preprocess images
        image_tgt = self.preproc(image_tgt)
        image_src = self.preproc(image_src)

        if self.channels_first:
            flow_bwd = flow_bwd.transpose([2, 0, 1])

        return {
            'image_tgt': image_tgt,
            'image_src': image_src,
            'flow_bwd': flow_bwd,
        }


class MPISintelDataset(torch.utils.data.Dataset):
    '''
    Loads images from folders with video sequences
    '''
    def  __init__(
        self,
        path,
        resize=None,
        augmentations_parallel=None,
        augmentations_indep=None,
        channels_first=True,
    ):
        """
        Parameters
        ----------
        TODO
        resize (tuple or None): size to which images must be scaled
        """

        self.resize = resize
        self.augmentations_parallel = augmentations_parallel
        self.augmentations_indep = augmentations_indep
        self.channels_first = channels_first

        self.preproc = nnio.Preprocessing(
            resize=resize,
            channels_first=channels_first,
            divide_by_255=True,
            dtype='float32',
        )

        # Paths
        clean_path = path.replace('*', 'clean')
        final_path = path.replace('*', 'final')
        flow_path = path.replace('*', 'flow')

        # Find image paths
        all_files = sorted(os.listdir(final_path))
        self.imgs_clean = [os.path.join(clean_path, f) for f in all_files]
        self.imgs_final = [os.path.join(final_path, f) for f in all_files]
        if os.path.isdir(flow_path):
            self.flows = [os.path.join(flow_path, f) for f in sorted(os.listdir(flow_path))]
        else:
            self.flows = None

    def __len__(self):
        return len(self.imgs_final) - 1

    def __getitem__(self, item):
        # Load images
        image_tgt = cv2.imread(self.imgs_final[item])
        image_src = cv2.imread(self.imgs_final[item + 1])
        if image_tgt is None:
            print(f'WARNING: cannot load image {self.imgs_final[item]}')
        if image_src is None:
            print(f'WARNING: cannot load image {self.imgs_final[item + 1]}')
        image_tgt = image_tgt[:, :, ::-1]
        image_src = image_src[:, :, ::-1]

        # Load optical flow
        if self.flows is not None:
            flow_bwd = flow_io.read(self.flows[item])[:, :, :2].copy()

            # Clear up NaN's
            nan_mask = flow_bwd != flow_bwd
            if nan_mask.any():
                flow_bwd[nan_mask] = 0

        # Augment images
        if self.augmentations_parallel is not None:
            images = self.augmentations_parallel(
                image=image_tgt,
                image_2=image_src,
                **({} if flow_bwd is None else {'flow_bwd': flow_bwd}),
            )
            image_tgt = images['image']
            image_src = images['image_2']
            if flow_bwd is not None:
                flow_bwd = images['flow_bwd']
        if self.augmentations_indep is not None:
            image_tgt = self.augmentations_indep(
                image=image_tgt,
            )['image']
            image_src = self.augmentations_indep(
                image=image_src,
            )['image']

        # Resize and preprocess images
        image_tgt = self.preproc(image_tgt)
        image_src = self.preproc(image_src)

        if self.channels_first and flow_bwd is not None:
            flow_bwd = flow_bwd.transpose([2, 0, 1])

        result = {
            'image_tgt': image_tgt,
            'image_src': image_src,
            **({} if flow_bwd is None else {'flow_bwd': flow_bwd}),
        }
        if flow_bwd is not None:
            result['flow_bwd'] = flow_bwd

        return result



def run_test():
    import matplotlib.pyplot as plt
    from flow_sv.utils import visualization
    from flow_sv import geometry

    path = '/home/ruslan/data/datasets_hdd/flow/MPI-Sintel-complete/training/*/alley_1'

    augmentations_parallel = A.Compose([
        # Crops
        # A.RandomCrop(540, 540),
        A.RandomCrop(384, 384),
        # Flips
        FlowFlip('x', p=0.5),
        FlowFlip('y', p=0.5),
        # Random conditions
        A.RandomRain(p=0.001),
        A.RandomSunFlare(p=0.01),
        A.RandomFog(p=0.01),
        # Noises
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.05),
        A.ImageCompression(quality_lower=1, p=0.1),
        # Color transforms
        A.HueSaturationValue(p=0.5),
        A.ChannelShuffle(p=0.1),
        A.ChannelDropout(p=0.01),
        A.ToGray(p=0.1),
        A.InvertImg(p=0.1),
    ], additional_targets={
        'image_2': 'image',
        'flow_fwd': 'mask',
        'flow_bwd': 'mask',
    })
    augmentations_indep = A.Compose([
        # Random conditions
        A.RandomRain(p=0.001),
        A.RandomSunFlare(p=0.01),
        A.RandomFog(p=0.01),
        # Noises
        A.RandomBrightnessContrast(p=0.02),
        A.GaussNoise(p=0.01),
        A.ImageCompression(quality_lower=1, p=0.1),
        A.ISONoise(color_shift=(0.1, 0.5), p=0.1),
        # Color transforms
        A.HueSaturationValue(p=0.05),
        A.HueSaturationValue(sat_shift_limit=100, val_shift_limit=100, p=0.05),
        A.ChannelShuffle(p=0.01),
        A.ChannelDropout(p=0.01),
        A.ToGray(p=0.01),
        A.InvertImg(p=0.01),
    ])

    train_dataset = MPISintelDataset(
        path,
        # resize=(256, 256),
        channels_first=True,
        augmentations_parallel=augmentations_parallel,
        augmentations_indep=augmentations_indep,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=False,
    )

    flownet = nnio.ONNXModel('onnx_output/2022.02.17_size512/size512_op12_simp.onnx')

    import matplotlib
    print(matplotlib.get_backend())

    for batch in dataloader:
        # batch = train_dataset[i]
        image_tgt = batch['image_tgt'][0].numpy().transpose([1, 2, 0])
        image_src = batch['image_src'][0].numpy().transpose([1, 2, 0])
        flow_bwd = batch['flow_bwd'][0].numpy().transpose([1, 2, 0])

        print([(key, batch[key].shape) for key in batch])

        # Reprojection
        # pylint: disable=not-callable
        reprojection = geometry.warp(
            torch.tensor(image_src.transpose([2, 0, 1])[None]),
            torch.tensor(flow_bwd.transpose([2, 0, 1])[None]),
        )
        reprojection = reprojection.numpy()[0].transpose([1, 2, 0])

        # Predict flow
        net_inp = np.concatenate([
            cv2.resize(image_tgt, (256, 256)).transpose([2, 0, 1])[None],
            cv2.resize(image_src, (256, 256)).transpose([2, 0, 1])[None],
        ], 1)
        flow_pred = flownet(net_inp)[0].transpose([1, 2, 0])

        fig, axes = plt.subplots(2, 3, figsize=(12, 10))

        axes[0, 0].set_title('image_tgt')
        axes[0, 0].imshow(image_tgt)
        axes[0, 1].set_title('image_src')
        axes[0, 1].imshow(image_src)
        axes[1, 0].set_title('src -> tgt')
        axes[1, 0].imshow(reprojection)

        axes[0, 2].set_title('Flow')
        axes[0, 2].imshow(visualization.flow_to_rgb(flow_bwd))
        axes[1, 2].set_title('Supervised model')
        axes[1, 2].imshow(visualization.flow_to_rgb(flow_pred))

        plt.show()


if __name__ == '__main__':
    run_test()
