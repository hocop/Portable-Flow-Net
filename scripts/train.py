'''
Training script
'''

import os
import argparse
import pytorch_lightning as pl
import torch
import matplotlib
import datetime
matplotlib.use("agg")

import flow_sv


def main(config):

    # Initialize config
    data_config, logger = flow_sv.initialize(config)

    # Create train, valid, and test datasets
    data_module = flow_sv.datasets.FlowDataModule(config, data_config)

    # Create lit module
    lit_module = flow_sv.pl_module.LitFlow(**config.__dict__)
    if config.load_from is not None:
        lit_module = lit_module.load_from_checkpoint(checkpoint_path=config.load_from, **config.__dict__)
        print()
        print('Loaded', config.load_from)
        print()

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=logger,
    )

    # Train model
    # pylint: disable=no-member
    trainer.fit(
        lit_module,
        data_module
    )

    # Save path
    if config.save == 1:
        now = datetime.datetime.now()
        model_path = os.path.join(config.save_to, f'{now.strftime("%Y.%m.%d")}_{config.name}')
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, f'model.pt')
    else:
        model_path = None

    # Save model (with optimizer and scheduler for future)
    print('Saving model as', model_path)
    trainer.save_checkpoint(model_path)

    # Test
    if len(config.test_sequences) > 0:
        trainer.test(
            lit_module,
            datamodule=data_module,
        )


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train visual odometry and depth at the same time')

    parser.add_argument(
        '--config', type=str,
        help='configuration file in yaml format (ex.: config_kitti.yaml)')
    parser.add_argument(
        '--data_config', type=str, default=None,
        help='dataset configuration file in yaml format (ex.: datasets/datasets[dol-ml5].yaml)')
    parser.add_argument(
        '--name', type=str, required=True,
        help='name of the experiment for logging')
    parser.add_argument(
        '--wandb_project', type=str, default=None,
        help='project name for Weights&Biases')
    parser.add_argument(
        '--load_from', type=str, default=None,
        help='saved checkpoint')
    parser.add_argument(
        '--save', type=int, default=0,
        help='If 1, save trained model to ./saved_models/{name}/')
    parser.add_argument(
        '--save_to', type=str, default='saved_model'
    )

    parser = flow_sv.pl_module.LitFlow.add_argparse_args(parser)
    parser = flow_sv.datasets.FlowDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = flow_sv.utils.initialization.set_argparse_defaults_from_yaml(parser)

    args = parser.parse_args()

    # Run program
    main(args)
