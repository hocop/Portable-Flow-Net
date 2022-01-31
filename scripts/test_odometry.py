'''
'''
import argparse
import pytorch_lightning as pl

import vis_odo


def main(config):
    # Initialize config
    data_config, logger = vis_odo.initialize(config)

    # Create train, valid, and test datasets
    data_module = vis_odo.datasets.OdometryDataModule(config, data_config)

    # Create lit module
    lit_module = vis_odo.pl_module.LitOdometry.load_from_checkpoint(
        config.load_from,
        device='cuda',
        image_h=config.image_h,
        image_w=config.image_w,
        original_h=config.original_h,
        original_w=config.original_w,
        camera_matrix=config.camera_matrix,
        lambda_smoothness=config.lambda_smoothness,
        distance_btw_cameras=config.distance_btw_cameras,
        learning_rate=config.learning_rate,
        stereo=config.stereo,
        dev_sequences=config.dev_sequences,
        test_sequences=config.test_sequences,
    )

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=logger,
    )

    # Evaluate
    # pylint: disable=no-member
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
        '--name', type=str, default=None,
        help='name of the experiment for logging')
    parser.add_argument(
        '--wandb_project', type=str, default=None,
        help='project name for Weights&Biases')
    parser.add_argument(
        '--load_from', type=str,
        help='saved checkpoint')

    parser = vis_odo.pl_module.LitOdometry.add_argparse_args(parser)
    parser = vis_odo.datasets.OdometryDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser = vis_odo.utils.initialization.set_argparse_defaults_from_yaml(parser)

    args = parser.parse_args()

    # Run program
    main(args)
