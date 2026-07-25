# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""CLI entry point: load configuration and run training or inference."""

import os
import socket
import argparse
from datetime import datetime
import warnings
import yaml

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, EarlyStopping, ModelCheckpoint
import pprint

from .data.graph_data import graph_data_module
from .config.config_parsing import load_config
from .models.Model import Model
from .version import get_version, get_full_version_info, soft_logo
from .models.hamgnn_transformer import HamGNNTransformer 
from .models.hamgnn_conv import HamGNNConvE3
from .models.hamgnn_output import HamGNNPlusPlusOut
from .utils.hparam import get_hparam_dict


def _normalize_num_gpus(num_gpus):
    """Normalize the GPU configuration for PyTorch Lightning."""
    if num_gpus in (None, 0, '0'):
        return None
    if isinstance(num_gpus, (list, tuple)) and len(num_gpus) == 0:
        return None
    return num_gpus


def _count_requested_gpus(num_gpus) -> int:
    """Count how many GPUs are requested by the configuration."""
    if num_gpus is None:
        return 0
    if isinstance(num_gpus, int):
        return max(num_gpus, 0)
    if isinstance(num_gpus, (list, tuple)):
        return len(num_gpus)
    return 0


def _is_primary_process() -> bool:
    """Return ``True`` for rank-0 style processes used in distributed runs."""
    for rank_env in ('LOCAL_RANK', 'SLURM_LOCALID'):
        if rank_env in os.environ:
            try:
                return int(os.environ[rank_env]) == 0
            except ValueError:
                return True
    return True


def initialize_output_parameters(output_params):
    """Initialize default values for output parameters if they don't already exist."""
    default_params = {
        'add_H_nonsoc': False,          # Add non-spin-orbit coupling Hamiltonian
        'get_nonzero_mask_tensor': False, # Generate mask for non-zero elements
        'zero_point_shift': True,       # Apply zero-point energy shift
        'soc_basis': 'so3',             # Spin-orbit coupling basis
    }
    
    for param_name, default_value in default_params.items():
        if not hasattr(output_params, param_name):
            setattr(output_params, param_name, default_value)
    
    return output_params


def prepare_dataset(config):
    """Prepare the graph dataset for training, validation, and testing."""
    train_ratio = config.dataset_params.train_ratio
    val_ratio = config.dataset_params.val_ratio
    test_ratio = config.dataset_params.test_ratio
    batch_size = config.dataset_params.batch_size
    split_file = config.dataset_params.split_file
    graph_data_path = config.dataset_params.graph_data_path
    
    if not os.path.isfile(graph_data_path) and not graph_data_path.lower().endswith(".lmdb"):
        graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    
    num_workers = getattr(config.dataset_params, 'num_workers', 4)
    preload = getattr(config.dataset_params, 'preload', 0)
    data_format = getattr(config.dataset_params, 'data_format', 'auto')
    is_test_mode = (config.setup.stage == 'test')
    
    graph_dataset = graph_data_module(
        dataset=graph_data_path, 
        train_ratio=train_ratio, 
        val_ratio=val_ratio, 
        test_ratio=test_ratio, 
        batch_size=batch_size, 
        split_file=split_file,
        num_workers=num_workers,
        preload=preload,
        test_mode=is_test_mode,
        data_format=data_format
    )
    
    return graph_dataset


def build_hamgnn_model(config):
    """Build the HamGNN model components based on configuration."""
    print("Building model")
    
    config.representation_nets.HamGNN_pre.radius_type = config.output_nets.HamGNN_out.ham_type.lower()
    
    gnn_net_type = config.setup.GNN_Net.lower()
    if gnn_net_type in ['hamgnnconv', 'hamgnnpre', 'hamgnn_pre']:
        if 'use_corr_prod' not in config.representation_nets.HamGNN_pre:
            config.representation_nets.HamGNN_pre.use_corr_prod = True
        graph_representation = HamGNNConvE3(config.representation_nets)
    elif gnn_net_type == 'hamgnntransformer':
        graph_representation = HamGNNTransformer(config.representation_nets)
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        raise SystemExit(1)
    
    property_type = config.setup.property.lower()
    if property_type == 'hamiltonian':
        output_params = config.output_nets.HamGNN_out
        output_params = initialize_output_parameters(output_params)
        
        output_module = HamGNNPlusPlusOut(
            irreps_in_node=graph_representation.irreps_node_features, 
            irreps_in_edge=graph_representation.irreps_node_features, 
            nao_max=output_params.nao_max, 
            ham_type=output_params.ham_type,
            ham_only=output_params.ham_only, 
            symmetrize=output_params.symmetrize,
            calculate_band_energy=output_params.calculate_band_energy,
            num_k=output_params.num_k,
            k_path=output_params.k_path,
            band_num_control=output_params.band_num_control, 
            soc_switch=output_params.soc_switch, 
            soc_basis=output_params.soc_basis,
            nonlinearity_type=output_params.nonlinearity_type, 
            add_H0=output_params.add_H0, 
            spin_constrained=output_params.spin_constrained, 
            collinear_spin=output_params.collinear_spin, 
            minMagneticMoment=output_params.minMagneticMoment, 
            add_H_nonsoc=output_params.add_H_nonsoc,
            get_nonzero_mask_tensor=output_params.get_nonzero_mask_tensor, 
            zero_point_shift=output_params.zero_point_shift,
        )
    else:
        print(f'Property type "{property_type}" is not supported!')
        raise SystemExit(1)
    
    post_processing_utility = None
    return graph_representation, output_module, post_processing_utility


def setup_trainer(config, callbacks):
    """Set up PyTorch Lightning trainer updated for PyTorch 2.x & PL 2.0+."""
    tb_logger = TensorBoardLogger(
        save_dir=config.profiler_params.train_dir, 
        name="", 
        default_hp_metric=False
    )
    
    num_gpus = _normalize_num_gpus(getattr(config.setup, 'num_gpus', None))
    requested_gpu_count = _count_requested_gpus(num_gpus)
    
    prec_setting = getattr(config.setup, 'precision', 32)
    precision_str = f"{prec_setting}-true" if prec_setting in (32, 64, 16) else "32-true"

    if requested_gpu_count > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = num_gpus
        if requested_gpu_count > 1:
            strategy = DDPStrategy(static_graph=True)
        else:
            strategy = "auto"
    else:
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"

    trainer_callbacks = callbacks if callbacks is not None else []
    trainer_callbacks.append(TQDMProgressBar(refresh_rate=1))

    trainer_params = {
        'accelerator': accelerator,
        'devices': devices,
        'strategy': strategy,
        'precision': precision_str,
        'callbacks': trainer_callbacks,
        'logger': tb_logger,
        'gradient_clip_val': config.optim_params.gradient_clip_val,
        'max_epochs': config.optim_params.max_epochs,
        'default_root_dir': config.profiler_params.train_dir,
        'min_epochs': config.optim_params.min_epochs,
    }

    trainer = pl.Trainer(**trainer_params)
    return trainer, tb_logger


def load_or_create_model(config, graph_representation, output_module, post_processing_utility, losses, metrics):
    """Load an existing model from checkpoint or create a new one."""
    model_params = {
        'representation': graph_representation,
        'output': output_module,
        'post_processing': post_processing_utility,
        'losses': losses,
        'validation_metrics': metrics,
        'lr': config.optim_params.lr,
        'lr_decay': config.optim_params.lr_decay,
        'lr_patience': config.optim_params.lr_patience
    }
    
    is_load_checkpoint = config.setup.load_from_checkpoint and not config.setup.resume
    if is_load_checkpoint:
        model = Model.load_from_checkpoint(
            checkpoint_path=config.setup.checkpoint_path,
            **model_params
        )
    else:
        model = Model(**model_params)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params_count = sum([np.prod(p.size()) for p in model_parameters])
    print(f"The model you built has {params_count:,} parameters.")
    
    return model


def train_model(trainer, model, data_module, ckpt_path=None):
    """Train the model using the configured trainer."""
    print("Starting training...")
    
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    
    print("Training completed.")
    print("Starting evaluation...")
    
    test_results = trainer.test(model, datamodule=data_module)
    print("Evaluation completed.")
    
    return test_results


def test_model(trainer, model, data_module):
    """Test the model using the configured trainer."""
    print("Starting model testing...")
    trainer.test(model=model, datamodule=data_module)
    print("Testing completed.")


def train_and_evaluate(config):
    """Train and evaluate the HamGNN model based on configuration."""
    data_module = prepare_dataset(config)
    graph_representation, output_module, post_processing_utility = build_hamgnn_model(config)
    
    dtype = torch.float32 if config.setup.precision == 32 else torch.float64
    torch.set_default_dtype(dtype)
    
    graph_representation.to(dtype)
    output_module.to(dtype)
    
    losses = config.losses_metrics.losses
    metrics = config.losses_metrics.metrics
    
    callbacks = [
        LearningRateMonitor(),
        EarlyStopping(
            monitor="training/total_loss",
            patience=config.optim_params.stop_patience, 
            min_delta=1e-6,
        ),
        ModelCheckpoint(
            filename="{epoch}-{val_loss:.6f}",
            save_top_k=1,
            verbose=False,
            monitor='validation/total_loss',
            mode='min',
        )
    ]
    
    if config.setup.stage == 'fit':
        model = load_or_create_model(
            config, graph_representation, output_module, 
            post_processing_utility, losses, metrics
        )
        
        trainer, tb_logger = setup_trainer(config, callbacks)
        
        version_info = get_full_version_info()
        for key, value in version_info.items():
            tb_logger.experiment.add_text(f"version/{key}", str(value), global_step=0)
        
        ckpt_path = None
        if config.setup.resume and config.setup.checkpoint_path:
            ckpt_path = config.setup.checkpoint_path

        test_results = train_model(trainer, model, data_module, ckpt_path=ckpt_path)
        
        hparam_dict = get_hparam_dict(config)
        hparam_dict['version'] = get_version()
        metric_dict = {}
        for result_dict in test_results:
            metric_dict.update(result_dict)
        tb_logger.experiment.add_hparams(hparam_dict, metric_dict)
    
    elif config.setup.stage == 'test':
        model = Model.load_from_checkpoint(
            checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_processing_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
        )
        
        trainer, _ = setup_trainer(config, callbacks=None)
        test_model(trainer, model, data_module)


def HamGNN():
    pl.seed_everything(666)
    
    if _is_primary_process():
        print(soft_logo)
        version_info = get_full_version_info()
        print(f"Build timestamp: {version_info['timestamp']}")
        if version_info['is_dirty']:
            print("WARNING: This version was built with uncommitted changes")
    
    parser = argparse.ArgumentParser(description='Hamiltonian Graph Neural Network')
    parser.add_argument('--config', default='config.yaml', type=str, metavar='N')
    args = parser.parse_args()

    config = load_config(config_file_path=args.config)
    hostname = socket.getfqdn(socket.gethostname())
    config.setup.hostname = hostname
    
    if _is_primary_process():
        pprint.pprint(config)
    
    if config.setup.ignore_warnings:
        warnings.filterwarnings('ignore')
    
    train_and_evaluate(config)


if __name__ == '__main__':
    HamGNN()