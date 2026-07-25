# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""PyTorch Lightning training module wiring representation and output networks.

Defines :class:`Model` with optimizers, losses, metrics, and distributed training hooks.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import List, Dict, Union, Callable, Optional, Any

from ..utils.visualization import scatter_plot


class Model(pl.LightningModule):
    """
    A PyTorch Lightning module for scientific machine learning models.
    Updated for PyTorch 2.x & PyTorch Lightning 2.0+ compatibility.
    """

    def __init__(
            self,
            representation: nn.Module,
            output: nn.Module,
            losses: List[Dict],
            validation_metrics: List[Dict],
            lr: float = 1e-3,
            lr_decay: float = 0.1,
            lr_patience: int = 100,
            lr_monitor: str = "training/total_loss",
            epsilon: float = 1e-8,
            beta1: float = 0.99,
            beta2: float = 0.999,
            amsgrad: bool = True,
            max_points_to_scatter: int = 100000,
            post_processing: Optional[Callable] = None
    ):
        super().__init__()
        self.representation = representation
        self.output_module = output
        self.losses = losses
        self.metrics = validation_metrics
        
        # Optimizer parameters
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = amsgrad
        
        # Visualization parameters
        self.max_points_to_scatter = max_points_to_scatter
        
        # Post-processing for gradient-dependent physical quantities
        self.post_processing = post_processing
        
        # Track if derivatives are required
        self.requires_derivatives = self.output_module.derivative

        # PyTorch Lightning 2.0+ 要求的 Step 輸出容器
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _use_sync_dist(self) -> bool:
        """Return whether distributed metric synchronization is active."""
        return dist.is_available() and dist.is_initialized()

    def _is_global_zero(self) -> bool:
        """Return whether the current process is the global rank zero process."""
        return getattr(self.trainer, 'is_global_zero', True)

    def _gather_step_outputs(self, step_outputs: List[Dict]) -> List[Dict]:
        """Gather validation/test outputs from all distributed ranks."""
        if not self._use_sync_dist():
            return step_outputs

        gathered_outputs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_outputs, step_outputs)

        merged_outputs = []
        for rank_outputs in gathered_outputs:
            if rank_outputs is not None:
                merged_outputs.extend(rank_outputs)

        return merged_outputs

    def calculate_loss(self, batch: Dict[str, torch.Tensor], 
                       predictions: Dict[str, torch.Tensor], 
                       mode: str) -> torch.Tensor:
        """Calculate total loss by summing weighted individual loss components."""
        total_loss = torch.tensor(0.0, device=self.device)
        
        for loss_dict in self.losses:
            loss_fn = loss_dict["metric"]
            
            if "target" in loss_dict:
                prediction = predictions[loss_dict["prediction"].lower()]
                target = batch[loss_dict["target"].lower()]
                component_loss = loss_fn(prediction, target)
                
                if ('sparsity_ratio' in predictions and 
                    loss_dict["prediction"].lower() in ['hamiltonian', 'hamiltonian_real', 'hamiltonian_imag']):
                    sparsity_ratio = predictions['sparsity_ratio']
                    component_loss = component_loss * sparsity_ratio
            else:
                component_loss = loss_fn(predictions[loss_dict["prediction"].lower()])
                
            total_loss += loss_dict["loss_weight"] * component_loss
            
            loss_name = getattr(loss_fn, "name", type(loss_fn).__name__.split(".")[-1])
            self.log(
                f"{mode}/{loss_name}_{loss_dict['prediction']}",
                component_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=self._use_sync_dist(),
            )
            
        return total_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        self._enable_position_gradients(batch)
        predictions = self(batch)
        loss = self.calculate_loss(batch, predictions, 'training')
        self.log(
            "training/total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=self._use_sync_dist(),
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """Perform a single validation step."""
        torch.set_grad_enabled(self.requires_derivatives)
        self._enable_position_gradients(batch)
        predictions = self(batch)
        
        val_loss = self.calculate_loss(batch, predictions, 'validation')
        self.log(
            "validation/total_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=self._use_sync_dist(),
        )
        self.log_metrics(batch, predictions, 'validation')
        
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            if "target" in loss_dict:
                outputs_pred[loss_dict["prediction"]] = predictions[loss_dict["prediction"].lower()].detach().cpu().numpy()
                outputs_target[loss_dict["target"]] = batch[loss_dict["target"].lower()].detach().cpu().numpy()
                
        output_dict = {'pred': outputs_pred, 'target': outputs_target}
        self.validation_step_outputs.append(output_dict)
        return output_dict

    def on_validation_epoch_end(self) -> None:
        """Process and log validation results at the end of an epoch (PL 2.0+ API)."""
        outputs = self._gather_step_outputs(self.validation_step_outputs)

        if self._is_global_zero() and outputs:
            self._plot_prediction_vs_target(outputs, mode='validation')

        # 清空暫存避免 GPU 記憶體洩漏
        self.validation_step_outputs.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """Perform a single test step."""
        torch.set_grad_enabled(self.requires_derivatives)
        self._enable_position_gradients(batch)
        
        processed_values = None
        if self.post_processing is not None:
            predictions = self.post_processing(batch)
            post_processing_name = type(self.post_processing).__name__.split(".")[-1].lower()
            
            if post_processing_name == 'epc_output':
                processed_values = {'epc_mat': predictions['epc_mat'].detach().cpu().numpy()}
            else:
                raise NotImplementedError(f"Post-processing type {post_processing_name} not implemented")
        else:
            predictions = self(batch)
        
        test_loss = self.calculate_loss(batch, predictions, 'test')
        self.log(
            "test/total_loss",
            test_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=self._use_sync_dist(),
        )
        self.log_metrics(batch, predictions, "test")
        
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            if "target" in loss_dict:
                outputs_pred[loss_dict["prediction"]] = predictions[loss_dict["prediction"].lower()].detach().cpu().numpy()  
                outputs_target[loss_dict["target"]] = batch[loss_dict["target"].lower()].detach().cpu().numpy()
                
        output_dict = {
            'pred': outputs_pred, 
            'target': outputs_target, 
            'processed_values': processed_values
        }
        self.test_step_outputs.append(output_dict)
        return output_dict

    def on_test_epoch_end(self) -> None:
        """Process and log test results at the end of testing (PL 2.0+ API)."""
        outputs = self._gather_step_outputs(self.test_step_outputs)

        if self._is_global_zero() and outputs:
            log_dir = self.trainer.logger.log_dir if self.trainer.logger else getattr(self.trainer, 'default_root_dir', './')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            self._save_predictions_and_targets(outputs, log_dir)
            self._plot_prediction_vs_target(outputs, mode='test')
            
            if self.post_processing is not None:
                post_processing_name = type(self.post_processing).__name__.split(".")[-1].lower()
                if post_processing_name == 'epc_output':
                    processed_values = np.concatenate([
                        out['processed_values']["epc_mat"] for out in outputs if out.get('processed_values') is not None
                    ])
                    np.save(os.path.join(log_dir, 'processed_values_epc_mat.npy'), processed_values)

        self.test_step_outputs.clear()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        self._enable_position_gradients(batch)
        representation = self.representation(batch)
        predictions = self.output_module(batch, representation)
        return predictions

    def log_metrics(self, batch: Dict[str, torch.Tensor], 
                   predictions: Dict[str, torch.Tensor], 
                   mode: str) -> None:
        """Log evaluation metrics for the current batch."""
        for metric_dict in self.metrics:
            metric_fn = metric_dict["metric"]
            
            if "target" in metric_dict:
                prediction = predictions[metric_dict["prediction"].lower()]
                target = batch[metric_dict["target"].lower()]
                metric_value = metric_fn(prediction, target).detach().item()
            else:
                metric_value = metric_fn(predictions[metric_dict["prediction"].lower()]).detach().item()
                
            metric_name = getattr(metric_fn, "name", type(metric_fn).__name__.split(".")[-1])
            
            self.log(
                f"{mode}/{metric_name}_{metric_dict['prediction']}",
                metric_value,
                on_step=False,
                on_epoch=True,
                sync_dist=self._use_sync_dist(),
            )

    def configure_optimizers(self) -> Dict:
        """Configure optimizers and learning rate schedulers."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            eps=self.epsilon,
            betas=(self.beta1, self.beta2),
            weight_decay=0.0,
            amsgrad=self.amsgrad
        )
        
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                threshold=1e-6,
                cooldown=self.lr_patience // 2,
                min_lr=1e-6,
            ),
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        
        return [optimizer], [scheduler]

    def _enable_position_gradients(self, batch: Dict[str, torch.Tensor]) -> None:
        """Enable gradients for position vectors if derivatives are required."""
        if self.requires_derivatives and hasattr(batch, 'pos'):
            batch.pos.requires_grad_()

    def _prepare_data_for_scatter_plot(self, pred: np.ndarray, target: np.ndarray) -> tuple:
        """Prepare complex data for scatter plotting and handle subsampling."""
        if (pred.dtype == np.complex64) and (target.dtype == np.complex64):
            for loss_dict in self.losses:
                if hasattr(loss_dict.get('metric', None), 'name'):
                    lossname = loss_dict['metric'].name
                elif loss_dict.get('metric', None) is not None:
                    lossname = type(loss_dict['metric']).__name__.split(".")[-1]
                else:
                    lossname = ""
                    
                if lossname.lower() == 'abs_mae':
                    pred = np.absolute(pred)
                    target = np.absolute(target)
                    break
            else:
                pred = np.concatenate([pred.real, pred.imag], axis=-1)
                target = np.concatenate([target.real, target.imag], axis=-1)
        
        if pred.size > self.max_points_to_scatter:
            random_state = np.random.RandomState(seed=42)
            perm = random_state.permutation(np.arange(pred.size))
            pred = pred.reshape(-1)[perm[:self.max_points_to_scatter]]
            target = target.reshape(-1)[perm[:self.max_points_to_scatter]]
            
        return pred.reshape(-1), target.reshape(-1)

    def _plot_prediction_vs_target(self, step_outputs: List[Dict], mode: str) -> None:
        """Create and log scatter plots comparing predictions to targets."""
        for loss_dict in self.losses:
            if "target" in loss_dict:
                pred_key = loss_dict["prediction"]
                target_key = loss_dict["target"]
                
                if not all(pred_key in out['pred'] and target_key in out['target'] for out in step_outputs):
                    continue
                
                pred = np.concatenate([out['pred'][pred_key] for out in step_outputs])
                target = np.concatenate([out['target'][target_key] for out in step_outputs])
                
                plot_pred, plot_target = self._prepare_data_for_scatter_plot(pred, target)
                
                figure = scatter_plot(plot_pred, plot_target)
                figname = f'PredVSTarget_{pred_key}'
                if self.logger:
                    self.logger.experiment.add_figure(
                        f'{mode}/{figname}', figure, global_step=self.global_step
                    )

    def _save_predictions_and_targets(self, test_outputs: List[Dict], log_dir: str) -> None:
        """Save prediction and target arrays to disk."""
        for loss_dict in self.losses:
            if "target" in loss_dict:
                pred_key = loss_dict["prediction"]
                target_key = loss_dict["target"]
                
                if not all(pred_key in out['pred'] and target_key in out['target'] for out in test_outputs):
                    continue
                
                pred = np.concatenate([out['pred'][pred_key] for out in test_outputs])
                target = np.concatenate([out['target'][target_key] for out in test_outputs])
                
                np.save(os.path.join(log_dir, f'prediction_{pred_key}.npy'), pred)
                np.save(os.path.join(log_dir, f'target_{target_key}.npy'), target)