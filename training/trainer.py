# ABOUTME: Training pipeline for PAMCL model
# ABOUTME: Handles multi-phase training, mixed precision, and gradient accumulation

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from training.losses import PAMCLLoss
from utils.helpers import compute_metrics, save_checkpoint, load_checkpoint


class PAMCLTrainer:
    """Trainer class for PAMCL model.
    
    Features:
    - Multi-phase training (phoneme pre-training, fine-tuning)
    - Mixed precision training for memory efficiency
    - Gradient accumulation for effective larger batches
    - Early stopping with patience
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """Initialize trainer.
        
        Args:
            model: PAMCL model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device (cuda/cpu)
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # DataParallel for multi-GPU training
        self.is_parallel = False
        use_dp = config.get('model', {}).get('use_data_parallel', True)
        if use_dp and device.type == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            self.is_parallel = True
        
        training_cfg = config['training']
        
        # Get class distribution for weighted loss
        num_real = training_cfg.get('num_real', 1)
        num_fake = training_cfg.get('num_fake', 1)
        
        # Loss function with class weighting for imbalanced data
        self.criterion = PAMCLLoss(
            lambda_ce=training_cfg.get('lambda_ce', 1.0),
            lambda_contrastive=training_cfg.get('lambda_contrastive', 0.5),
            lambda_phoneme_consistency=training_cfg.get('lambda_phoneme_consistency', 0.3),
            label_smoothing=training_cfg.get('label_smoothing', 0.1),
            num_real=num_real,
            num_fake=num_fake
        )
        
        # Optimizer setup
        self.setup_optimizer(training_cfg)
        
        # Mixed precision scaler
        self.use_amp = training_cfg.get('mixed_precision', True) and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Gradient accumulation
        self.accumulation_steps = training_cfg.get('gradient_accumulation_steps', 1)
        
        # Checkpointing
        self.checkpoint_dir = Path(training_cfg.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.weight_file_name = training_cfg.get('weight_file_name', 'pamcl')

        # Logging
        log_dir = config.get('logging', {}).get('log_dir', 'logs')
        self.writer = None
        if SummaryWriter is not None and config.get('logging', {}).get('tensorboard', False):
            try:
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                pass
        self.log_every = config.get('logging', {}).get('log_every_steps', 100)
        
        # Early stopping and best-model selection
        self.patience = training_cfg.get('patience', 10)
        self.min_delta = training_cfg.get('min_delta', 0.001)
        self.save_best_by = training_cfg.get('save_best_by', 'loss')  # 'loss' or 'eer'
        self.best_val_loss = float('inf')
        self.best_val_eer = 1.0
        self.patience_counter = 0
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
    def setup_optimizer(self, training_cfg: Dict[str, Any]):
        """Setup optimizer with parameter groups.
        
        Args:
            training_cfg: Training configuration
        """
        # Get parameter groups
        base_model = self.model.module if self.is_parallel else self.model
        param_groups = base_model.get_trainable_params()
        
        # Different learning rates for SSL backbone vs new modules
        phase2_cfg = training_cfg.get('phase2', {})
        lr_ssl = phase2_cfg.get('lr_ssl', 1e-6)
        lr_new = phase2_cfg.get('lr_new', 1e-4)
        weight_decay = phase2_cfg.get('weight_decay', 0.01)
        
        optimizer_params = []
        
        if param_groups['ssl']:
            optimizer_params.append({
                'params': param_groups['ssl'],
                'lr': lr_ssl,
                'weight_decay': weight_decay
            })
        
        if param_groups['new']:
            optimizer_params.append({
                'params': param_groups['new'],
                'lr': lr_new,
                'weight_decay': weight_decay
            })
        
        self.optimizer = torch.optim.AdamW(optimizer_params)
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * training_cfg.get('phase2', {}).get('epochs', 30)
        warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[lr_ssl, lr_new] if param_groups['ssl'] else [lr_new],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch}',
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            audio = batch['audio'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(audio, labels=labels)
                losses = self.criterion(outputs, labels)
                loss = losses['total_loss'] / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation with gradient clipping to prevent vanishing/exploding
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping to prevent exploding gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0,
                        error_if_nonfinite=False
                    )
                    
                    # Skip step if gradients are invalid (NaN/Inf)
                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                    else:
                        print(f"Warning: Skipping step due to non-finite gradients (norm={grad_norm})")
                    
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log gradient norm periodically
                if self.global_step % self.log_every == 0 and self.writer is not None:
                    self.writer.add_scalar('train/grad_norm', grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, self.global_step)
            
            # Record metrics
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key].append(value.item())
            
            # Compute accuracy (including per-class for imbalanced data)
            with torch.no_grad():
                preds = outputs['logits'].argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                epoch_metrics['accuracy'].append(acc)
                
                # Track per-class accuracy for balanced assessment
                real_mask = labels == 0
                fake_mask = labels == 1
                
                if real_mask.sum() > 0:
                    real_acc = (preds[real_mask] == labels[real_mask]).float().mean().item()
                    epoch_metrics['real_acc'].append(real_acc)
                
                if fake_mask.sum() > 0:
                    fake_acc = (preds[fake_mask] == labels[fake_mask]).float().mean().item()
                    epoch_metrics['fake_acc'].append(fake_acc)
            
            # Update progress bar with balanced metrics
            postfix = {
                'loss': f"{losses['total_loss'].item():.4f}",
                'acc': f"{acc:.2f}"
            }
            if 'real_acc' in epoch_metrics and epoch_metrics['real_acc']:
                postfix['real'] = f"{epoch_metrics['real_acc'][-1]:.2f}"
            if 'fake_acc' in epoch_metrics and epoch_metrics['fake_acc']:
                postfix['fake'] = f"{epoch_metrics['fake_acc'][-1]:.2f}"
            progress_bar.set_postfix(postfix)
            
            # Logging
            if self.global_step % self.log_every == 0:
                self._log_metrics(losses, prefix='train')
        
        # Compute epoch averages
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_labels = []
        all_scores = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            audio = batch['audio'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(audio)
                losses = self.criterion(outputs, labels)
            
            total_loss += losses['total_loss'].item()
            num_batches += 1
            
            # Collect predictions
            probs = torch.softmax(outputs['logits'], dim=-1)
            all_scores.extend(probs[:, 1].cpu().numpy())  # Fake probability
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        metrics = compute_metrics(all_labels, all_scores)
        metrics['val_loss'] = total_loss / num_batches
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'val/{key}', value, self.global_step)
        
        return metrics
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> Dict[str, float]:
        """Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            Best validation metrics
        """
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.model, self.optimizer, start_epoch, _ = load_checkpoint(
                resume_from, self.model, self.optimizer, self.device
            )
            print(f"Resumed from epoch {start_epoch}")
        
        best_metrics = {}
        
        # Print training configuration
        print("\n" + "=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Gradient accumulation: {self.accumulation_steps}")
        print(f"Effective batch size: {self.config['training']['batch_size'] * self.accumulation_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Early stopping patience: {self.patience}")
        print("=" * 60 + "\n")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            
            # Compute balanced accuracy
            real_acc = train_metrics.get('real_acc', 0)
            fake_acc = train_metrics.get('fake_acc', 0)
            balanced_acc = (real_acc + fake_acc) / 2 if real_acc and fake_acc else train_metrics.get('accuracy', 0)
            
            # Print detailed epoch summary
            print("\n" + "=" * 60)
            print(f"EPOCH {epoch + 1}/{num_epochs} COMPLETE (Time: {epoch_time:.1f}s)")
            print("=" * 60)
            print(f"  Training:")
            print(f"    Loss:     {train_metrics.get('total_loss', 0):.4f}")
            print(f"    CE Loss:  {train_metrics.get('ce_loss', 0):.4f}")
            if 'contrastive_loss' in train_metrics:
                print(f"    Contrast: {train_metrics.get('contrastive_loss', 0):.4f}")
            print(f"    Accuracy: {train_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"    Real Acc: {real_acc*100:.2f}%  |  Fake Acc: {fake_acc*100:.2f}%")
            print(f"    Balanced: {balanced_acc*100:.2f}%")
            print(f"  Validation:")
            print(f"    Loss:     {val_metrics['val_loss']:.4f}")
            print(f"    EER:      {val_metrics['eer']*100:.2f}%")
            print(f"    AUC:      {val_metrics['auc']:.4f}")
            print(f"    Accuracy: {val_metrics['accuracy']*100:.2f}%")
            
            # Check for improvement (by loss or by EER)
            current_val_loss = val_metrics['val_loss']
            current_eer = val_metrics['eer']
            if self.save_best_by == 'eer':
                improved = current_eer < self.best_val_eer - self.min_delta
                if improved:
                    self.best_val_eer = current_eer
                    self.best_val_loss = current_val_loss
            else:
                improved = current_val_loss < self.best_val_loss - self.min_delta
                if improved:
                    self.best_val_loss = current_val_loss
                    self.best_val_eer = current_eer

            if improved:
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch
                self.patience_counter = 0
                # Save best model checkpoint (handle DataParallel)
                model_to_save = self.model.module if self.is_parallel else self.model
                save_checkpoint(
                    model_to_save, self.optimizer, epoch,
                    val_metrics, str(self.checkpoint_dir),
                    filename='best_model.pt'
                )
                print(f"  -> NEW BEST MODEL SAVED! (EER: {val_metrics['eer']*100:.2f}%)")
            else:
                self.patience_counter += 1
                if not best_metrics:
                    best_metrics = val_metrics.copy()
                    best_metrics['epoch'] = epoch
                print(f"  -> No improvement ({self.patience_counter}/{self.patience})")
            
            # Always save latest checkpoint (for resume after crash/timeout)
            model_to_save = self.model.module if self.is_parallel else self.model
            save_checkpoint(
                model_to_save, self.optimizer, epoch,
                val_metrics, str(self.checkpoint_dir),
                filename='last_checkpoint.pt'
            )

            # Save per-epoch weights to weights/weight_file_name_epoch_N.pt
            epoch_filename = f"{self.weight_file_name}_epoch_{epoch + 1}.pt"
            ckpt_path = save_checkpoint(
                model_to_save, self.optimizer, epoch,
                val_metrics, str(self.checkpoint_dir),
                filename=epoch_filename
            )
            print(f"  -> Weights saved: {ckpt_path}")

            print("=" * 60)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n*** EARLY STOPPING at epoch {epoch + 1} ***")
                break
        
        # Final summary: EER and other metrics after all epochs
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - EER AND METRICS (after {} epochs)".format(num_epochs))
        print("=" * 60)
        if best_metrics:
            print(f"  Best model at epoch: {best_metrics.get('epoch', 0) + 1}")
            print(f"  EER:      {best_metrics.get('eer', 0) * 100:.2f}%")
            print(f"  EER (as fraction): {best_metrics.get('eer', 0):.4f}")
            print(f"  AUC:      {best_metrics.get('auc', 0):.4f}")
            print(f"  Accuracy: {best_metrics.get('accuracy', 0) * 100:.2f}%")
            print(f"  Val loss: {best_metrics.get('val_loss', 0):.4f}")
        print("  Weights saved in: {} ({}_epoch_1.pt ... {}_epoch_{}.pt)".format(
            self.checkpoint_dir, self.weight_file_name, self.weight_file_name, num_epochs))
        print("=" * 60)

        if self.writer is not None:
            self.writer.close()

        return best_metrics
    
    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str = 'train'):
        """Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metric values
            prefix: Metric name prefix
        """
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'{prefix}/{key}', value.item(), self.global_step)
            else:
                self.writer.add_scalar(f'{prefix}/{key}', value, self.global_step)
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'lr/group_{i}', param_group['lr'], self.global_step)
