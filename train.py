# ABOUTME: Main training script for PAMCL model
# ABOUTME: Handles configuration loading, model initialization, and training

import argparse
import os
import sys
from pathlib import Path

import torch

# CUDA performance optimizations (auto-adapts: no-ops on unsupported GPUs)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True          # Auto-tune convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmul (Ampere+)
    torch.backends.cudnn.allow_tf32 = True         # TF32 cuDNN ops (Ampere+)
    torch.set_float32_matmul_precision('high')     # Prefer TF32 over FP32

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import PAMCLModel
from data import ASVSpoofDataset, collate_fn, AudioAugmentor
from training import PAMCLTrainer
from utils import load_config, setup_logging, set_seed, get_device, count_parameters
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PAMCL model for audio deepfake detection'
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode with smaller batch size'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(str(config_path))
    
    # Setup logging
    logger = setup_logging(
        config.get('logging', {}).get('log_dir', 'logs'),
        name='pamcl_train'
    )
    logger.info(f"Loaded configuration from {config_path}")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Set random seed: {args.seed}")
    
    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")
    
    # Debug mode adjustments
    if args.debug:
        logger.info("Running in debug mode")
        config['training']['batch_size'] = 2
        config['training']['phase2']['epochs'] = 2
        config['data']['num_workers'] = 0
    
    # Create augmentor for training
    aug_cfg = config.get('augmentation', {})
    if aug_cfg.get('enabled', True):
        augmentor = AudioAugmentor(
            probability=aug_cfg.get('probability', 0.5),
            add_noise=aug_cfg.get('add_noise', True),
            noise_snr_range=aug_cfg.get('noise_snr_range', [10, 30]),
            speed_perturb=aug_cfg.get('speed_perturb', True),
            speed_range=aug_cfg.get('speed_range', [0.9, 1.1]),
            sample_rate=config['data']['sample_rate']
        )
        logger.info("Audio augmentation enabled")
    else:
        augmentor = None
    
    # Create datasets with train/val split
    data_cfg = config['data']
    base_dir = Path(__file__).parent.parent  # Research directory
    
    # Resolve paths (handle both absolute and relative paths)
    train_dir_cfg = data_cfg['train_dir']
    protocol_cfg = data_cfg['train_protocol']
    
    if Path(train_dir_cfg).is_absolute():
        audio_dir = Path(train_dir_cfg)
    else:
        audio_dir = base_dir / train_dir_cfg
        
    if Path(protocol_cfg).is_absolute():
        protocol_path = Path(protocol_cfg)
    else:
        protocol_path = base_dir / protocol_cfg
    
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Protocol file: {protocol_path}")
    
    # Check if files exist
    if not protocol_path.exists():
        logger.error(f"Protocol file not found: {protocol_path}")
        return
    
    logger.info("Loading full dataset (filtering to existing files)...")
    full_dataset = ASVSpoofDataset(
        audio_dir=str(audio_dir),
        protocol_path=str(protocol_path),
        sample_rate=data_cfg['sample_rate'],
        max_audio_len=data_cfg['max_audio_len'],
        min_audio_len=data_cfg['min_audio_len'],
        augmentor=None,
        return_phoneme_info=True,
        subset_indices=None
    )
    logger.info(f"Found {len(full_dataset)} valid audio files")
    
    if len(full_dataset) == 0:
        logger.error("No valid audio files found!")
        return
    
    # Create train/val split from the filtered dataset
    num_samples = len(full_dataset)
    import random as rand_module
    indices = list(range(num_samples))
    rand_module.seed(args.seed)
    rand_module.shuffle(indices)
    
    split_point = int(len(indices) * 0.8)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    logger.info(f"Created train/val split: {len(train_indices)} train, {len(val_indices)} val")
    
    # Create train and val datasets by subsetting the full dataset's samples
    # This avoids re-filtering files
    all_samples = full_dataset.samples
    
    logger.info("Creating training dataset...")
    train_dataset = ASVSpoofDataset.__new__(ASVSpoofDataset)
    train_dataset.audio_dir = full_dataset.audio_dir
    train_dataset.sample_rate = full_dataset.sample_rate
    train_dataset.max_audio_len = full_dataset.max_audio_len
    train_dataset.min_audio_len = full_dataset.min_audio_len
    train_dataset.augmentor = augmentor
    train_dataset.return_phoneme_info = True
    train_dataset.protocol_format = full_dataset.protocol_format
    train_dataset.samples = [all_samples[i] for i in train_indices]
    
    train_dist = train_dataset.get_label_distribution()
    logger.info(f"Training set: {len(train_dataset)} samples (real: {train_dist['real']}, fake: {train_dist['fake']})")
    
    logger.info("Creating validation dataset...")
    dev_dataset = ASVSpoofDataset.__new__(ASVSpoofDataset)
    dev_dataset.audio_dir = full_dataset.audio_dir
    dev_dataset.sample_rate = full_dataset.sample_rate
    dev_dataset.max_audio_len = full_dataset.max_audio_len
    dev_dataset.min_audio_len = full_dataset.min_audio_len
    dev_dataset.augmentor = None
    dev_dataset.return_phoneme_info = False
    dev_dataset.protocol_format = full_dataset.protocol_format
    dev_dataset.samples = [all_samples[i] for i in val_indices]
    
    dev_dist = dev_dataset.get_label_distribution()
    logger.info(f"Validation set: {len(dev_dataset)} samples (real: {dev_dist['real']}, fake: {dev_dist['fake']})")
    
    # Create dataloaders with balanced sampling for imbalanced data
    training_cfg = config['training']
    
    # Use balanced batch sampler to handle class imbalance
    from data.balanced_sampler import BalancedBatchSampler
    
    # Get labels for balanced sampling
    train_labels = [s['label'] for s in train_dataset.samples]
    balanced_sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=training_cfg['batch_size'],
        drop_last=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=balanced_sampler,  # Use balanced sampler instead of shuffle
        num_workers=data_cfg.get('num_workers', 0),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    logger.info("Initializing PAMCL model...")
    model = PAMCLModel(config)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # With balanced sampling, we don't need aggressive class weighting
    # Set equal weights since batches are now balanced
    config['training']['num_real'] = 1  # Equal weight
    config['training']['num_fake'] = 1  # Equal weight
    
    logger.info(f"Original class imbalance: 1:{train_dist['fake']/max(train_dist['real'],1):.1f} (real:fake)")
    logger.info("Using balanced batch sampling - each batch has equal real/fake samples")
    
    # Create trainer
    trainer = PAMCLTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=dev_loader,
        device=device
    )
    
    # Train
    num_epochs = training_cfg.get('phase2', {}).get('epochs', 30)
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    best_metrics = trainer.train(
        num_epochs=num_epochs,
        resume_from=args.resume
    )
    
    # Print final results
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    logger.info(f"Best Validation EER: {best_metrics.get('eer', 0):.4f}")
    logger.info(f"Best Validation AUC: {best_metrics.get('auc', 0):.4f}")
    logger.info(f"Best Validation Accuracy: {best_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Model saved to: {config['training']['checkpoint_dir']}/best_model.pt")


if __name__ == '__main__':
    main()
