# ABOUTME: Evaluation script for PAMCL model
# ABOUTME: Computes EER, AUC, and other metrics on test datasets

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import PAMCLModel, PAMCLInference
from data import ASVSpoofDataset, collate_fn
from utils import load_config, load_checkpoint, compute_metrics, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate PAMCL model on test datasets'
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--dataset', type=str, default='eval',
        choices=['dev', 'eval', 'inthewild'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save predictions (optional)'
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> dict:
    """Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Evaluation dataloader
        device: Compute device
        use_amp: Use automatic mixed precision
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_labels = []
    all_scores = []
    all_audio_ids = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        audio = batch['audio'].to(device)
        labels = batch['labels']
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(audio)
        
        # Get fake probability
        probs = torch.softmax(outputs['logits'], dim=-1)
        scores = probs[:, 1].cpu().numpy()  # Probability of fake
        
        all_labels.extend(labels.numpy())
        all_scores.extend(scores)
        
        if 'audio_ids' in batch:
            all_audio_ids.extend(batch['audio_ids'])
    
    # Compute metrics
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    metrics = compute_metrics(all_labels, all_scores)
    
    # Additional statistics
    real_scores = all_scores[all_labels == 0]
    fake_scores = all_scores[all_labels == 1]
    
    metrics['real_score_mean'] = float(np.mean(real_scores)) if len(real_scores) > 0 else 0
    metrics['real_score_std'] = float(np.std(real_scores)) if len(real_scores) > 0 else 0
    metrics['fake_score_mean'] = float(np.mean(fake_scores)) if len(fake_scores) > 0 else 0
    metrics['fake_score_std'] = float(np.std(fake_scores)) if len(fake_scores) > 0 else 0
    
    return metrics, all_labels, all_scores, all_audio_ids


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(str(config_path))
    
    # Get device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = PAMCLModel(config)
    model, _, _, _ = load_checkpoint(args.checkpoint, model, device=str(device))
    model = model.to(device)
    model.eval()
    
    # Determine dataset paths (relative to pamcl folder where this script lives)
    data_cfg = config['data']
    base_dir = Path(__file__).parent

    def resolve_path(cfg_value: str) -> Path:
        p = Path(cfg_value)
        return p if p.is_absolute() else base_dir / cfg_value

    if args.dataset == 'dev':
        audio_dir = str(resolve_path(data_cfg.get('dev_dir', data_cfg['train_dir'])))
        protocol_path = str(resolve_path(data_cfg.get('dev_protocol', data_cfg['train_protocol'])))
    elif args.dataset == 'eval':
        audio_dir = str(resolve_path(data_cfg['eval_dir']))
        protocol_path = str(resolve_path(data_cfg['eval_protocol']))
    elif args.dataset == 'inthewild':
        eval_cfg = config.get('evaluation', {})
        audio_dir = str(resolve_path(eval_cfg.get('inthewild_dir', 'InTheWild')))
        protocol_path = str(resolve_path(eval_cfg.get('inthewild_protocol', 'InTheWild/meta.csv')))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Evaluating on {args.dataset} dataset...")
    print(f"  Audio dir: {audio_dir}")
    print(f"  Protocol: {protocol_path}")
    
    # Create dataset
    dataset = ASVSpoofDataset(
        audio_dir=audio_dir,
        protocol_path=protocol_path,
        sample_rate=data_cfg['sample_rate'],
        max_audio_len=data_cfg['max_audio_len'],
        min_audio_len=data_cfg['min_audio_len'],
        augmentor=None,
        return_phoneme_info=True
    )
    
    dist = dataset.get_label_distribution()
    print(f"  Samples: {len(dataset)} (real: {dist['real']}, fake: {dist['fake']})")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Evaluate
    use_amp = config['training'].get('mixed_precision', True)
    metrics, labels, scores, audio_ids = evaluate(model, dataloader, device, use_amp)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"Evaluation Results - {args.dataset}")
    print("=" * 50)
    print(f"EER:           {metrics['eer'] * 100:.2f}%")
    print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
    print(f"AUC:           {metrics['auc']:.4f}")
    print(f"Accuracy:      {metrics['accuracy'] * 100:.2f}%")
    print()
    print("Score Statistics:")
    print(f"  Real samples: {metrics['real_score_mean']:.4f} +/- {metrics['real_score_std']:.4f}")
    print(f"  Fake samples: {metrics['fake_score_mean']:.4f} +/- {metrics['fake_score_std']:.4f}")
    
    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("audio_id,label,score\n")
            for i in range(len(labels)):
                audio_id = audio_ids[i] if i < len(audio_ids) else str(i)
                f.write(f"{audio_id},{labels[i]},{scores[i]:.6f}\n")
        
        print(f"\nPredictions saved to: {output_path}")


if __name__ == '__main__':
    main()
