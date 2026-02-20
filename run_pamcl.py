# ABOUTME: Single entry point to train PAMCL: 5 epochs, EER/metrics printed, weights in weights/pamcl_epoch_N.pt
# ABOUTME: Upload pamcl + datasets folder to AMD instance; paths in config are relative to this folder.

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import PAMCLModel
from data import ASVSpoofDataset, collate_fn, AudioAugmentor
from data.balanced_sampler import BalancedBatchSampler
from training import PAMCLTrainer
from utils import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
    count_parameters,
    compute_metrics,
    load_checkpoint,
)


def resolve_path(cfg_value: str, base_dir: Path) -> Path:
    """Resolve path: use as-is if absolute, else relative to base_dir."""
    p = Path(cfg_value)
    return p if p.is_absolute() else base_dir / cfg_value


def build_datasets_and_loaders(config: dict, base_dir: Path, logger):
    """Build train/val from train_dir+train_protocol (80/20) and eval dataset."""
    data_cfg = config["data"]
    training_cfg = config["training"]

    train_dir_cfg = data_cfg["train_dir"]
    protocol_cfg = data_cfg["train_protocol"]
    audio_dir = resolve_path(train_dir_cfg, base_dir)
    protocol_path = resolve_path(protocol_cfg, base_dir)

    if not protocol_path.exists():
        logger.error(f"Protocol file not found: {protocol_path}")
        raise FileNotFoundError(str(protocol_path))

    logger.info(f"Train audio dir: {audio_dir}")
    logger.info(f"Train protocol: {protocol_path}")

    full_dataset = ASVSpoofDataset(
        audio_dir=str(audio_dir),
        protocol_path=str(protocol_path),
        sample_rate=data_cfg["sample_rate"],
        max_audio_len=data_cfg["max_audio_len"],
        min_audio_len=data_cfg["min_audio_len"],
        augmentor=None,
        return_phoneme_info=True,
        subset_indices=None,
    )
    logger.info(f"Total samples: {len(full_dataset)}")

    if len(full_dataset) == 0:
        raise ValueError("No valid audio files found.")

    # 80/20 train/val split
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split_point = int(num_samples * 0.8)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    aug_cfg = config.get("augmentation", {})
    augmentor = None
    if aug_cfg.get("enabled", True):
        augmentor = AudioAugmentor(
            probability=aug_cfg.get("probability", 0.5),
            add_noise=aug_cfg.get("add_noise", True),
            noise_snr_range=tuple(aug_cfg.get("noise_snr_range", [10, 30])),
            speed_perturb=aug_cfg.get("speed_perturb", True),
            speed_range=tuple(aug_cfg.get("speed_range", [0.9, 1.1])),
            sample_rate=data_cfg["sample_rate"],
        )

    all_samples = full_dataset.samples

    train_dataset = ASVSpoofDataset.__new__(ASVSpoofDataset)
    train_dataset.audio_dir = full_dataset.audio_dir
    train_dataset.sample_rate = full_dataset.sample_rate
    train_dataset.max_audio_len = full_dataset.max_audio_len
    train_dataset.min_audio_len = full_dataset.min_audio_len
    train_dataset.augmentor = augmentor
    train_dataset.return_phoneme_info = True
    train_dataset.protocol_format = full_dataset.protocol_format
    train_dataset.samples = [all_samples[i] for i in train_indices]

    val_dataset = ASVSpoofDataset.__new__(ASVSpoofDataset)
    val_dataset.audio_dir = full_dataset.audio_dir
    val_dataset.sample_rate = full_dataset.sample_rate
    val_dataset.max_audio_len = full_dataset.max_audio_len
    val_dataset.min_audio_len = full_dataset.min_audio_len
    val_dataset.augmentor = None
    val_dataset.return_phoneme_info = False
    val_dataset.protocol_format = full_dataset.protocol_format
    val_dataset.samples = [all_samples[i] for i in val_indices]

    batch_size = training_cfg["batch_size"]
    train_labels = [s["label"] for s in train_dataset.samples]
    balanced_sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=batch_size,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=balanced_sampler,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    config["training"]["num_real"] = 1
    config["training"]["num_fake"] = 1

    eval_dataset = None
    eval_loader = None
    eval_dir_cfg = data_cfg.get("eval_dir")
    eval_protocol_cfg = data_cfg.get("eval_protocol")
    if eval_dir_cfg and eval_protocol_cfg:
        eval_audio_dir = resolve_path(eval_dir_cfg, base_dir)
        eval_protocol_path = resolve_path(eval_protocol_cfg, base_dir)
        if eval_protocol_path.exists():
            eval_dataset = ASVSpoofDataset(
                audio_dir=str(eval_audio_dir),
                protocol_path=str(eval_protocol_path),
                sample_rate=data_cfg["sample_rate"],
                max_audio_len=data_cfg["max_audio_len"],
                min_audio_len=data_cfg["min_audio_len"],
                augmentor=None,
                return_phoneme_info=False,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=data_cfg.get("num_workers", 0),
                collate_fn=collate_fn,
                pin_memory=True,
            )
            logger.info(f"Eval set: {len(eval_dataset)} samples from {eval_audio_dir}")
        else:
            logger.warning(f"Eval protocol not found: {eval_protocol_path}; skipping eval set.")

    return train_loader, val_loader, eval_loader


@torch.no_grad()
def evaluate_model(model, dataloader, device, use_amp=True):
    """Run model on dataloader and return (metrics, all_labels, all_scores)."""
    model.eval()
    all_labels = []
    all_scores = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        audio = batch["audio"].to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(audio)
        probs = torch.softmax(outputs["logits"], dim=-1)
        scores = probs[:, 1].cpu().numpy()
        all_labels.extend(batch["labels"].numpy())
        all_scores.extend(scores)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    metrics = compute_metrics(all_labels, all_scores)
    return metrics, all_labels, all_scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PAMCL and evaluate EER (target 1-1.5%% on ASVSpoof 2021 LA)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run_amd.yaml",
        help="Config YAML (default: configs/run_amd.yaml for 5 epochs, weights/ saving)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Only run evaluation on this checkpoint",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final evaluation on eval set after training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).parent
    config_path = base_dir / args.config
    config = load_config(str(config_path))

    log_dir = config.get("logging", {}).get("log_dir", "logs")
    logger = setup_logging(log_dir, name="run_pamcl")
    logger.info(f"Config: {config_path}")

    set_seed(args.seed)
    device = get_device(config)
    logger.info(f"Device: {device}")

    train_loader, val_loader, eval_loader = build_datasets_and_loaders(config, base_dir, logger)

    if args.eval_only:
        logger.info(f"Eval-only mode: loading {args.eval_only}")
        model = PAMCLModel(config)
        model, _, _, _ = load_checkpoint(args.eval_only, model, device=str(device))
        model = model.to(device)
        use_amp = config["training"].get("mixed_precision", True)
        if eval_loader is not None:
            metrics, _, _ = evaluate_model(model, eval_loader, device, use_amp)
            logger.info(f"Eval set EER: {metrics['eer']*100:.2f}%%")
            logger.info(f"Eval set AUC: {metrics['auc']:.4f}")
        else:
            metrics, _, _ = evaluate_model(model, val_loader, device, use_amp)
            logger.info(f"Val set EER: {metrics['eer']*100:.2f}%%")
        return

    model = PAMCLModel(config)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    trainer = PAMCLTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    num_epochs = config["training"].get("phase2", {}).get("epochs", 30)
    best_metrics = trainer.train(num_epochs=num_epochs, resume_from=args.resume)

    logger.info("Training complete.")
    logger.info(f"Best val EER: {best_metrics.get('eer', 0)*100:.2f}%%")
    logger.info(f"Best val AUC: {best_metrics.get('auc', 0):.4f}")

    if args.skip_eval or eval_loader is None:
        return

    checkpoint_dir = Path(config["training"].get("checkpoint_dir", "checkpoints"))
    best_ckpt = checkpoint_dir / "best_model.pt"
    if not best_ckpt.exists():
        logger.warning("No best_model.pt found; skipping final eval.")
        return

    logger.info(f"Loading best checkpoint for final eval: {best_ckpt}")
    model = PAMCLModel(config)
    model, _, _, _ = load_checkpoint(str(best_ckpt), model, device=str(device))
    model = model.to(device)
    use_amp = config["training"].get("mixed_precision", True)
    metrics, _, _ = evaluate_model(model, eval_loader, device, use_amp)

    logger.info("=" * 50)
    logger.info("FINAL EVAL SET RESULTS (target EER 1-1.5%%)")
    logger.info("=" * 50)
    logger.info(f"EER:        {metrics['eer']*100:.2f}%%")
    logger.info(f"EER thresh: {metrics['eer_threshold']:.4f}")
    logger.info(f"AUC:        {metrics['auc']:.4f}")
    logger.info(f"Accuracy:   {metrics['accuracy']*100:.2f}%%")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
