# ABOUTME: ASVSpoof dataset loader for audio deepfake detection
# ABOUTME: Supports ASVSpoof 2019 LA and 2021 LA/DF protocols with efficient audio loading

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ASVSpoofDataset(Dataset):
    """ASVSpoof dataset for audio deepfake detection.
    
    Supports ASVSpoof 2019 LA and 2021 LA/DF protocols.
    Labels: 0 = bonafide (real), 1 = spoof (fake)
    """
    
    def __init__(
        self,
        audio_dir: str,
        protocol_path: str,
        sample_rate: int = 16000,
        max_audio_len: int = 64000,
        min_audio_len: int = 16000,
        augmentor: Optional[Any] = None,
        return_phoneme_info: bool = False,
        protocol_format: str = "auto",
        subset_indices: Optional[List[int]] = None
    ):
        """Initialize ASVSpoof dataset.
        
        Args:
            audio_dir: Directory containing audio files (.flac)
            protocol_path: Path to protocol file (CM protocols)
            sample_rate: Target sample rate
            max_audio_len: Maximum audio length in samples (truncate if longer)
            min_audio_len: Minimum audio length in samples (pad if shorter)
            augmentor: Optional audio augmentor for training
            return_phoneme_info: If True, return speaker ID for phoneme prototype building
            protocol_format: "2019" for ASVSpoof2019, "2021" for ASVSpoof2021, "auto" to detect
            subset_indices: Optional list of indices to use (for train/val split)
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len
        self.augmentor = augmentor
        self.return_phoneme_info = return_phoneme_info
        
        # Detect or set protocol format
        self.protocol_format = self._detect_format(protocol_path) if protocol_format == "auto" else protocol_format
        
        # Parse protocol file
        all_samples = self._parse_protocol(protocol_path)
        
        # Apply subset if specified
        if subset_indices is not None:
            self.samples = [all_samples[i] for i in subset_indices]
        else:
            self.samples = all_samples
        
    def _detect_format(self, protocol_path: str) -> str:
        """Detect protocol format by reading first line."""
        with open(protocol_path, 'r') as f:
            first_line = f.readline().strip()
            parts = first_line.split()
            
            # ASVSpoof 2021 has more columns and different format
            if len(parts) >= 6 and parts[5] in ['spoof', 'bonafide']:
                return "2021"
            # ASVSpoof 2019 format
            elif len(parts) >= 4 and parts[-1] in ['spoof', 'bonafide']:
                return "2019"
            else:
                # Default to 2021
                return "2021"
    
    def _parse_protocol(self, protocol_path: str) -> List[Dict[str, Any]]:
        """Parse ASVSpoof CM protocol file."""
        samples = []
        
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                if self.protocol_format == "2021":
                    # ASVSpoof 2021 format:
                    # SPEAKER AUDIO_ID CODEC SOURCE ATTACK_TYPE LABEL ...
                    speaker_id = parts[0]
                    audio_id = parts[1]
                    attack_type = parts[4] if len(parts) > 4 else "-"
                    label_str = parts[5] if len(parts) > 5 else "spoof"
                else:
                    # ASVSpoof 2019 format:
                    # SPEAKER AUDIO_ID - ATTACK_TYPE LABEL
                    speaker_id = parts[0]
                    audio_id = parts[1]
                    attack_type = parts[3] if len(parts) > 3 else "-"
                    label_str = parts[-1]
                
                # Convert label to binary: bonafide=0, spoof=1
                label = 0 if label_str == "bonafide" else 1
                
                # Check if audio file exists (filter out missing files)
                audio_path = self.audio_dir / f"{audio_id}.flac"
                if not audio_path.exists():
                    continue
                
                samples.append({
                    'speaker_id': speaker_id,
                    'audio_id': audio_id,
                    'attack_type': attack_type,
                    'label': label
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        sample = self.samples[idx]
        audio_path = self.audio_dir / f"{sample['audio_id']}.flac"
        
        # Handle missing files gracefully
        if not audio_path.exists():
            # Try alternative paths
            alt_paths = [
                self.audio_dir / f"{sample['audio_id']}.wav",
                self.audio_dir / "flac" / f"{sample['audio_id']}.flac",
            ]
            for alt in alt_paths:
                if alt.exists():
                    audio_path = alt
                    break
        
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            # Return silence if file not found
            print(f"Warning: Could not load {audio_path}: {e}")
            waveform = torch.zeros(1, self.max_audio_len)
            sr = self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Store original length
        original_len = waveform.shape[1]
        
        # Apply augmentation
        if self.augmentor is not None:
            waveform = self.augmentor(waveform)
        
        # Truncate or pad
        waveform = self._pad_or_truncate(waveform)
        
        result = {
            'audio': waveform.squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'audio_len': torch.tensor(min(original_len, self.max_audio_len), dtype=torch.long)
        }
        
        if self.return_phoneme_info:
            result['speaker_id'] = sample['speaker_id']
            result['audio_id'] = sample['audio_id']
        
        return result
    
    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or truncate waveform to target length."""
        length = waveform.shape[1]
        
        if length > self.max_audio_len:
            start = np.random.randint(0, length - self.max_audio_len + 1)
            waveform = waveform[:, start:start + self.max_audio_len]
        elif length < self.max_audio_len:
            padding = self.max_audio_len - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in dataset."""
        real_count = sum(1 for s in self.samples if s['label'] == 0)
        fake_count = len(self.samples) - real_count
        return {'real': real_count, 'fake': fake_count}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    audio = torch.stack([item['audio'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    audio_lens = torch.stack([item['audio_len'] for item in batch])
    
    result = {
        'audio': audio,
        'labels': labels,
        'audio_lens': audio_lens
    }
    
    if 'speaker_id' in batch[0]:
        result['speaker_ids'] = [item['speaker_id'] for item in batch]
    if 'audio_id' in batch[0]:
        result['audio_ids'] = [item['audio_id'] for item in batch]
    
    return result


def create_train_val_split(
    protocol_path: str,
    train_samples: int = 50000,
    val_samples: int = 7000,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Create train/val split indices from a protocol file.
    
    Uses fixed sample counts with stratified sampling to preserve
    the bonafide/spoof label ratio in both splits.
    
    Args:
        protocol_path: Path to protocol file
        train_samples: Number of samples for training (default: 50000)
        val_samples: Number of samples for validation (default: 7000)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices)
    """
    # Parse labels for stratified split
    labels = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            # Detect label position (2021: col 5, 2019: last col)
            label_str = parts[5] if len(parts) > 5 and parts[5] in ('spoof', 'bonafide') else parts[-1]
            labels.append(0 if label_str == 'bonafide' else 1)
    
    total = len(labels)
    requested = train_samples + val_samples
    
    if requested > total:
        print(f"  Warning: Requested {requested} samples but only {total} available.")
        print(f"  Adjusting to {total} total with proportional split.")
        ratio = train_samples / requested
        train_samples = int(total * ratio)
        val_samples = total - train_samples
    
    # Stratified split: separate indices by label
    real_indices = [i for i, l in enumerate(labels) if l == 0]
    fake_indices = [i for i, l in enumerate(labels) if l == 1]
    
    rng = random.Random(seed)
    rng.shuffle(real_indices)
    rng.shuffle(fake_indices)
    
    # Compute per-class counts proportional to label distribution
    real_ratio = len(real_indices) / total
    
    train_real = int(train_samples * real_ratio)
    train_fake = train_samples - train_real
    val_real = int(val_samples * real_ratio)
    val_fake = val_samples - val_real
    
    # Clamp to available
    train_real = min(train_real, len(real_indices))
    train_fake = min(train_fake, len(fake_indices))
    val_real = min(val_real, len(real_indices) - train_real)
    val_fake = min(val_fake, len(fake_indices) - train_fake)
    
    train_indices = real_indices[:train_real] + fake_indices[:train_fake]
    val_indices = real_indices[train_real:train_real + val_real] + \
                  fake_indices[train_fake:train_fake + val_fake]
    
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    print(f"  Split: {len(train_indices)} train ({train_real} real + {train_fake} fake) "
          f"| {len(val_indices)} val ({val_real} real + {val_fake} fake)")
    
    return train_indices, val_indices


def create_dataloaders(
    config: Dict[str, Any],
    augmentor: Optional[Any] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, dev, and eval dataloaders."""
    data_cfg = config['data']
    training_cfg = config['training']
    
    # Create train/val split from the available data
    protocol_path = data_cfg['train_protocol']
    train_samples = data_cfg.get('train_samples', 50000)
    val_samples = data_cfg.get('val_samples', 7000)
    train_indices, val_indices = create_train_val_split(
        protocol_path, train_samples=train_samples, val_samples=val_samples
    )
    
    # Training dataset
    train_dataset = ASVSpoofDataset(
        audio_dir=data_cfg['train_dir'],
        protocol_path=protocol_path,
        sample_rate=data_cfg['sample_rate'],
        max_audio_len=data_cfg['max_audio_len'],
        min_audio_len=data_cfg['min_audio_len'],
        augmentor=augmentor,
        return_phoneme_info=True,
        subset_indices=train_indices
    )
    
    # Validation dataset
    val_dataset = ASVSpoofDataset(
        audio_dir=data_cfg['train_dir'],
        protocol_path=protocol_path,
        sample_rate=data_cfg['sample_rate'],
        max_audio_len=data_cfg['max_audio_len'],
        min_audio_len=data_cfg['min_audio_len'],
        augmentor=None,
        return_phoneme_info=False,
        subset_indices=val_indices
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, None
