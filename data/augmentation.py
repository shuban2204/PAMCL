# ABOUTME: Audio augmentation for PAMCL training
# ABOUTME: RawBoost-inspired augmentations for improved generalization

import random
from typing import Tuple, Optional

import torch
import torchaudio
import numpy as np


class AudioAugmentor:
    """Audio augmentation module for training.
    
    Implements RawBoost-inspired augmentations that operate directly on waveforms.
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        add_noise: bool = True,
        noise_snr_range: Tuple[float, float] = (10, 30),
        speed_perturb: bool = True,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        sample_rate: int = 16000
    ):
        """Initialize augmentor.
        
        Args:
            probability: Probability of applying augmentation
            add_noise: Whether to add Gaussian noise
            noise_snr_range: SNR range for noise addition (dB)
            speed_perturb: Whether to apply speed perturbation
            speed_range: Speed factor range
            sample_rate: Audio sample rate
        """
        self.probability = probability
        self.add_noise = add_noise
        self.noise_snr_range = noise_snr_range
        self.speed_perturb = speed_perturb
        self.speed_range = speed_range
        self.sample_rate = sample_rate
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to waveform.
        
        Args:
            waveform: Input waveform [C, T]
            
        Returns:
            Augmented waveform [C, T']
        """
        if random.random() > self.probability:
            return waveform
        
        # Randomly choose augmentation(s)
        augmentations = []
        
        if self.add_noise and random.random() > 0.5:
            augmentations.append(self._add_noise)
        
        if self.speed_perturb and random.random() > 0.5:
            augmentations.append(self._speed_perturb)
        
        # Apply selected augmentations
        for aug in augmentations:
            waveform = aug(waveform)
        
        return waveform
    
    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR level.
        
        Args:
            waveform: Input waveform [C, T]
            
        Returns:
            Noisy waveform [C, T]
        """
        snr_db = random.uniform(*self.noise_snr_range)
        
        # Calculate signal power
        signal_power = waveform.pow(2).mean()
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        return waveform + noise
    
    def _speed_perturb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply speed perturbation using simple time stretching.
        
        Args:
            waveform: Input waveform [C, T]
            
        Returns:
            Speed-perturbed waveform [C, T']
        """
        speed_factor = random.uniform(*self.speed_range)
        
        # Simple interpolation-based speed change (memory efficient)
        original_length = waveform.shape[-1]
        new_length = int(original_length / speed_factor)
        
        if new_length == original_length:
            return waveform
        
        # Use linear interpolation for speed change
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0),  # [1, C, T]
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)  # [C, T']
        
        return waveform


class RawBoostAugmentor:
    """RawBoost Algorithm 3 implementation.
    
    Based on: "Improved RawNet with Feature Map Scaling for 
    Text-independent Speaker Verification using Raw Waveforms"
    
    Applies linear and non-linear convolutive noise.
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        n_lin: int = 5,
        n_nlin: int = 5,
        lin_min_G: float = 0.5,
        lin_max_G: float = 2.0,
        nlin_min_B: float = 0.1,
        nlin_max_B: float = 0.5,
        sample_rate: int = 16000
    ):
        """Initialize RawBoost augmentor.
        
        Args:
            probability: Probability of applying augmentation
            n_lin: Number of linear notch filters
            n_nlin: Number of non-linear notch filters
            lin_min_G: Minimum linear filter gain
            lin_max_G: Maximum linear filter gain
            nlin_min_B: Minimum non-linear bias
            nlin_max_B: Maximum non-linear bias
            sample_rate: Audio sample rate
        """
        self.probability = probability
        self.n_lin = n_lin
        self.n_nlin = n_nlin
        self.lin_min_G = lin_min_G
        self.lin_max_G = lin_max_G
        self.nlin_min_B = nlin_min_B
        self.nlin_max_B = nlin_max_B
        self.sample_rate = sample_rate
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply RawBoost augmentation.
        
        Args:
            waveform: Input waveform [C, T]
            
        Returns:
            Augmented waveform [C, T]
        """
        if random.random() > self.probability:
            return waveform
        
        # Convert to numpy for processing
        x = waveform.numpy()
        
        # Apply linear convolutive noise
        x = self._lin_convolutive_noise(x)
        
        # Apply non-linear convolutive noise
        x = self._nlin_convolutive_noise(x)
        
        return torch.from_numpy(x)
    
    def _lin_convolutive_noise(self, x: np.ndarray) -> np.ndarray:
        """Apply linear convolutive noise via notch filters."""
        for _ in range(self.n_lin):
            # Random notch frequency
            f0 = random.randint(100, self.sample_rate // 4)
            
            # Random Q factor
            Q = random.uniform(5, 20)
            
            # Random gain
            G = random.uniform(self.lin_min_G, self.lin_max_G)
            
            # Apply notch filter
            x = self._apply_notch(x, f0, Q, G)
        
        return x
    
    def _nlin_convolutive_noise(self, x: np.ndarray) -> np.ndarray:
        """Apply non-linear convolutive noise."""
        for _ in range(self.n_nlin):
            # Random bias
            bias = random.uniform(self.nlin_min_B, self.nlin_max_B)
            
            # Non-linear transformation
            x = x + bias * np.sign(x) * (x ** 2)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(x))
        if max_val > 1.0:
            x = x / max_val
        
        return x
    
    def _apply_notch(
        self,
        x: np.ndarray,
        f0: int,
        Q: float,
        G: float
    ) -> np.ndarray:
        """Apply a notch filter.
        
        Args:
            x: Input signal
            f0: Notch center frequency
            Q: Quality factor
            G: Gain
            
        Returns:
            Filtered signal
        """
        from scipy.signal import iirnotch, lfilter
        
        # Design notch filter
        w0 = f0 / (self.sample_rate / 2)
        if w0 >= 1.0:
            w0 = 0.99  # Clamp to valid range
        
        b, a = iirnotch(w0, Q)
        
        # Apply filter with gain
        x_filtered = lfilter(b, a, x)
        x = G * x_filtered + (1 - G) * x
        
        return x
