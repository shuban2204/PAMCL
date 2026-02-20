# ABOUTME: Balanced batch sampler for handling class imbalance
# ABOUTME: Ensures each batch contains equal numbers of real and fake samples

import random
from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """Sampler that ensures each batch has balanced class distribution.
    
    For binary classification with imbalanced data, this sampler:
    - Maintains separate indices for each class
    - Samples equally from both classes for each batch
    - Oversamples the minority class to match the majority class
    """
    
    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        drop_last: bool = True
    ):
        """Initialize balanced sampler.
        
        Args:
            labels: List of class labels (0 or 1) for each sample
            batch_size: Total batch size (will be split equally between classes)
            drop_last: Whether to drop the last incomplete batch
        """
        super().__init__(labels)
        
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Split indices by class
        self.class_indices = {0: [], 1: []}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        self.num_real = len(self.class_indices[0])
        self.num_fake = len(self.class_indices[1])
        
        # Samples per class per batch
        self.samples_per_class = batch_size // 2
        
        # Number of batches based on minority class (with oversampling)
        # Use the larger class to determine epoch length
        self.num_batches = max(self.num_real, self.num_fake) // self.samples_per_class
        
        print(f"BalancedBatchSampler: {self.num_real} real, {self.num_fake} fake")
        print(f"  Batch size: {batch_size} ({self.samples_per_class} per class)")
        print(f"  Batches per epoch: {self.num_batches}")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate balanced batches.
        
        Yields:
            List of indices for each batch
        """
        # Shuffle indices for each class
        real_indices = self.class_indices[0].copy()
        fake_indices = self.class_indices[1].copy()
        
        random.shuffle(real_indices)
        random.shuffle(fake_indices)
        
        # Pointers for each class
        real_ptr = 0
        fake_ptr = 0
        
        for _ in range(self.num_batches):
            batch = []
            
            # Sample from real class (with wraparound/oversampling)
            for _ in range(self.samples_per_class):
                if real_ptr >= len(real_indices):
                    # Reshuffle and reset pointer (oversampling)
                    random.shuffle(real_indices)
                    real_ptr = 0
                batch.append(real_indices[real_ptr])
                real_ptr += 1
            
            # Sample from fake class
            for _ in range(self.samples_per_class):
                if fake_ptr >= len(fake_indices):
                    random.shuffle(fake_indices)
                    fake_ptr = 0
                batch.append(fake_indices[fake_ptr])
                fake_ptr += 1
            
            # Shuffle the batch so real/fake aren't in predictable order
            random.shuffle(batch)
            
            yield batch
    
    def __len__(self) -> int:
        return self.num_batches


class BalancedRandomSampler(Sampler):
    """Alternative: Random sampler with class balancing via oversampling.
    
    Unlike BalancedBatchSampler, this returns individual indices
    and relies on DataLoader's batching.
    """
    
    def __init__(self, labels: List[int], num_samples: int = None):
        """Initialize sampler.
        
        Args:
            labels: List of class labels
            num_samples: Total samples per epoch (default: 2x minority class)
        """
        super().__init__(labels)
        
        self.class_indices = {0: [], 1: []}
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        self.num_real = len(self.class_indices[0])
        self.num_fake = len(self.class_indices[1])
        
        # Default: sample 2x minority class samples total
        if num_samples is None:
            minority_size = min(self.num_real, self.num_fake)
            self.num_samples = minority_size * 4  # 2x from each class
        else:
            self.num_samples = num_samples
    
    def __iter__(self) -> Iterator[int]:
        """Generate balanced sample indices."""
        # Alternate between classes
        indices = []
        
        real_indices = self.class_indices[0].copy()
        fake_indices = self.class_indices[1].copy()
        
        for i in range(self.num_samples):
            if i % 2 == 0:
                # Sample from real (with replacement if needed)
                idx = random.choice(real_indices)
            else:
                # Sample from fake
                idx = random.choice(fake_indices)
            indices.append(idx)
        
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples
