# ABOUTME: Contrastive learning module for phoneme-level representation learning
# ABOUTME: Implements InfoNCE loss with phoneme prototypes for domain-invariant features

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class ContrastiveModule(nn.Module):
    """Contrastive learning module for phoneme-level features.
    
    Key concept: Learn phoneme representations where:
    - Same phoneme from different speakers cluster together (positive pairs)
    - Different phonemes are pushed apart (negative pairs)
    - Real phonemes form tight clusters, fake phonemes are outliers
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07,
        use_queue: bool = True,
        queue_size: int = 4096
    ):
        """Initialize contrastive module.
        
        Args:
            input_dim: Input feature dimension
            projection_dim: Projection head output dimension
            temperature: InfoNCE temperature parameter
            use_queue: Whether to use a memory queue for negatives
            queue_size: Size of memory queue
        """
        super().__init__()
        
        self.temperature = temperature
        self.use_queue = use_queue
        self.queue_size = queue_size
        self.projection_dim = projection_dim
        
        # Projection head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )
        
        # Memory queue for negative samples (if enabled)
        if use_queue:
            self.register_buffer('queue', torch.randn(queue_size, projection_dim))
            self.queue = F.normalize(self.queue, dim=1)
            self.register_buffer('queue_labels', torch.zeros(queue_size, dtype=torch.long))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(
        self,
        features: torch.Tensor,
        phoneme_labels: torch.Tensor,
        is_real: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute contrastive loss.
        
        Args:
            features: Phoneme features [N, D]
            phoneme_labels: Phoneme class labels [N]
            is_real: Binary mask indicating real samples [N]
            
        Returns:
            Tuple of:
                - loss: Contrastive loss scalar
                - metrics: Dictionary of additional metrics
        """
        # Project features
        projections = self.projection_head(features)  # [N, projection_dim]
        projections = F.normalize(projections, dim=1)
        
        # Compute InfoNCE loss
        loss, metrics = self._infonce_loss(projections, phoneme_labels, is_real)
        
        # Update queue
        if self.use_queue and self.training:
            self._update_queue(projections, phoneme_labels)
        
        return loss, metrics
    
    def _infonce_loss(
        self,
        projections: torch.Tensor,
        phoneme_labels: torch.Tensor,
        is_real: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute InfoNCE contrastive loss.
        
        Positive pairs: same phoneme, both real
        Negative pairs: different phonemes, or fake samples
        
        Args:
            projections: Normalized projected features [N, D]
            phoneme_labels: Phoneme class labels [N]
            is_real: Binary mask for real samples [N]
            
        Returns:
            Tuple of loss and metrics
        """
        batch_size = projections.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=projections.device, requires_grad=True), {}
        
        # Compute pairwise similarity
        sim_matrix = torch.matmul(projections, projections.T) / self.temperature  # [N, N]
        
        # Create positive mask: same phoneme AND both real
        phoneme_match = phoneme_labels.unsqueeze(0) == phoneme_labels.unsqueeze(1)  # [N, N]
        both_real = is_real.unsqueeze(0) & is_real.unsqueeze(1)  # [N, N]
        positive_mask = phoneme_match & both_real
        
        # Remove self-connections from positive mask
        positive_mask.fill_diagonal_(False)
        
        # Check if we have any positive pairs
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=projections.device, requires_grad=True), {'no_positives': torch.tensor(1.0)}
        
        # Mask out self-similarity
        mask = torch.eye(batch_size, device=projections.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Add queue negatives if available
        if self.use_queue and self.queue is not None:
            queue_sim = torch.matmul(projections, self.queue.T) / self.temperature  # [N, queue_size]
            sim_matrix = torch.cat([sim_matrix, queue_sim], dim=1)  # [N, N + queue_size]
            
            # Extend positive mask for queue (queue items are not positives for current batch)
            queue_positive = torch.zeros(
                batch_size, self.queue_size, 
                dtype=torch.bool, device=projections.device
            )
            positive_mask = torch.cat([positive_mask, queue_positive], dim=1)
        
        # Compute loss using log-softmax
        # For each anchor, compute loss over its positive pairs
        log_probs = F.log_softmax(sim_matrix, dim=1)
        
        # Average log prob over positive pairs
        positive_log_probs = log_probs * positive_mask.float()
        num_positives = positive_mask.sum(dim=1).clamp(min=1)
        per_sample_loss = positive_log_probs.sum(dim=1) / num_positives
        
        # Filter out samples with no positives to avoid NaN
        valid_samples = positive_mask.sum(dim=1) > 0
        if valid_samples.sum() == 0:
            loss = torch.tensor(0.0, device=projections.device, requires_grad=True)
        else:
            loss = -per_sample_loss[valid_samples].mean()
        
        # Compute avg off-diagonal similarity (diagonal is masked to -inf)
        off_diag_mask = ~torch.eye(batch_size, device=projections.device, dtype=torch.bool)
        off_diag_sim = sim_matrix[:, :batch_size][off_diag_mask]
        avg_sim = off_diag_sim.mean() if off_diag_sim.numel() > 0 else torch.tensor(0.0)
        metrics = {
            'avg_positives_per_anchor': num_positives.float().mean(),
            'avg_similarity': avg_sim
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def _update_queue(
        self,
        projections: torch.Tensor,
        phoneme_labels: torch.Tensor
    ):
        """Update memory queue with new samples.
        
        Args:
            projections: New projected features [N, D]
            phoneme_labels: Corresponding phoneme labels [N]
        """
        batch_size = projections.shape[0]
        ptr = int(self.queue_ptr)
        
        # Handle wrap-around
        if ptr + batch_size > self.queue_size:
            # Split and wrap
            first_part = self.queue_size - ptr
            self.queue[ptr:] = projections[:first_part]
            self.queue_labels[ptr:] = phoneme_labels[:first_part]
            
            remaining = batch_size - first_part
            self.queue[:remaining] = projections[first_part:]
            self.queue_labels[:remaining] = phoneme_labels[first_part:]
            
            self.queue_ptr[0] = remaining
        else:
            self.queue[ptr:ptr + batch_size] = projections
            self.queue_labels[ptr:ptr + batch_size] = phoneme_labels
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size


class PhonemePrototypeBank(nn.Module):
    """Phoneme prototype bank for cross-speaker phoneme anchoring.
    
    Key innovation: Build prototype representations for each phoneme class
    from diverse real speakers. These prototypes serve as anchors in 
    contrastive learning, helping the model learn what "real" phonemes look like.
    """
    
    def __init__(
        self,
        num_phonemes: int = 39,
        prototype_dim: int = 256,
        num_prototypes_per_phoneme: int = 1,
        momentum: float = 0.999
    ):
        """Initialize prototype bank.
        
        Args:
            num_phonemes: Number of phoneme classes
            prototype_dim: Dimension of prototype vectors
            num_prototypes_per_phoneme: Number of prototypes per phoneme
            momentum: EMA momentum for prototype updates
        """
        super().__init__()
        
        self.num_phonemes = num_phonemes
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes_per_phoneme
        self.momentum = momentum
        
        # Initialize prototypes
        self.register_buffer(
            'prototypes',
            torch.randn(num_phonemes, num_prototypes_per_phoneme, prototype_dim)
        )
        self.prototypes = F.normalize(self.prototypes, dim=-1)
        
        # Track update counts for initialization
        self.register_buffer(
            'prototype_counts',
            torch.zeros(num_phonemes, dtype=torch.long)
        )
        
        # Projection layer to match prototype dimension
        self.projection = nn.Linear(prototype_dim, prototype_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        phoneme_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance to phoneme prototypes.
        
        Args:
            features: Input features [N, D]
            phoneme_labels: Phoneme class labels [N]
            
        Returns:
            prototype_distances: Distance to class prototype [N]
        """
        # Project features
        projected = F.normalize(self.projection(features), dim=-1)  # [N, D]
        
        # Get prototypes for each sample's phoneme class
        batch_prototypes = self.prototypes[phoneme_labels, 0, :]  # [N, D]
        
        # Compute cosine similarity (higher = closer)
        similarity = (projected * batch_prototypes).sum(dim=-1)  # [N]
        
        # Convert to distance (lower = closer)
        distance = 1 - similarity
        
        return distance
    
    @torch.no_grad()
    def update_prototypes(
        self,
        features: torch.Tensor,
        phoneme_labels: torch.Tensor,
        is_real: torch.Tensor
    ):
        """Update prototypes with EMA from real samples.
        
        Args:
            features: Input features [N, D]
            phoneme_labels: Phoneme class labels [N]
            is_real: Binary mask for real samples [N]
        """
        # Only use real samples for prototype updates
        real_mask = is_real.bool()
        if not real_mask.any():
            return
        
        real_features = features[real_mask]
        real_labels = phoneme_labels[real_mask]
        
        # Normalize features
        real_features = F.normalize(real_features, dim=-1)
        
        # Update each phoneme's prototype
        for phoneme_id in real_labels.unique():
            mask = real_labels == phoneme_id
            phoneme_features = real_features[mask]
            
            # Compute mean feature for this phoneme
            mean_feature = phoneme_features.mean(dim=0)
            mean_feature = F.normalize(mean_feature, dim=0)
            
            # EMA update
            current = self.prototypes[phoneme_id, 0]
            if self.prototype_counts[phoneme_id] == 0:
                # First update: initialize directly
                self.prototypes[phoneme_id, 0] = mean_feature
            else:
                # EMA update
                self.prototypes[phoneme_id, 0] = (
                    self.momentum * current + 
                    (1 - self.momentum) * mean_feature
                )
                self.prototypes[phoneme_id, 0] = F.normalize(
                    self.prototypes[phoneme_id, 0], dim=0
                )
            
            self.prototype_counts[phoneme_id] += mask.sum()
    
    def get_prototype_loss(
        self,
        features: torch.Tensor,
        phoneme_labels: torch.Tensor,
        is_real: torch.Tensor
    ) -> torch.Tensor:
        """Compute prototype alignment loss.
        
        Real samples should be close to their phoneme prototype.
        Fake samples should be far from all prototypes.
        
        Args:
            features: Input features [N, D]
            phoneme_labels: Phoneme class labels [N]
            is_real: Binary mask for real samples [N]
            
        Returns:
            loss: Prototype alignment loss
        """
        projected = F.normalize(self.projection(features), dim=-1)
        
        # Get distances for real samples (should be small)
        real_mask = is_real.bool()
        if real_mask.any():
            real_features = projected[real_mask]
            real_labels = phoneme_labels[real_mask]
            real_prototypes = self.prototypes[real_labels, 0]
            
            # Cosine distance loss for real samples
            real_similarity = (real_features * real_prototypes).sum(dim=-1)
            real_loss = (1 - real_similarity).mean()
        else:
            real_loss = torch.tensor(0.0, device=features.device)
        
        # Get distances for fake samples (should be large, i.e., low similarity to any prototype)
        fake_mask = ~is_real.bool()
        if fake_mask.any():
            fake_features = projected[fake_mask]  # [M, D]
            all_prototypes = self.prototypes[:, 0, :]  # [num_phonemes, D]
            
            # Similarity to all prototypes
            fake_to_all = torch.matmul(fake_features, all_prototypes.T)  # [M, num_phonemes]
            
            # Max similarity (fake should have low max similarity)
            max_similarity = fake_to_all.max(dim=-1)[0]
            fake_loss = max_similarity.mean()  # Penalize high similarity
        else:
            fake_loss = torch.tensor(0.0, device=features.device)
        
        # Combined loss: minimize real distance, maximize fake distance
        loss = real_loss + 0.5 * fake_loss
        
        return loss
