# ABOUTME: Loss functions for PAMCL training
# ABOUTME: Combines classification, contrastive, and phoneme consistency losses

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PAMCLLoss(nn.Module):
    """Combined loss function for PAMCL training.
    
    L_total = λ_CE * L_CE + λ_contrastive * L_contrastive + λ_phoneme * L_phoneme_consistency
    
    Where:
    - L_CE: Cross-entropy for real/fake classification
    - L_contrastive: InfoNCE loss for phoneme-level contrastive learning
    - L_phoneme_consistency: Forces same phoneme to have similar features
    """
    
    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_contrastive: float = 0.5,
        lambda_phoneme_consistency: float = 0.3,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        num_real: int = 1,
        num_fake: int = 1
    ):
        """Initialize loss function.
        
        Args:
            lambda_ce: Weight for classification loss
            lambda_contrastive: Weight for contrastive loss
            lambda_phoneme_consistency: Weight for phoneme consistency loss
            label_smoothing: Label smoothing factor for CE loss
            class_weights: Optional class weights for imbalanced data
            num_real: Number of real samples (for auto class weighting)
            num_fake: Number of fake samples (for auto class weighting)
        """
        super().__init__()
        
        self.lambda_ce = lambda_ce
        self.lambda_contrastive = lambda_contrastive
        self.lambda_phoneme = lambda_phoneme_consistency
        
        # Compute class weights if not provided
        # Use inverse frequency weighting to handle imbalance
        if class_weights is None and num_real > 0 and num_fake > 0:
            total = num_real + num_fake
            # Weight = total / (2 * class_count) - gives equal importance to both classes
            weight_real = total / (2.0 * num_real)
            weight_fake = total / (2.0 * num_fake)
            class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32)
            print(f"Class weights computed: real={weight_real:.2f}, fake={weight_fake:.2f}")
        
        self.register_buffer('class_weights', class_weights)
        self.label_smoothing = label_smoothing
        
        # Will be set in forward based on device
        self._ce_loss_fn = None
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            model_outputs: Dictionary from model forward pass containing:
                - logits: Classification logits [B, 2]
                - contrastive_loss: Contrastive loss (optional)
                - prototype_loss: Prototype alignment loss (optional)
            labels: Ground truth labels [B]
            
        Returns:
            Dictionary containing:
                - total_loss: Combined loss for backprop
                - ce_loss: Classification loss
                - contrastive_loss: Contrastive loss (if present)
                - prototype_loss: Prototype loss (if present)
        """
        losses = {}
        
        # Classification loss with class weights
        logits = model_outputs['logits']
        
        # Apply class-weighted cross entropy
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            ce_loss = F.cross_entropy(
                logits, labels, 
                weight=weights,
                label_smoothing=self.label_smoothing
            )
        else:
            ce_loss = F.cross_entropy(
                logits, labels,
                label_smoothing=self.label_smoothing
            )
        
        losses['ce_loss'] = ce_loss
        
        total_loss = self.lambda_ce * ce_loss
        
        # Contrastive loss (if available and not NaN)
        # Handle DataParallel: loss might be a tensor with multiple values
        if 'contrastive_loss' in model_outputs:
            contrastive_loss = model_outputs['contrastive_loss']
            # Handle multi-GPU case: take mean if multiple values
            if contrastive_loss.dim() > 0:
                contrastive_loss = contrastive_loss.mean()
            if torch.isfinite(contrastive_loss).all():
                losses['contrastive_loss'] = contrastive_loss
                total_loss = total_loss + self.lambda_contrastive * contrastive_loss
        
        # Prototype/phoneme consistency loss (if available and not NaN)
        if 'prototype_loss' in model_outputs:
            prototype_loss = model_outputs['prototype_loss']
            # Handle multi-GPU case: take mean if multiple values
            if prototype_loss.dim() > 0:
                prototype_loss = prototype_loss.mean()
            if torch.isfinite(prototype_loss).all():
                losses['prototype_loss'] = prototype_loss
                total_loss = total_loss + self.lambda_phoneme * prototype_loss
        
        # Ensure total loss is finite
        if not torch.isfinite(total_loss).all():
            # Fall back to just CE loss if total is NaN
            total_loss = self.lambda_ce * ce_loss
        
        losses['total_loss'] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Focuses more on hard, misclassified examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits [B, C]
            targets: Ground truth labels [B]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        targets_f = targets.float()
        alpha_t = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PhonemeConsistencyLoss(nn.Module):
    """Phoneme consistency loss.
    
    Encourages features of the same phoneme (from real audio) to be similar
    across different speakers and utterances.
    """
    
    def __init__(self, margin: float = 0.5):
        """Initialize consistency loss.
        
        Args:
            margin: Margin for triplet-style loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        features: torch.Tensor,
        phoneme_labels: torch.Tensor,
        is_real: torch.Tensor
    ) -> torch.Tensor:
        """Compute phoneme consistency loss.
        
        Args:
            features: Phoneme features [N, D]
            phoneme_labels: Phoneme class labels [N]
            is_real: Binary mask for real samples [N]
            
        Returns:
            Consistency loss value
        """
        # Only consider real samples for consistency
        real_mask = is_real.bool()
        if real_mask.sum() < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        real_features = F.normalize(features[real_mask], dim=-1)
        real_labels = phoneme_labels[real_mask]
        
        # Compute pairwise distances
        distances = 1 - torch.matmul(real_features, real_features.T)  # [N, N]
        
        # Create same-phoneme mask
        same_phoneme = real_labels.unsqueeze(0) == real_labels.unsqueeze(1)
        same_phoneme.fill_diagonal_(False)  # Exclude self-pairs
        
        # Create different-phoneme mask
        diff_phoneme = real_labels.unsqueeze(0) != real_labels.unsqueeze(1)
        
        if same_phoneme.sum() == 0 or diff_phoneme.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # Average distance for same phoneme (should be small)
        same_dist = (distances * same_phoneme.float()).sum() / same_phoneme.sum()
        
        # Average distance for different phoneme (should be large)
        diff_dist = (distances * diff_phoneme.float()).sum() / diff_phoneme.sum()
        
        # Triplet-style loss with margin
        loss = F.relu(same_dist - diff_dist + self.margin)
        
        return loss


class LayerRegularizationLoss(nn.Module):
    """Regularization for SLS layer weights.
    
    Encourages the model to use multiple layers rather than collapsing
    to a single layer selection.
    """
    
    def __init__(self, entropy_weight: float = 0.01):
        """Initialize regularization loss.
        
        Args:
            entropy_weight: Weight for entropy regularization
        """
        super().__init__()
        self.entropy_weight = entropy_weight
    
    def forward(self, layer_weights: torch.Tensor) -> torch.Tensor:
        """Compute layer regularization loss.
        
        Args:
            layer_weights: Layer weights from SLS [B, num_layers] or [B, T, num_layers]
            
        Returns:
            Regularization loss
        """
        # Normalize weights to form a distribution
        if layer_weights.dim() == 3:
            # [B, T, L] -> [B*T, L]
            weights = layer_weights.reshape(-1, layer_weights.shape[-1])
        else:
            weights = layer_weights
        
        # Normalize to probability distribution
        weights_normalized = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        
        # Compute entropy (higher = more spread)
        entropy = -(weights_normalized * torch.log(weights_normalized + 1e-6)).sum(dim=-1)
        
        # We want high entropy, so minimize negative entropy
        loss = -entropy.mean()
        
        return self.entropy_weight * loss
