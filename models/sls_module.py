# ABOUTME: Sensitive Layer Selection (SLS) module for multi-layer SSL feature aggregation
# ABOUTME: Learns per-layer weights to combine features from all transformer layers

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class SLSModule(nn.Module):
    """Sensitive Layer Selection module for aggregating multi-layer SSL features.
    
    Based on: "Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier"
    
    Key insight: Different layers of SSL models capture different information:
    - Lower layers: acoustic/spectral properties
    - Middle layers: phonetic content
    - Higher layers: linguistic/contextual information
    
    SLS learns to weight layers adaptively based on the input.
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        use_sigmoid: bool = True,
        dropout: float = 0.1
    ):
        """Initialize SLS module.
        
        Args:
            num_layers: Number of transformer layers in SSL model
            hidden_dim: Hidden dimension of SSL features
            use_sigmoid: If True, use sigmoid for weights (allows multiple high weights)
                        If False, use softmax (forces single layer selection)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_sigmoid = use_sigmoid
        
        # Layer-wise weight predictor
        # Input: concatenated layer features after pooling
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_layers)
        )
        
        # Layer normalization for each layer's features
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Learnable base weights (prior)
        self.base_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate multi-layer features using learned weights.
        
        Args:
            hidden_states: Tuple of layer hidden states, each [B, T, D]
            attention_mask: Optional attention mask [B, T]
            
        Returns:
            Tuple of:
                - aggregated_features: Weighted sum of layers [B, T, D]
                - layer_weights: Learned weights per layer [B, num_layers]
        """
        # Stack hidden states: [num_layers, B, T, D]
        stacked = torch.stack(hidden_states, dim=0)
        num_layers, batch_size, seq_len, hidden_dim = stacked.shape
        
        # Apply layer normalization to each layer
        normalized = []
        for l in range(num_layers):
            normalized.append(self.layer_norms[l](stacked[l]))
        stacked = torch.stack(normalized, dim=0)  # [num_layers, B, T, D]
        
        # Compute temporal average for weight prediction
        if attention_mask is not None:
            # Mask padding tokens
            mask = attention_mask.unsqueeze(0).unsqueeze(-1)  # [1, B, T, 1]
            masked_stacked = stacked * mask
            pooled = masked_stacked.sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # [num_layers, B, D]
        else:
            pooled = stacked.mean(dim=2)  # [num_layers, B, D]
        
        # Use mean across layers for weight prediction
        pooled_mean = pooled.mean(dim=0)  # [B, D]
        
        # Predict layer weights
        weight_logits = self.weight_predictor(pooled_mean)  # [B, num_layers]
        
        # Add base weights
        weight_logits = weight_logits + self.base_weights.unsqueeze(0)
        
        # Apply activation
        if self.use_sigmoid:
            layer_weights = torch.sigmoid(weight_logits)  # [B, num_layers]
        else:
            layer_weights = torch.softmax(weight_logits, dim=-1)  # [B, num_layers]
        
        # Weighted aggregation
        # layer_weights: [B, num_layers] -> [num_layers, B, 1, 1]
        weights_expanded = layer_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum: [num_layers, B, T, D] * [num_layers, B, 1, 1] -> [B, T, D]
        aggregated = (stacked * weights_expanded).sum(dim=0)
        
        # Normalize by sum of weights if using sigmoid
        if self.use_sigmoid:
            weight_sum = layer_weights.sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            aggregated = aggregated / weight_sum.clamp(min=1e-6)
        
        return aggregated, layer_weights


class PhonemeAwareSLS(nn.Module):
    """Phoneme-aware SLS that learns different layer weights for different phoneme categories.
    
    Key innovation: Different phonemes may benefit from different layer combinations.
    - Sibilants (/s/, /z/) may need more high-frequency (lower layer) attention
    - Vowels may need more formant (middle layer) attention
    - This module learns phoneme-specific layer weights
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_phoneme_categories: int = 6,
        use_sigmoid: bool = True,
        dropout: float = 0.1
    ):
        """Initialize phoneme-aware SLS.
        
        Args:
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension of features
            num_phoneme_categories: Number of phoneme category types
            use_sigmoid: Use sigmoid activation for weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_categories = num_phoneme_categories
        self.use_sigmoid = use_sigmoid
        
        # Shared feature processing
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Per-category weight predictors
        self.category_weight_predictors = nn.ModuleList([
            nn.Linear(hidden_dim, num_layers)
            for _ in range(num_phoneme_categories)
        ])
        
        # Global weight predictor (fallback)
        self.global_weight_predictor = nn.Linear(hidden_dim, num_layers)
        
        # Learnable category-specific base weights
        self.category_base_weights = nn.Parameter(
            torch.ones(num_phoneme_categories, num_layers) / num_layers
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        phoneme_categories: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate features with phoneme-aware layer weighting.
        
        Args:
            hidden_states: Tuple of layer hidden states, each [B, T, D]
            phoneme_categories: Per-frame phoneme category indices [B, T]
            attention_mask: Optional attention mask [B, T]
            
        Returns:
            Tuple of:
                - aggregated_features: [B, T, D]
                - layer_weights: [B, T, num_layers] (per-frame weights)
        """
        # Stack and normalize hidden states
        stacked = torch.stack(hidden_states, dim=0)  # [L, B, T, D]
        num_layers, batch_size, seq_len, hidden_dim = stacked.shape
        
        # Apply layer normalization
        normalized = []
        for l in range(num_layers):
            normalized.append(self.layer_norms[l](stacked[l]))
        stacked = torch.stack(normalized, dim=0)
        
        # Compute per-frame features for weight prediction
        frame_features = stacked.mean(dim=0)  # [B, T, D] (average across layers)
        frame_features = self.shared_projection(frame_features)
        
        if phoneme_categories is None:
            # Use global weights
            weight_logits = self.global_weight_predictor(frame_features)  # [B, T, L]
            layer_weights = torch.sigmoid(weight_logits) if self.use_sigmoid else torch.softmax(weight_logits, dim=-1)
        else:
            # Compute category-specific weights
            all_weight_logits = torch.stack([
                predictor(frame_features) for predictor in self.category_weight_predictors
            ], dim=-1)  # [B, T, L, num_categories]
            
            # Add base weights
            base = self.category_base_weights.permute(1, 0)  # [L, num_categories]
            all_weight_logits = all_weight_logits + base.unsqueeze(0).unsqueeze(0)
            
            # Select weights based on phoneme category
            # phoneme_categories: [B, T] -> [B, T, 1, 1]
            cat_idx = phoneme_categories.unsqueeze(-1).unsqueeze(-1)
            cat_idx = cat_idx.expand(-1, -1, num_layers, 1)
            
            # Gather category-specific weights
            weight_logits = torch.gather(all_weight_logits, dim=-1, index=cat_idx).squeeze(-1)  # [B, T, L]
            
            if self.use_sigmoid:
                layer_weights = torch.sigmoid(weight_logits)
            else:
                layer_weights = torch.softmax(weight_logits, dim=-1)
        
        # Weighted aggregation per frame
        # layer_weights: [B, T, L] -> [L, B, T, 1]
        weights_expanded = layer_weights.permute(2, 0, 1).unsqueeze(-1)
        
        # Weighted sum
        aggregated = (stacked * weights_expanded).sum(dim=0)  # [B, T, D]
        
        # Normalize
        if self.use_sigmoid:
            weight_sum = layer_weights.sum(dim=-1, keepdim=True)  # [B, T, 1]
            aggregated = aggregated / weight_sum.clamp(min=1e-6)
        
        return aggregated, layer_weights
