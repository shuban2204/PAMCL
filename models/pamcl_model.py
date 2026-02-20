# ABOUTME: Main PAMCL model combining all components for audio deepfake detection
# ABOUTME: Phoneme-Aware Multi-Layer Contrastive Learning architecture

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import Wav2Vec2Model, WavLMModel

from models.phoneme_encoder import PhonemeEncoder, PHONEME_CATEGORIES
from models.sls_module import SLSModule, PhonemeAwareSLS
from models.contrastive_module import ContrastiveModule, PhonemePrototypeBank


class PAMCLModel(nn.Module):
    """Phoneme-Aware Multi-Layer Contrastive Learning Model.
    
    Architecture:
    1. SSL backbone (XLS-R or WavLM) for feature extraction
    2. Phoneme encoder for phoneme segmentation
    3. Phoneme-aware SLS for multi-layer feature aggregation
    4. Contrastive module for representation learning
    5. Classification head for real/fake prediction
    
    Key innovations:
    - Phoneme-level granular analysis
    - Per-phoneme layer weighting
    - Cross-speaker phoneme prototypes
    - Multi-task learning (classification + contrastive)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PAMCL model.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        super().__init__()
        
        self.config = config
        model_cfg = config['model']
        
        # Load SSL backbone
        ssl_backbone_name = model_cfg.get('ssl_backbone', 'facebook/wav2vec2-xls-r-300m')
        self.ssl_backbone = self._load_ssl_backbone(ssl_backbone_name)
        
        # Get SSL model dimensions
        self.hidden_size = self.ssl_backbone.config.hidden_size
        self.num_layers = self.ssl_backbone.config.num_hidden_layers + 1  # +1 for embedding layer
        
        # Freeze backbone if specified
        if model_cfg.get('freeze_ssl', True):
            for param in self.ssl_backbone.parameters():
                param.requires_grad = False
        
        # Enable gradient checkpointing for memory efficiency
        if model_cfg.get('gradient_checkpointing', True):
            self.ssl_backbone.gradient_checkpointing_enable()
        
        # Phoneme encoder
        phoneme_cfg = model_cfg.get('phoneme_encoder', {})
        self.phoneme_encoder = PhonemeEncoder(
            ctc_model_name=phoneme_cfg.get('ctc_model', 'facebook/wav2vec2-base-960h'),
            num_phonemes=phoneme_cfg.get('num_phonemes', 39),
            embedding_dim=phoneme_cfg.get('embedding_dim', 256),
            freeze_ctc=True
        )
        
        # SLS module (phoneme-aware version)
        sls_cfg = model_cfg.get('sls', {})
        self.sls_module = PhonemeAwareSLS(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_size,
            num_phoneme_categories=len(PHONEME_CATEGORIES),
            use_sigmoid=sls_cfg.get('use_sigmoid', True),
            dropout=0.1
        )
        
        # Standard SLS for global features
        self.global_sls = SLSModule(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_size,
            use_sigmoid=sls_cfg.get('use_sigmoid', True)
        )
        
        # Contrastive module
        contrastive_cfg = model_cfg.get('contrastive', {})
        projection_dim = contrastive_cfg.get('projection_dim', 256)
        self.contrastive_module = ContrastiveModule(
            input_dim=self.hidden_size,
            projection_dim=projection_dim,
            temperature=contrastive_cfg.get('temperature', 0.07),
            use_queue=True,
            queue_size=4096
        )
        
        # Phoneme prototype bank
        if contrastive_cfg.get('use_phoneme_prototypes', True):
            self.prototype_bank = PhonemePrototypeBank(
                num_phonemes=phoneme_cfg.get('num_phonemes', 39),
                prototype_dim=self.hidden_size,
                momentum=0.999
            )
        else:
            self.prototype_bank = None
        
        # Classification head
        classifier_cfg = model_cfg.get('classifier', {})
        classifier_hidden = classifier_cfg.get('hidden_dim', 512)
        classifier_dropout = classifier_cfg.get('dropout', 0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden // 2, 2)  # Binary classification
        )
        
        # Phoneme-aware feature fusion
        self.phoneme_fusion = nn.Sequential(
            nn.Linear(self.hidden_size + phoneme_cfg.get('embedding_dim', 256), self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Pre-build phoneme-to-category mapping (avoid rebuilding every forward pass)
        self._build_category_map()
        
    def _load_ssl_backbone(self, model_name: str) -> nn.Module:
        """Load SSL backbone model.
        
        Args:
            model_name: HuggingFace model name/path
            
        Returns:
            Loaded SSL model
        """
        if 'wavlm' in model_name.lower():
            model = WavLMModel.from_pretrained(model_name)
        else:
            # Default to Wav2Vec2 (includes XLS-R)
            model = Wav2Vec2Model.from_pretrained(model_name)
        
        return model
    
    def forward(
        self,
        audio: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            audio: Raw audio waveforms [B, T]
            labels: Optional labels for contrastive learning [B]
            return_features: If True, return intermediate features
            
        Returns:
            Dictionary containing:
                - logits: Classification logits [B, 2]
                - features: Aggregated features [B, D] (if return_features)
                - contrastive_loss: Contrastive loss (if labels provided)
                - prototype_loss: Prototype alignment loss (if enabled)
        """
        batch_size = audio.shape[0]
        device = audio.device
        
        # Extract SSL features (all layers)
        ssl_outputs = self.ssl_backbone(
            audio,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get all hidden states (including embedding layer)
        hidden_states = ssl_outputs.hidden_states  # Tuple of [B, T', D]
        
        # Extract phoneme information
        phoneme_outputs = self.phoneme_encoder(audio, return_segments=True)
        phoneme_ids = phoneme_outputs['phoneme_ids']  # [B, T_phoneme]
        phoneme_features = phoneme_outputs['phoneme_features']  # [B, T_phoneme, D_phoneme]
        
        # Map phoneme IDs to categories
        phoneme_categories = self._phoneme_to_category(phoneme_ids)
        
        # Align phoneme sequence length with SSL sequence length
        ssl_seq_len = hidden_states[0].shape[1]
        phoneme_seq_len = phoneme_categories.shape[1]
        
        if phoneme_seq_len != ssl_seq_len:
            # Interpolate phoneme categories to match SSL length
            phoneme_categories = self._align_sequences(
                phoneme_categories, ssl_seq_len
            )
        
        # Phoneme-aware SLS aggregation
        aggregated_features, layer_weights = self.sls_module(
            hidden_states,
            phoneme_categories=phoneme_categories
        )  # [B, T', D]
        
        # Global pooling
        pooled_features = aggregated_features.mean(dim=1)  # [B, D]
        
        # Classification
        logits = self.classifier(pooled_features)  # [B, 2]
        
        result = {'logits': logits}
        
        if return_features:
            result['features'] = pooled_features
            result['layer_weights'] = layer_weights
        
        # Contrastive learning (if training with labels)
        if labels is not None and self.training:
            # Get phoneme-level features for contrastive learning
            segments = phoneme_outputs.get('segments', None)
            
            if segments is not None:
                phoneme_embeddings, phoneme_labels = self.phoneme_encoder.get_phoneme_embeddings(
                    aggregated_features, segments
                )
                
                if phoneme_embeddings.shape[0] > 0:
                    # Create is_real mask (expand labels to match phoneme count)
                    is_real = []
                    for b, seg_list in enumerate(segments):
                        is_real.extend([labels[b] == 0] * len(seg_list))
                    is_real = torch.tensor(is_real, device=device)
                    
                    # Contrastive loss
                    contrastive_loss, contrastive_metrics = self.contrastive_module(
                        phoneme_embeddings, phoneme_labels, is_real
                    )
                    result['contrastive_loss'] = contrastive_loss
                    result['contrastive_metrics'] = contrastive_metrics
                    
                    # Prototype loss (if enabled)
                    if self.prototype_bank is not None:
                        prototype_loss = self.prototype_bank.get_prototype_loss(
                            phoneme_embeddings, phoneme_labels, is_real
                        )
                        result['prototype_loss'] = prototype_loss
                        
                        # Update prototypes with real samples
                        self.prototype_bank.update_prototypes(
                            phoneme_embeddings, phoneme_labels, is_real
                        )
        
        return result
    
    def _build_category_map(self):
        """Pre-build phoneme ID -> category index mapping (called once in __init__)."""
        # Use CTC vocab size to cover all possible predicted IDs
        vocab_size = self.phoneme_encoder.ctc_model.config.vocab_size
        map_size = max(vocab_size, 100)
        category_to_idx = {cat: i for i, cat in enumerate(PHONEME_CATEGORIES.keys())}
        category_map = torch.zeros(map_size, dtype=torch.long)
        for phoneme_id in range(map_size):
            category_name = self.phoneme_encoder.get_category_weights(phoneme_id)
            category_map[phoneme_id] = category_to_idx.get(category_name, 0)
        self.register_buffer('_category_map', category_map)

    def _phoneme_to_category(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """Map phoneme IDs to category indices using pre-built mapping."""
        phoneme_ids_clamped = phoneme_ids.clamp(0, len(self._category_map) - 1)
        return self._category_map[phoneme_ids_clamped]
    
    def _align_sequences(
        self,
        source: torch.Tensor,
        target_len: int
    ) -> torch.Tensor:
        """Align source sequence to target length via interpolation.
        
        Args:
            source: Source tensor [B, T_source]
            target_len: Target sequence length
            
        Returns:
            Aligned tensor [B, target_len]
        """
        source_len = source.shape[1]
        
        if source_len == target_len:
            return source
        
        # Use nearest neighbor interpolation for discrete values
        source_float = source.float().unsqueeze(1)  # [B, 1, T]
        aligned = nn.functional.interpolate(
            source_float, size=target_len, mode='nearest'
        )
        
        return aligned.squeeze(1).long()
    
    def get_trainable_params(self) -> Dict[str, list]:
        """Get trainable parameters grouped by learning rate.
        
        Returns:
            Dictionary with 'ssl' and 'new' parameter groups
        """
        ssl_params = []
        new_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'ssl_backbone' in name:
                ssl_params.append(param)
            else:
                new_params.append(param)
        
        return {
            'ssl': ssl_params,
            'new': new_params
        }


class PAMCLInference(nn.Module):
    """Lightweight inference wrapper for PAMCL.
    
    Removes training-only components for efficient inference.
    """
    
    def __init__(self, model: PAMCLModel):
        """Initialize inference model.
        
        Args:
            model: Trained PAMCL model
        """
        super().__init__()
        
        self.ssl_backbone = model.ssl_backbone
        self.global_sls = model.global_sls
        self.classifier = model.classifier
        
        # Copy config for dimensions
        self.hidden_size = model.hidden_size
        self.num_layers = model.num_layers
    
    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Inference forward pass.
        
        Args:
            audio: Raw audio waveforms [B, T]
            
        Returns:
            Probability of fake [B]
        """
        # Extract SSL features
        ssl_outputs = self.ssl_backbone(
            audio,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = ssl_outputs.hidden_states
        
        # Global SLS aggregation
        aggregated, _ = self.global_sls(hidden_states)
        
        # Pool and classify
        pooled = aggregated.mean(dim=1)
        logits = self.classifier(pooled)
        
        # Return probability of fake class
        probs = torch.softmax(logits, dim=-1)
        
        return probs[:, 1]  # Fake probability
