# ABOUTME: Phoneme encoder using Wav2Vec 2.0 CTC for phoneme segmentation
# ABOUTME: Extracts phoneme boundaries and per-phoneme features for contrastive learning

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# CMU ARPAbet phoneme set (39 phonemes)
CMU_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]

# Phoneme categories for attention
PHONEME_CATEGORIES = {
    'vowels': ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'],
    'fricatives': ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'],
    'plosives': ['P', 'B', 'T', 'D', 'K', 'G'],
    'nasals': ['M', 'N', 'NG'],
    'approximants': ['L', 'R', 'W', 'Y'],
    'affricates': ['CH', 'JH']
}


class PhonemeEncoder(nn.Module):
    """Phoneme encoder using Wav2Vec 2.0 for CTC-based phoneme segmentation.
    
    This module:
    1. Uses pretrained Wav2Vec 2.0 CTC model to extract phoneme predictions
    2. Segments audio into phoneme-level chunks
    3. Projects phoneme features into a shared embedding space
    """
    
    def __init__(
        self,
        ctc_model_name: str = "facebook/wav2vec2-base-960h",
        num_phonemes: int = 39,
        embedding_dim: int = 256,
        freeze_ctc: bool = True,
        device: str = "cuda"
    ):
        """Initialize phoneme encoder.
        
        Args:
            ctc_model_name: Pretrained Wav2Vec 2.0 CTC model for phoneme recognition
            num_phonemes: Number of phoneme classes
            embedding_dim: Output phoneme embedding dimension
            freeze_ctc: Whether to freeze CTC model weights
            device: Device to load model on
        """
        super().__init__()
        
        self.num_phonemes = num_phonemes
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Load pretrained CTC model for phoneme extraction
        self.processor = Wav2Vec2Processor.from_pretrained(ctc_model_name)
        self.ctc_model = Wav2Vec2ForCTC.from_pretrained(ctc_model_name)
        
        if freeze_ctc:
            for param in self.ctc_model.parameters():
                param.requires_grad = False
        
        # Phoneme embedding layer
        # Maps from hidden dim to phoneme embedding
        hidden_size = self.ctc_model.config.hidden_size
        self.phoneme_projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Phoneme-specific attention weights for different phoneme categories
        self.category_attention = nn.ModuleDict({
            category: nn.Linear(embedding_dim, 1)
            for category in PHONEME_CATEGORIES.keys()
        })
        
        # Character to phoneme mapping (simplified)
        self._build_char_to_phoneme_map()
        
    def _build_char_to_phoneme_map(self):
        """Build mapping from CTC characters to phoneme indices."""
        self.char_to_phoneme = {}
        vocab = self.processor.tokenizer.get_vocab()
        
        # Map vocabulary to phoneme indices
        for char, idx in vocab.items():
            char_upper = char.upper()
            if char_upper in CMU_PHONEMES:
                self.char_to_phoneme[idx] = CMU_PHONEMES.index(char_upper)
    
    def forward(
        self,
        waveforms: torch.Tensor,
        return_segments: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Extract phoneme features from audio.
        
        Args:
            waveforms: Audio waveforms [B, T]
            return_segments: If True, return segment boundaries
            
        Returns:
            Dictionary containing:
                - phoneme_features: Per-frame phoneme features [B, T', D]
                - phoneme_ids: Predicted phoneme IDs [B, T']
                - phoneme_probs: Phoneme probabilities [B, T', num_phonemes]
        """
        batch_size = waveforms.shape[0]
        
        # Process through Wav2Vec 2.0 CTC
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.ctc_model(
                waveforms,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get logits and hidden states
        logits = outputs.logits  # [B, T', vocab_size]
        hidden_states = outputs.hidden_states[-1]  # [B, T', hidden_size]
        
        # Get phoneme predictions via argmax
        phoneme_preds = torch.argmax(logits, dim=-1)  # [B, T']
        phoneme_probs = torch.softmax(logits, dim=-1)  # [B, T', vocab_size]
        
        # Project hidden states to phoneme embeddings
        phoneme_features = self.phoneme_projection(hidden_states)  # [B, T', embedding_dim]
        
        result = {
            'phoneme_features': phoneme_features,
            'phoneme_ids': phoneme_preds,
            'phoneme_probs': phoneme_probs,
            'hidden_states': hidden_states
        }
        
        if return_segments:
            segments = self._extract_segments(phoneme_preds)
            result['segments'] = segments
        
        return result
    
    def _extract_segments(
        self,
        phoneme_preds: torch.Tensor
    ) -> List[List[Dict[str, int]]]:
        """Extract phoneme segment boundaries.
        
        Args:
            phoneme_preds: Predicted phoneme IDs [B, T']
            
        Returns:
            List of segment lists per batch item.
            Each segment is a dict with 'phoneme_id', 'start', 'end'
        """
        batch_segments = []
        
        for b in range(phoneme_preds.shape[0]):
            preds = phoneme_preds[b].cpu().tolist()
            segments = []
            
            current_phoneme = preds[0]
            start_idx = 0
            
            for t, phoneme in enumerate(preds[1:], 1):
                if phoneme != current_phoneme:
                    if current_phoneme != 0:  # Skip blank token
                        segments.append({
                            'phoneme_id': current_phoneme,
                            'start': start_idx,
                            'end': t
                        })
                    current_phoneme = phoneme
                    start_idx = t
            
            # Add last segment
            if current_phoneme != 0:
                segments.append({
                    'phoneme_id': current_phoneme,
                    'start': start_idx,
                    'end': len(preds)
                })
            
            batch_segments.append(segments)
        
        return batch_segments
    
    def get_phoneme_embeddings(
        self,
        phoneme_features: torch.Tensor,
        segments: List[List[Dict[str, int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get per-phoneme embeddings by averaging over segments.
        
        Args:
            phoneme_features: Frame-level features [B, T', D]
            segments: Segment boundaries from _extract_segments
            
        Returns:
            Tuple of:
                - phoneme_embeddings: [total_segments, D]
                - phoneme_labels: [total_segments] phoneme class IDs
        """
        all_embeddings = []
        all_labels = []
        
        for b, seg_list in enumerate(segments):
            for seg in seg_list:
                # Average features over segment
                start, end = seg['start'], seg['end']
                segment_features = phoneme_features[b, start:end, :]
                embedding = segment_features.mean(dim=0)
                
                all_embeddings.append(embedding)
                all_labels.append(seg['phoneme_id'])
        
        if len(all_embeddings) == 0:
            # Return empty tensors if no segments
            return (
                torch.zeros(0, phoneme_features.shape[-1], device=phoneme_features.device),
                torch.zeros(0, dtype=torch.long, device=phoneme_features.device)
            )
        
        phoneme_embeddings = torch.stack(all_embeddings, dim=0)
        phoneme_labels = torch.tensor(all_labels, dtype=torch.long, device=phoneme_features.device)
        
        return phoneme_embeddings, phoneme_labels
    
    def get_category_weights(self, phoneme_id: int) -> str:
        """Get the category for a phoneme ID.
        
        Args:
            phoneme_id: Phoneme index
            
        Returns:
            Category name
        """
        if phoneme_id >= len(CMU_PHONEMES):
            return 'vowels'  # Default
        
        phoneme = CMU_PHONEMES[phoneme_id]
        
        for category, phonemes in PHONEME_CATEGORIES.items():
            if phoneme in phonemes:
                return category
        
        return 'vowels'  # Default fallback
