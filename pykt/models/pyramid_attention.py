import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidAttention(nn.Module):
    """Multi-head attention with simple pyramid down-sampling.

    The module computes attention outputs at multiple temporal resolutions and
    blends them with learnable weights. Each resolution reuses a standard
    ``nn.MultiheadAttention`` layer while keys/values are progressively
    down-sampled through average pooling. Outputs from all resolutions are
    combined through a learnable weighted sum followed by a projection layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        num_levels: int = 2,
    ) -> None:
        super().__init__()
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) for _ in range(num_levels)]
        )
        self.level_weights = nn.Parameter(torch.ones(num_levels))
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = []
        weights_list = []
        level_importance = torch.softmax(self.level_weights, dim=0)
        for level, attn in enumerate(self.attn_layers):
            key_level = self._downsample_sequence(key, level)
            value_level = self._downsample_sequence(value, level)
            attn_mask_level = self._downsample_attn_mask(attn_mask, level)
            padding_mask_level = self._downsample_padding_mask(key_padding_mask, level)
            out, attn_weights = attn(
                query,
                key_level,
                value_level,
                attn_mask=attn_mask_level,
                key_padding_mask=padding_mask_level,
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
            )
            outputs.append(out)
            if need_weights:
                weights_list.append(attn_weights)
        combined = sum(w * o for w, o in zip(level_importance, outputs))
        combined = self.output_proj(combined)
        combined = self.dropout(combined)
        if need_weights:
            stacked = torch.stack(weights_list)
            level_importance = level_importance.view(-1, *([1] * (stacked.dim() - 1)))
            attn_weights = (stacked * level_importance).sum(dim=0)
        else:
            attn_weights = None
        return combined, attn_weights

    def _downsample_sequence(self, tensor: torch.Tensor, level: int) -> torch.Tensor:
        if level == 0:
            return tensor
        scale = 2 ** level
        tensor_batched = tensor.permute(1, 2, 0)
        pooled = F.avg_pool1d(tensor_batched, kernel_size=scale, stride=scale, ceil_mode=True)
        return pooled.permute(2, 0, 1)

    def _downsample_attn_mask(self, mask: Optional[torch.Tensor], level: int) -> Optional[torch.Tensor]:
        if mask is None or level == 0:
            return mask
        scale = 2 ** level
        seq_q, seq_k = mask.shape
        new_seq_k = math.ceil(seq_k / scale)
        pad_cols = new_seq_k * scale - seq_k
        if pad_cols > 0:
            pad_values = torch.ones(seq_q, pad_cols, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, pad_values], dim=1)
        mask = mask.view(seq_q, new_seq_k, scale)
        mask = mask.any(dim=-1)
        return mask

    def _downsample_padding_mask(
        self, mask: Optional[torch.Tensor], level: int
    ) -> Optional[torch.Tensor]:
        if mask is None or level == 0:
            return mask
        scale = 2 ** level
        pooled = F.max_pool1d(mask.unsqueeze(1).float(), kernel_size=scale, stride=scale, ceil_mode=True)
        return pooled.squeeze(1) > 0.5
