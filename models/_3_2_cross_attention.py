from __future__ import annotations

import math

import torch
import torch.nn as nn

from models._1_3_multi_head_output_projection import MultiHeadOutputProjection


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention giữa caption và ảnh.

    Query:
        Lấy từ Text Decoder [B, text_length, d_model]

    Key và Value:
        Lấy từ Vision Transformer Encoder. [B, num_patches, d_model]

    Output:
        cross_attention_residual: [B, text_length, d_model]
        projected_attention_output: [B, text_length, d_model]
        attention_weights: [B, num_heads, text_length, num_patches]
    """

    def __init__(self, d_model: int = 512, num_heads: int = 4, dropout: float = 0.1, layer_norm_eps: float = 1e-6):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} phải chia hết cho num_heads={num_heads}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Pre-LayerNorm dành cho text input.
        self.text_input_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

        # Query lấy từ text.
        self.query_projection = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

        # Key và Value lấy từ visual features.
        self.key_projection = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

        self.value_projection = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

        self.attention_dropout = nn.Dropout(dropout)

        # Ghép các head và nhân với W_O.
        self.output_projection = MultiHeadOutputProjection(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)

        if self.query_projection.bias is not None:
            nn.init.zeros_(self.query_projection.bias)

        if self.key_projection.bias is not None:
            nn.init.zeros_(self.key_projection.bias)

        if self.value_projection.bias is not None:
            nn.init.zeros_(self.value_projection.bias)

    def split_into_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        [B, sequence_length, d_model]
                    ↓
        [B, num_heads, sequence_length, head_dim]
        """

        batch_size, sequence_length, _ = tensor.shape

        tensor = tensor.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)

        tensor = tensor.permute(0, 2, 1, 3)

        return tensor.contiguous()

    def forward(self, text_states: torch.Tensor, visual_features: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --------------------------------------------------
        # Bước 1: Pre-LayerNorm cho text.
        # --------------------------------------------------
        normalized_text = self.text_input_norm(text_states)

        # --------------------------------------------------
        # Bước 2: Tạo Query từ text.
        # [B,L,512] -> [B,L,512] -> [B,4,L,128]
        # --------------------------------------------------
        query = self.query_projection(normalized_text)
        query = self.split_into_heads(query)

        # --------------------------------------------------
        # Bước 3: Tạo Key và Value từ ảnh.
        # [B,196,512] -> [B,4,196,128]
        # --------------------------------------------------
        key = self.key_projection(visual_features)
        value = self.value_projection(visual_features)

        key = self.split_into_heads(key)
        value = self.split_into_heads(value)

        # --------------------------------------------------
        # Bước 4: Tính QKᵀ / sqrt(head_dim).
        #
        # [B,4,L,128] @ [B,4,128,196]
        #               ↓
        #          [B,4,L,196]
        # --------------------------------------------------
        attention_scores = torch.matmul(query, key.transpose(-2, -1))

        attention_scores = (attention_scores * self.scale)

        # --------------------------------------------------
        # Bước 5: Softmax trên 196 image patch.
        # --------------------------------------------------
        attention_weights = torch.softmax(attention_scores, dim=-1)

        dropped_attention_weights = (self.attention_dropout(attention_weights))

        # --------------------------------------------------
        # Bước 6: Attention weights × Value.
        #
        # [B,4,L,196] @ [B,4,196,128]
        #               ↓
        #          [B,4,L,128]
        # --------------------------------------------------
        attention_output = torch.matmul(dropped_attention_weights, value)

        # --------------------------------------------------
        # Bước 7: Ghép head và W_O.
        #
        # [B,4,L,128] -> [B,L,512]
        # --------------------------------------------------
        (
            _,
            projected_attention_output
        ) = self.output_projection(attention_output)

        # --------------------------------------------------
        # Bước 8: Residual Connection.
        # --------------------------------------------------
        cross_attention_residual = (text_states + projected_attention_output)

        return (
            cross_attention_residual,
            projected_attention_output,
            attention_weights
        )
