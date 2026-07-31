from __future__ import annotations

import math

import torch
import torch.nn as nn

from models._1_3_multi_head_output_projection import MultiHeadOutputProjection
from models._1_1_multihead_qkv_projection import MultiHeadQKVProjection


class MaskedMultiHeadSelfAttention(nn.Module):
    """
    Masked Multi-Head Self-Attention dành cho Text Decoder.

    Causal mask bảo đảm token ở vị trí hiện tại không nhìn thấy các token đứng phía sau.

    Padding mask bảo đảm mô hình không chú ý đến PAD token.

    Input:
        text_embeddings: [B, sequence_length, d_model]

        padding_mask: [B, sequence_length]
            True  = PAD token
            False = token hợp lệ

    Output:
        self_attention_residual: [B, sequence_length, d_model]
        projected_attention_output: [B, sequence_length, d_model]
        attention_weights: [B, num_heads, sequence_length, sequence_length]
    """

    def __init__(self, d_model: int = 512, num_heads: int = 4, dropout: float = 0.1, layer_norm_eps: float = 1e-6):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} phải chia hết cho num_heads={num_heads}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Pre-LayerNorm.
        self.input_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

        # W_Q, W_K và W_V.
        self.qkv_projection = MultiHeadQKVProjection(d_model=d_model, num_heads=num_heads)

        self.attention_dropout = nn.Dropout(dropout)

        # Ghép head và W_O.
        self.output_projection = (MultiHeadOutputProjection(d_model=d_model, num_heads=num_heads, dropout=dropout))

    @staticmethod
    def create_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
        """
        Tạo causal mask.

        Ví dụ sequence_length=4:

        False True  True  True
        False False True  True
        False False False True
        False False False False

        True nghĩa là vị trí bị che.
        """

        return torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
            diagonal=1
        )

    def forward(self, text_embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, sequence_length, _ = text_embeddings.shape

        # --------------------------------------------------
        # Bước 1: Pre-LayerNorm.
        # --------------------------------------------------
        normalized_text = self.input_norm(text_embeddings)

        # --------------------------------------------------
        # Bước 2: Tạo Q, K và V.
        #
        # [B,L,512] -> [B,4,L,128]
        # --------------------------------------------------
        query, key, value = self.qkv_projection(normalized_text)

        # --------------------------------------------------
        # Bước 3: Tính QKᵀ / sqrt(head_dim).
        #
        # [B,4,L,128] @ [B,4,128,L]
        #              ↓
        #          [B,4,L,L]
        # --------------------------------------------------
        attention_scores = torch.matmul(query, key.transpose(-2, -1))

        attention_scores = (attention_scores * self.scale)

        # --------------------------------------------------
        # Bước 4: Tạo causal mask.
        #
        # [L,L] -> [1,1,L,L]
        # --------------------------------------------------
        causal_mask = self.create_causal_mask(sequence_length=sequence_length, device=text_embeddings.device)

        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # --------------------------------------------------
        # Bước 5: Kết hợp padding mask.
        #
        # [B,L] -> [B,1,1,L]
        # --------------------------------------------------
        if padding_mask is not None:
            key_padding_mask = padding_mask[
                :,
                None,
                None,
                :
            ]

            combined_mask = (combined_mask | key_padding_mask)

        attention_scores = attention_scores.masked_fill(
            combined_mask,
            torch.finfo(attention_scores.dtype).min
        )

        # --------------------------------------------------
        # Bước 6: Softmax.
        #
        # [B,4,L,L]
        # --------------------------------------------------
        attention_weights = torch.softmax(attention_scores, dim=-1)

        dropped_attention_weights = (self.attention_dropout(attention_weights))

        # --------------------------------------------------
        # Bước 7: Attention weights × V.
        #
        # [B,4,L,L] @ [B,4,L,128]
        #              ↓
        #         [B,4,L,128]
        # --------------------------------------------------
        attention_output = torch.matmul(dropped_attention_weights, value)

        # --------------------------------------------------
        # Bước 8: Ghép head và W_O.
        #
        # [B,4,L,128] -> [B,L,512]
        # --------------------------------------------------
        (
            _,
            projected_attention_output
        ) = self.output_projection(attention_output)

        # --------------------------------------------------
        # Bước 9: Residual Connection.
        # --------------------------------------------------
        self_attention_residual = (text_embeddings + projected_attention_output)

        return (
            self_attention_residual,
            projected_attention_output,
            attention_weights
        )
