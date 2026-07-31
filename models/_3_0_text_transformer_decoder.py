from __future__ import annotations

import torch
import torch.nn as nn

from models._3_2_cross_attention import CrossAttention
from models._3_1_masked_multi_head_self_attention import (
    MaskedMultiHeadSelfAttention
)
from models._3_3_text_decoder_feed_forward import (
    TextDecoderFeedForward
)


class TextDecoderBlock(nn.Module):
    """
    Một Transformer Text Decoder Block bao gồm:

    1. Masked Multi-Head Self-Attention.
    2. Cross-Attention với visual features.
    3. Feed-Forward Network.

    Input:
        text_states: [B, text_length, d_model]
        visual_features: [B, num_patches, d_model]
        padding_mask: [B, text_length]

    Output:
        decoder_output: [B, text_length, d_model]
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 4,
            d_ff: int = 2048,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        self.masked_self_attention = (
            MaskedMultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
        )

        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

        self.feed_forward = TextDecoderFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

    def forward(self, text_states: torch.Tensor, visual_features: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --------------------------------------------------
        # Bước 1: Masked Self-Attention.
        # --------------------------------------------------
        (
            self_attention_residual,
            _,
            self_attention_weights
        ) = self.masked_self_attention(text_embeddings=text_states, padding_mask=padding_mask)

        # --------------------------------------------------
        # Bước 2: Cross-Attention.
        # --------------------------------------------------
        (
            cross_attention_residual,
            _,
            cross_attention_weights
        ) = self.cross_attention(text_states=self_attention_residual, visual_features=visual_features)

        # --------------------------------------------------
        # Bước 3: Feed-Forward Network.
        # --------------------------------------------------
        (
            decoder_output,
            _
        ) = self.feed_forward(text_states=cross_attention_residual)

        return decoder_output, self_attention_weights, cross_attention_weights


class TextTransformerDecoder(nn.Module):
    """
    Text Transformer Decoder gồm nhiều Decoder Block.

    Input:
        text_embeddings: [B, text_length, d_model]
        visual_features: [B, num_patches, d_model]
        padding_mask: [B, text_length]

    Output:
        decoder_features: [B, text_length, d_model]
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 4,
            d_ff: int = 2048,
            num_layers: int = 4,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"num_layers phải lớn hơn 0, nhận được {num_layers}.")

        self.num_layers = num_layers

        # Mỗi Decoder Block có bộ tham số riêng.
        self.decoder_blocks = nn.ModuleList([
            TextDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

    def forward(
            self,
            text_embeddings: torch.Tensor,
            visual_features: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
            return_attention_weights: bool = False
    ) -> (torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]):

        decoder_states = text_embeddings
        all_self_attention_weights: list[torch.Tensor] = []
        all_cross_attention_weights: list[torch.Tensor] = []

        for decoder_block in self.decoder_blocks:
            (
                decoder_states,
                self_attention_weights,
                cross_attention_weights
            ) = decoder_block(text_states=decoder_states, visual_features=visual_features, padding_mask=padding_mask)

            if return_attention_weights:
                all_self_attention_weights.append(self_attention_weights)
                all_cross_attention_weights.append(cross_attention_weights)

        decoder_features = self.final_norm(decoder_states)

        if return_attention_weights:
            return (
                decoder_features,
                all_self_attention_weights,
                all_cross_attention_weights
            )

        return decoder_features
