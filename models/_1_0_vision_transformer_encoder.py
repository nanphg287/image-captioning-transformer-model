from __future__ import annotations

import torch
import torch.nn as nn

from config import Config
from models._1_3_multi_head_output_projection import MultiHeadOutputProjection
from models._1_2_scaled_dot_product_attention import ScaledDotProductAttention
from models._1_4_vit_encoder_feed_forward import ViTEncoderFeedForward
from models._1_1_multihead_qkv_projection import MultiHeadQKVProjection


class VisionTransformerEncoderBlock(nn.Module):
    """
    Một Vision Transformer Encoder Block sử dụng Pre-LayerNorm.

    Luồng xử lý:

    X
    ├─ LayerNorm
    ├─ QKV Projection
    ├─ Multi-Head Self-Attention
    ├─ Output Projection
    ├─ Residual Connection
    ├─ LayerNorm
    ├─ Feed-Forward Network
    └─ Residual Connection

    Input: [B, num_patches, d_model]
    Output: [B, num_patches, d_model]
    """

    def __init__(self, d_model: int = 512, num_heads: int = 4, d_ff: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-6):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} phải chia hết cho num_heads={num_heads}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # LayerNorm trước Self-Attention.
        self.attention_input_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

        # Tạo Q, K, V và chia thành nhiều head.
        self.qkv_projection = MultiHeadQKVProjection(d_model=d_model, num_heads=num_heads)

        # Tính softmax(QKᵀ / sqrt(head_dim))V.
        self.scaled_dot_product_attention = ScaledDotProductAttention(head_dim=self.head_dim, dropout=dropout)

        # Ghép các head và nhân với W_O.
        self.multi_head_output_projection = MultiHeadOutputProjection(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Residual Connection và Feed-Forward Network.
        self.feed_forward_block = ViTEncoderFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

    def forward(self, image_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
        encoder_output: [B, num_patches, d_model]
        attention_weights: [B, num_heads, num_patches, num_patches]
        """

        self._validate_input(image_tokens)

        # --------------------------------------------------
        # Bước 1: Pre-LayerNorm.
        # --------------------------------------------------
        normalized_image_tokens = self.attention_input_norm(image_tokens)

        # --------------------------------------------------
        # Bước 2: Tạo Query, Key và Value.
        # --------------------------------------------------
        query, key, value = self.qkv_projection(normalized_image_tokens)

        # --------------------------------------------------
        # Bước 3: Scaled Dot-Product Attention.
        # --------------------------------------------------
        (
            attention_output,
            _,
            attention_weights
        ) = self.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value
        )

        # --------------------------------------------------
        # Bước 4: Ghép head và Output Projection W_O.
        # --------------------------------------------------
        (
            _,
            projected_attention_output
        ) = self.multi_head_output_projection(
            attention_output
        )

        # --------------------------------------------------
        # Bước 5: Residual và Feed-Forward Network.
        # --------------------------------------------------
        (
            _,
            _,
            encoder_output
        ) = self.feed_forward_block(
            image_tokens=image_tokens,
            projected_attention_output=(
                projected_attention_output
            )
        )

        return encoder_output, attention_weights

    def _validate_input(self, image_tokens: torch.Tensor) -> None:
        if image_tokens.ndim != 3:
            raise ValueError(
                f"image_tokens phải có shape [B, num_patches, d_model], nhận được {tuple(image_tokens.shape)}.")

        if image_tokens.shape[-1] != self.d_model:
            raise ValueError(f"Chiều cuối phải bằng d_model={self.d_model}, nhận được {image_tokens.shape[-1]}.")


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer Encoder gồm nhiều Encoder Block.
    Input: [B, num_patches, d_mode
    Output: [B, num_patches, d_model]
    """

    def __init__(
            self,
            d_model: int = Config.d_model,
            num_heads: int = Config.num_heads,
            d_ff: int = Config.d_ff,
            num_layers: int = Config.num_encoder_layers,
            dropout: float = Config.attention_dropout,
            layer_norm_eps: float = Config.layer_norm_eps
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"num_layers phải lớn hơn 0, nhận được {num_layers}.")

        self.d_model = d_model
        self.num_layers = num_layers

        # Mỗi phần tử là một Encoder Block riêng biệt,
        # có bộ trọng số riêng.
        self.encoder_blocks = nn.ModuleList([
            VisionTransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])

        # LayerNorm cuối cùng sau toàn bộ Encoder Block.
        self.final_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

    def forward(self, image_tokens: torch.Tensor, return_attention_weights: bool = False) \
            -> (torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]):
        """
        Nếu return_attention_weights=False: trả về visual_features.
        Nếu return_attention_weights=True: trả về visual_features và Attention Weight của từng Encoder Block.
        """

        encoder_output = image_tokens

        collected_attention_weights: list[torch.Tensor] = []

        for encoder_block in self.encoder_blocks:
            (encoder_output, attention_weights) = encoder_block(encoder_output)

            if return_attention_weights:
                collected_attention_weights.append(attention_weights)

        # Final LayerNorm.
        visual_features = self.final_norm(encoder_output)

        if return_attention_weights:
            return visual_features, collected_attention_weights

        return visual_features
