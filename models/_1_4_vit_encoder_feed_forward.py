from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class ViTEncoderFeedForward(nn.Module):
    """
    Thực hiện phần còn lại của một ViT Encoder Block:

    1. Residual Connection sau Self-Attention.
    2. Layer Normalization trước Feed-Forward.
    3. Feed-Forward Network.
    4. Residual Connection sau Feed-Forward.

    Input:
        image_tokens: [B, N, d_model]
        projected_attention_output: [B, N, d_model]
    Output:
        encoder_output: [B, N, d_model]
    """

    def __init__(
            self,
            d_model: int = Config.d_model,
            d_ff: int = 2048,
            dropout: float = Config.attention_dropout,
            layer_norm_eps: float = Config.layer_norm_eps
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model phải lớn hơn 0, nhận được {d_model}.")

        if d_ff <= 0:
            raise ValueError(f"d_ff phải lớn hơn 0, nhận được {d_ff}.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout phải nằm trong [0,1), nhận được {dropout}.")

        self.d_model = d_model
        self.d_ff = d_ff

        # LayerNorm trước Feed-Forward Network.
        self.feed_forward_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

        # Feed-Forward Network:
        # 512 -> 2048 -> 512
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Khởi tạo trọng số của hai Linear Layer trong FFN.
        """

        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image_tokens: torch.Tensor, projected_attention_output: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trả về:

        attention_residual: Kết quả sau residual connection thứ nhất.
        feed_forward_output: Kết quả riêng của Feed-Forward Network.
        encoder_output: Kết quả cuối cùng của Encoder Block.
        """

        self._validate_inputs(image_tokens, projected_attention_output)

        # --------------------------------------------------
        # Residual Connection thứ nhất.
        #
        # X1 = X + MultiHeadSelfAttention(LayerNorm(X))
        #
        # [B, N, 512] + [B, N, 512]
        #              ↓
        #         [B, N, 512]
        # --------------------------------------------------
        attention_residual = (image_tokens + projected_attention_output)

        # --------------------------------------------------
        # LayerNorm trước Feed-Forward Network.
        # --------------------------------------------------
        normalized_attention = self.feed_forward_norm(attention_residual)

        # --------------------------------------------------
        # Feed-Forward Network.
        #
        # [B, N, 512]
        #      ↓
        # [B, N, 2048]
        #      ↓ GELU + Dropout
        # [B, N, 512]
        # --------------------------------------------------
        feed_forward_output = self.feed_forward(normalized_attention)

        # --------------------------------------------------
        # Residual Connection thứ hai.
        #
        # X2 = X1 + FFN(LayerNorm(X1))
        # --------------------------------------------------
        encoder_output = (attention_residual + feed_forward_output)

        return attention_residual, feed_forward_output, encoder_output

    def _validate_inputs(self, image_tokens: torch.Tensor, projected_attention_output: torch.Tensor) -> None:
        if image_tokens.ndim != 3:
            raise ValueError("Image tokens phải có shape [B, N, d_model], nhận được {tuple(image_tokens.shape)}.")

        if projected_attention_output.ndim != 3:
            raise ValueError(
                "Projected attention output phải có shape "
                "[B, N, d_model], "
                f"nhận được "
                f"{tuple(projected_attention_output.shape)}."
            )

        if image_tokens.shape != projected_attention_output.shape:
            raise ValueError(
                "Image tokens và projected attention output "
                "phải cùng shape: "
                f"image_tokens={tuple(image_tokens.shape)}, "
                f"attention={tuple(projected_attention_output.shape)}."
            )

        if image_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"Chiều cuối phải bằng d_model={self.d_model}, "
                f"nhận được {image_tokens.shape[-1]}."
            )

        if image_tokens.device != projected_attention_output.device:
            raise ValueError(
                "Hai tensor phải nằm trên cùng device: "
                f"image_tokens={image_tokens.device}, "
                f"attention={projected_attention_output.device}."
            )
