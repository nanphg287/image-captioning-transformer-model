from __future__ import annotations

import torch
import torch.nn as nn


class TextDecoderFeedForward(nn.Module):
    """
    Feed-Forward Network cuối Text Decoder Block.

    Input: text_states: [B, sequence_length, d_model]

    Output:
        decoder_output: [B, sequence_length, d_model]
        feed_forward_output: [B, sequence_length, d_model]
    """

    def __init__(
            self,
            d_model: int = 512,
            d_ff: int = 2048,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-6
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(normalized_shape=d_model, eps=layer_norm_eps)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, text_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-LayerNorm.
        normalized_text = self.input_norm(text_states)

        # [B,L,512] -> [B,L,2048] -> [B,L,512]
        feed_forward_output = self.feed_forward(normalized_text)

        # Residual Connection.
        decoder_output = (text_states + feed_forward_output)

        return decoder_output, feed_forward_output
