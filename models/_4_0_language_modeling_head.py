from __future__ import annotations

import torch
import torch.nn as nn


class LanguageModelingHead(nn.Module):
    """
    Chuyển Decoder Features thành logits trên vocabulary.

    Input:
        decoder_features: [B, sequence_length, d_model]

    Output:
        logits: [B, sequence_length, vocabulary_size]
    """

    def __init__(self, d_model: int, vocabulary_size: int):
        super().__init__()
        self.output_projection = nn.Linear(in_features=d_model, out_features=vocabulary_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.output_projection.weight)

        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, decoder_features: torch.Tensor) -> torch.Tensor:
        return self.output_projection(decoder_features)
