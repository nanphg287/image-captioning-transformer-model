from __future__ import annotations

import math

import torch
import torch.nn as nn


class TextEmbedding(nn.Module):
    """
    Chuyển caption token ID thành embedding và cộng learned positional embedding.

    Input: token_ids: [B, sequence_length]
    Output: text_embeddings: [B, sequence_length, d_model]
    """

    def __init__(
            self,
            vocabulary_size: int,
            d_model: int,
            max_sequence_length: int,
            pad_token_id: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

        # Mỗi từ trong vocabulary có một vector d_model chiều.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id
        )

        # Mỗi vị trí trong câu có một vector riêng.
        self.position_embedding = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=d_model)

        self.dropout = nn.Dropout(dropout)

        # Scale token embedding theo kiến trúc Transformer.
        self.embedding_scale = 1.0

        self.reset_parameters()

    # TODO: Cần xem lại đoạn này
    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Padding embedding luôn bằng 0.
        with torch.no_grad():
            self.token_embedding.weight[self.pad_token_id].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = (token_ids.shape)

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence_length={sequence_length} vượt quá "
                f"max_sequence_length="
                f"{self.max_sequence_length}."
            )

        # [B, L] -> [B, L, d_model]
        token_embeddings = self.token_embedding(token_ids)

        token_embeddings = (token_embeddings * self.embedding_scale)

        # Vị trí 0, 1, 2, ..., L-1.
        positions = torch.arange(sequence_length, device=token_ids.device)

        # [L] -> [L, d_model]
        position_embeddings = (self.position_embedding(positions))

        # Broadcasting:
        # [B, L, d_model] + [L, d_model]
        text_embeddings = (token_embeddings + position_embeddings)

        return self.dropout(text_embeddings)
