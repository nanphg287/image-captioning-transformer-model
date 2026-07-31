from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import Config


class ScaledDotProductAttention(nn.Module):
    """
    Tính Scaled Dot-Product Attention cho nhiều head.

    Input:
        query: [B, num_heads, N, head_dim]
        key:   [B, num_heads, N, head_dim]
        value: [B, num_heads, N, head_dim]

    Output:
        attention_output:
            [B, num_heads, N, head_dim]

        attention_scores:
            [B, num_heads, N, N]

        attention_weights:
            [B, num_heads, N, N]
    """

    def __init__(self, head_dim: int, dropout: float = Config.attention_dropout):
        super().__init__()

        if head_dim <= 0:
            raise ValueError(f"head_dim phải lớn hơn 0, nhận được {head_dim}.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout phải nằm trong [0,1), nhận được {dropout}.")

        self.head_dim = head_dim

        # sqrt(head_dim).
        # Với head_dim=128:
        # scale = 1 / sqrt(128)
        self.scale = 1.0 / math.sqrt(head_dim)

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) \
            -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor
            ]:
        """
        Thực hiện Scaled Dot-Product Attention.
        """

        self._validate_inputs(query, key, value)

        # --------------------------------------------------
        # Bước 1: Chuyển vị hai chiều cuối của Key.
        #
        # Key:
        # [B, num_heads, N, head_dim]
        #
        # Key transpose:
        # [B, num_heads, head_dim, N]
        # --------------------------------------------------
        key_transposed = key.transpose(-2, -1)

        # --------------------------------------------------
        # Bước 2: Tính Q @ K^T.
        #
        # [B, H, N, Dh] @ [B, H, Dh, N]
        #                  ↓
        #             [B, H, N, N]
        # --------------------------------------------------
        attention_scores = torch.matmul(query, key_transposed)

        # --------------------------------------------------
        # Bước 3: Scale Attention Score.
        #
        # attention_scores / sqrt(head_dim)
        # --------------------------------------------------
        attention_scores = attention_scores * self.scale

        # --------------------------------------------------
        # Bước 4: Softmax theo chiều Key/Patch cuối cùng.
        #
        # Mỗi patch sẽ có một phân phối trọng số trên
        # toàn bộ 196 patch.
        #
        # Shape không đổi:
        # [B, H, N, N]
        # --------------------------------------------------
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Dropout chỉ áp dụng khi model ở chế độ train.
        dropped_attention_weights = self.attention_dropout(attention_weights)

        # --------------------------------------------------
        # Bước 5: Nhân trọng số Attention với Value.
        #
        # [B, H, N, N] @ [B, H, N, Dh]
        #                  ↓
        #             [B, H, N, Dh]
        # --------------------------------------------------
        attention_output = torch.matmul(dropped_attention_weights, value)

        return attention_output, attention_scores, attention_weights

    def _validate_inputs(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Kiểm tra shape và device của Q, K, V.
        """

        if query.ndim != 4:
            raise ValueError(
                "Query phải có shape "
                "[B, num_heads, N, head_dim], "
                f"nhận được {tuple(query.shape)}."
            )

        if key.ndim != 4:
            raise ValueError(
                "Key phải có shape "
                "[B, num_heads, N, head_dim], "
                f"nhận được {tuple(key.shape)}."
            )

        if value.ndim != 4:
            raise ValueError(
                "Value phải có shape "
                "[B, num_heads, N, head_dim], "
                f"nhận được {tuple(value.shape)}."
            )

        if query.shape != key.shape:
            raise ValueError(
                f"Query và Key phải cùng shape: "
                f"query={tuple(query.shape)}, "
                f"key={tuple(key.shape)}."
            )

        if key.shape != value.shape:
            raise ValueError(
                f"Key và Value phải cùng shape: "
                f"key={tuple(key.shape)}, "
                f"value={tuple(value.shape)}."
            )

        if query.shape[-1] != self.head_dim:
            raise ValueError(
                f"Chiều cuối của Query phải bằng "
                f"head_dim={self.head_dim}, "
                f"nhận được {query.shape[-1]}."
            )

        if not (query.device == key.device == value.device):
            raise ValueError(
                "Query, Key và Value phải nằm trên cùng device: "
                f"query={query.device}, "
                f"key={key.device}, "
                f"value={value.device}."
            )
