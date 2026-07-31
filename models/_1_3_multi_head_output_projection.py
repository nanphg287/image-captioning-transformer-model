from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class MultiHeadOutputProjection(nn.Module):
    """
    Ghép kết quả của các Attention Head và thực hiện
    Linear Projection qua ma trận W_O.

    Input:
        attention_output:
            [B, num_heads, N, head_dim]

    Output:
        projected_output:
            [B, N, d_model]
    """

    def __init__(
            self,
            d_model: int = Config.d_model,
            num_heads: int = Config.num_heads,
            dropout: float = Config.attention_dropout
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho num_heads ({num_heads}).")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout phải nằm trong [0,1), nhận được {dropout}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Ma trận W_O:
        # [d_model, d_model]
        self.output_projection = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.output_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Khởi tạo ma trận W_O.
        """

        nn.init.xavier_uniform_(self.output_projection.weight)

        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def merge_heads(self, attention_output: torch.Tensor) -> torch.Tensor:
        """
        Ghép nhiều Attention Head.

        [B, H, N, Dh]
             ↓ permute
        [B, N, H, Dh]
             ↓ reshape
        [B, N, H * Dh]
             ↓
        [B, N, d_model]
        """

        if attention_output.ndim != 4:
            raise ValueError(
                "Attention output phải có shape "
                "[B, num_heads, N, head_dim], "
                f"nhận được {tuple(attention_output.shape)}."
            )

        batch_size, num_heads, num_patches, head_dim = attention_output.shape

        if num_heads != self.num_heads:
            raise ValueError(f"Số head phải bằng {self.num_heads}, nhận được {num_heads}.")

        if head_dim != self.head_dim:
            raise ValueError(f"head_dim phải bằng {self.head_dim}, nhận được {head_dim}.")

        # [B, H, N, Dh] -> [B, N, H, Dh]
        merged_output = attention_output.permute(0, 2, 1, 3)

        # Bảo đảm vùng nhớ liên tục trước khi reshape.
        merged_output = merged_output.contiguous()

        # [B, N, H, Dh] -> [B, N, H * Dh]
        merged_output = merged_output.reshape(batch_size, num_patches, self.d_model)

        return merged_output

    def forward(self, attention_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:

        merged_output: Kết quả sau khi ghép head, trước W_O.
        projected_output: Kết quả sau W_O và dropout.
        """

        # [B, 4, 196, 128] -> [B, 196, 512]
        merged_output = self.merge_heads(attention_output)

        # Nhân với W_O:
        # [B, 196, 512] -> [B, 196, 512]
        projected_output = self.output_projection(merged_output)

        projected_output = self.output_dropout(projected_output)

        return merged_output, projected_output
