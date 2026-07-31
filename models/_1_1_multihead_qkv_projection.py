import torch
import torch.nn as nn

from config import Config


class MultiHeadQKVProjection(nn.Module):
    """
        Tạo Query, Key, Value và chia thành nhiều Attention Head.

        Input:
            x: [B, N, d_model]

        Output:
            query: [B, num_heads, N, head_dim]
            key:   [B, num_heads, N, head_dim]
            value: [B, num_heads, N, head_dim]
    """

    def __init__(self, d_model: int = Config.d_model, num_heads: int = Config.num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Các phép chiếu tuyến tính tương ứng với W_Q, W_K, W_V.
        # Về mặt toán học: W_Q, W_K, W_V: [512, 512]
        self.query_projection = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.key_projection = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.value_projection = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Khởi tạo trọng số cho W_Q, W_K và W_V.
        """

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
        Chuyển tensor:

        [B, N, d_model]
                ↓
        [B, N, num_heads, head_dim]
                ↓
        [B, num_heads, N, head_dim]
        """

        batch_size, num_patches, d_model = tensor.shape

        tensor = tensor.reshape(batch_size, num_patches, self.num_heads, self.head_dim)
        tensor = tensor.permute(0, 2, 1, 3)

        # Bảo đảm vùng nhớ liên tục cho các phép reshape tiếp theo.
        return tensor.contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, N, d_model]
        """

        if x.ndim != 3:
            raise ValueError(f"Input phải có 3 chiều [B, N, d_model], nhưng nhận được shape {tuple(x.shape)}.")

        if x.shape[-1] != self.d_model:
            raise ValueError(f"Chiều cuối của input phải bằng {self.d_model}, nhưng nhận được {x.shape[-1]}.")

        # Linear Projection:
        # [B, N, 512] -> [B, N, 512]
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # Chia thành 4 head:
        # [B, N, 512] -> [B, 4, N, 128]
        query = self.split_into_heads(query)
        key = self.split_into_heads(key)
        value = self.split_into_heads(value)

        return query, key, value
