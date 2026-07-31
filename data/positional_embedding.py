import torch
import torch.nn as nn

from config import Config
from data.patching_embedding import PatchEmbedding


class PositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding
    """
    def __init__(self, num_patches: int, d_model: int = Config.d_model):
        super().__init__()
        # nn.Embedding(num_patches, d_model): bảng tra cứu, mỗi vị trí patch
        # (0, 1, 2, ..., 195) có 1 vector d_model chiều RIÊNG, học được qua train
        self.position_embedding_table = nn.Embedding(num_patches, d_model)
        nn.init.trunc_normal_(self.position_embedding_table.weight, std=0.02)
        self.dropout = nn.Dropout(Config.attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_patches, d_model = x.shape

        if num_patches > self.position_embedding_table.num_embeddings:
            raise ValueError(
                f"Input có {num_patches} patches nhưng positional embedding "
                f"chỉ hỗ trợ {self.position_embedding_table.num_embeddings}"
            )

        if d_model != self.position_embedding_table.embedding_dim:
            raise ValueError(
                f"d_model của input là {d_model}, expected "
                f"{self.position_embedding_table.embedding_dim}"
            )

        positions = torch.arange(num_patches, device=x.device)
        return x + self.position_embedding_table(positions)


# ==============================
# Kiểm tra nhanh - Ráp toàn bộ pipeline ảnh
# ==============================
if __name__ == "__main__":
    dummy_batch = torch.randn(4, 3, 224, 224)  # batch 4 ảnh

    patch_size = 16
    d_model = 512
    num_patches = (224 // patch_size) ** 2  # = 196

    patch_embed = PatchEmbedding(patch_size=patch_size, in_channels=3, d_model=d_model)
    pos_embed = PositionalEmbedding(num_patches=num_patches, d_model=d_model)

    x = patch_embed(dummy_batch)   # (4, 196, 512) - chưa có thông tin vị trí
    x = pos_embed(x)               # (4, 196, 512) - đã cộng thêm vị trí

    print("Sau Patch Embedding:", patch_embed(dummy_batch).shape)  # torch.Size([4, 196, 512])
    print("Sau Positional Embedding:", x.shape)                     # torch.Size([4, 196, 512])
    print("Position table shape:", pos_embed.position_embedding_table.weight.shape)  # (196, 512)