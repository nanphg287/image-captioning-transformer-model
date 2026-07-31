import torch
import torch.nn as nn

from config import Config


def patchify(images: torch.Tensor, patch_size: int = Config.patch_size) -> torch.Tensor:
    """
    images: tensor shape (B, C, H, W)
    patch_size: kích thước 1 cạnh patch (patch vuông)

    Trả về: tensor shape (B, num_patches, patch_size * patch_size * C)
    """
    B, C, H, W = images.shape

    assert H % patch_size == 0 and W % patch_size == 0, f"Kích thước ảnh ({H}x{W}) phải chia hết cho patch_size ({patch_size})"

    # unfold theo chiều Height (dim=2), rồi theo chiều Width (dim=3)
    # sau mỗi lần unfold, tensor sẽ có thêm 1 chiều mới chứa các "cửa sổ trượt"
    p = images.unfold(2, patch_size, patch_size)  # (B, C, H/p, W, p)
    p = p.unfold(3, patch_size, patch_size)  # (B, C, H/p, W/p, p, p)

    # Hiện tại: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
    # Cần đưa về: (B, num_patches, C * patch_size * patch_size)

    B, C, nH, nW, pH, pW = p.shape

    # Gom 2 chiều (nH, nW) thành 1 chiều num_patches
    p = p.contiguous().view(B, C, nH * nW, pH, pW)

    # Đưa chiều num_patches lên trước, rồi flatten (C, pH, pW) thành 1 vector
    p = p.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, pH, pW)
    p = p.contiguous().view(B, nH * nW, C * pH * pW)

    return p


class PatchEmbedding(nn.Module):
    """
    Bước Flatten + Linear Projection:
    Nhận patch thô (đã flatten sẵn từ patchify) -> chiếu thành vector d_model chiều
    """

    def __init__(self, patch_size: int = Config.patch_size, in_channels: int = 3, d_model: int = Config.d_model):
        super().__init__()
        self.patch_size = patch_size

        # patch_dim: số giá trị pixel thô trong 1 patch sau khi flatten
        # ví dụ patch_size=16, in_channels=3 -> 16*16*3 = 768
        patch_dim = patch_size * patch_size * in_channels

        # Linear projection: patch thô (patch_dim) -> vector embedding (d_model)
        # Đây là lớp CÓ trọng số học được (weight, bias), khác hoàn toàn với patchify
        # (patchify chỉ là thao tác cắt/sắp xếp lại tensor, không có tham số học)
        self.projection = nn.Linear(patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) - ảnh gốc, CHƯA patchify
        patches = patchify(x, self.patch_size)  # (B, num_patches, patch_dim)

        # patches đã ở dạng flatten sẵn (patch_dim = C*p*p) từ hàm patchify
        embeddings = self.projection(patches)  # (B, num_patches, d_model)

        return embeddings


class ConvStem(nn.Module):
    """224x224 -> 14x14 = 196 token, mỗi token có receptive field chồng lấn."""
    def __init__(self, d_model=256, ch=(32, 64, 128, 256)):
        super().__init__()
        layers, c_in = [], 3
        for c in ch:                       # 4 lần stride-2: 224->112->56->28->14
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(c), nn.GELU()]
            c_in = c
        layers += [nn.Conv2d(c_in, d_model, 1)]
        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x).flatten(2).transpose(1, 2)   # [B, 196, d_model]


if __name__ == "__main__":
    dummy_image = torch.randn(1, 3, 224, 224)  # giả lập batch 1 ảnh đã resize

    patch_embed = PatchEmbedding(patch_size=16, in_channels=3, d_model=512)
    out = patch_embed(dummy_image)

    print("Input shape:", dummy_image.shape)  # torch.Size([1, 3, 224, 224])
    print("Output shape:", out.shape)  # torch.Size([1, 196, 512])

    # Kiểm tra trọng số Linear có đúng shape mong đợi không
    print("Linear weight shape:", patch_embed.projection.weight.shape)  # (512, 768)
    print("Linear bias shape:", patch_embed.projection.bias.shape)  # (512,)
