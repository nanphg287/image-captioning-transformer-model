from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class ConvStem(nn.Module):
    """
    Convolutional Stem thay cho patchify tuyen tinh.

    Y tuong (Xiao et al. 2021, "Early Convolutions Help Transformers See Better"):
    thay vi cat anh thanh cac o 16x16 roi chieu tuyen tinh, ta dung mot stack
    conv 3x3 stride-2. Ket qua van la 196 token nhung moi token duoc tao ra tu
    mot vung CHONG LAN voi cac token ben canh -> mang san locality bias ma ViT
    von phai tu hoc tu du lieu.

    224 -> 112 -> 56 -> 28 -> 14   (4 lan stride-2 = ha 16x, bang patch_size)

    Input:  [B, 3, 224, 224]
    Output: [B, 196, d_model]
    """

    def __init__(
            self,
            d_model: int = Config.d_model,
            in_channels: int = Config.in_channels,
            channels: tuple[int, ...] = Config.conv_stem_channels,
            patch_size: int = Config.patch_size,
            use_batch_norm: bool = True
    ):
        super().__init__()

        downsample_factor = 2 ** len(channels)
        if downsample_factor != patch_size:
            raise ValueError(
                f"conv stem ha {downsample_factor}x nhung patch_size={patch_size}. "
                f"So token se khong khop voi positional embedding."
            )

        if channels[-1] > d_model:
            raise ValueError(
                f"Kenh cuoi ({channels[-1]}) khong nen lon hon d_model ({d_model})."
            )

        self.d_model = d_model
        self.patch_size = patch_size

        layers: list[nn.Module] = []
        current_channels = in_channels

        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=not use_batch_norm
                )
            )

            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            else:
                # GroupNorm khong phu thuoc batch size va khong co train/eval
                # discrepancy - an toan hon neu batch nho hoac dung grad accumulation.
                layers.append(nn.GroupNorm(num_groups=8, num_channels=out_channels))

            layers.append(nn.GELU())
            current_channels = out_channels

        # 1x1 conv cuoi: chieu ve d_model. Khong norm, khong activation -
        # tuong duong voi lop Linear projection cua patchify goc.
        layers.append(
            nn.Conv2d(current_channels, d_model, kernel_size=1, bias=True)
        )

        self.stem = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"Input phai co 4 chieu [B, C, H, W], nhan duoc {tuple(images.shape)}."
            )

        # [B, 3, 224, 224] -> [B, d_model, 14, 14]
        feature_map = self.stem(images)

        # [B, d_model, 14, 14] -> [B, d_model, 196] -> [B, 196, d_model]
        # LUU Y: transpose, KHONG phai view - view se xao tron thu tu patch.
        return feature_map.flatten(2).transpose(1, 2)


class ConvStemImageEmbedding(nn.Module):
    """
    Bo chuyen doi: chon giua ConvStem va patchify tuyen tinh qua Config.

    Ca hai deu tra ve [B, num_patches, d_model] nen phan con lai cua pipeline
    (PositionalEmbedding, VisionEncoder, CrossAttention) khong can sua gi.
    """

    def __init__(
            self,
            d_model: int = Config.d_model,
            patch_size: int = Config.patch_size,
            in_channels: int = Config.in_channels,
            use_conv_stem: bool = Config.use_conv_stem,
            channels: tuple[int, ...] = Config.conv_stem_channels
    ):
        super().__init__()
        self.use_conv_stem = use_conv_stem

        if use_conv_stem:
            self.embedding = ConvStem(
                d_model=d_model,
                in_channels=in_channels,
                channels=channels,
                patch_size=patch_size
            )
        else:
            # Tuong duong hoan toan voi patchify + nn.Linear cu,
            # nhung nhanh hon (khong con 2 lan .contiguous()).
            self.embedding = nn.Sequential()
            self.embedding.add_module(
                "proj",
                nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_conv_stem:
            return self.embedding(images)

        return self.embedding(images).flatten(2).transpose(1, 2)