from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models import resnet50

from config import Config


class CNNEncoder(nn.Module):
    """
    CNN encoder dùng backbone ResNet-50.

    Input và output giữ nguyên giao diện của Vision Transformer Encoder:
    [B, num_patches, d_model] -> [B, num_patches, d_model].
    """

    def __init__(
            self,
            d_model: int = Config.d_model,
            num_heads: int = Config.num_heads,
            d_ff: int = Config.d_ff,
            num_layers: int = Config.num_encoder_layers,
            dropout: float = Config.attention_dropout,
            layer_norm_eps: float = Config.layer_norm_eps
    ):
        super().__init__()

        # Không tải trọng số ImageNet để giữ nguyên cách huấn luyện from-scratch
        # của dự án. Các tham số không dùng thuộc giao diện encoder cũ được giữ
        # lại để nơi khởi tạo model không phải thay đổi.
        _ = (num_heads, d_ff, num_layers, dropout)
        backbone = resnet50(
            weights=None,
            replace_stride_with_dilation=[True, True, True]
        )

        # Token ảnh đã có d_model kênh, nên chỉ thay lớp vào của ResNet-50.
        backbone.conv1 = nn.Conv2d(
            d_model,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.d_model = d_model
        self.resnet_features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        self.output_projection = nn.Conv2d(
            backbone.fc.in_features,
            d_model,
            kernel_size=1
        )
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        self._validate_input(image_tokens)

        batch_size, num_patches, _ = image_tokens.shape
        grid_size = math.isqrt(num_patches)

        feature_map = image_tokens.transpose(1, 2).reshape(
            batch_size,
            self.d_model,
            grid_size,
            grid_size
        )
        feature_map = self.resnet_features(feature_map)
        feature_map = self.output_projection(feature_map)

        # Giữ nguyên số visual token để toàn bộ decoder phía sau không đổi.
        feature_map = functional.interpolate(
            feature_map,
            size=(grid_size, grid_size),
            mode="bilinear",
            align_corners=False
        )
        visual_features = feature_map.flatten(2).transpose(1, 2)
        return self.final_norm(visual_features)

    def _validate_input(self, image_tokens: torch.Tensor) -> None:
        if image_tokens.ndim != 3:
            raise ValueError(
                "image_tokens phải có shape [B, num_patches, d_model], "
                f"nhận được {tuple(image_tokens.shape)}."
            )

        if image_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"Chiều cuối phải bằng d_model={self.d_model}, "
                f"nhận được {image_tokens.shape[-1]}."
            )

        grid_size = math.isqrt(image_tokens.shape[1])
        if grid_size * grid_size != image_tokens.shape[1]:
            raise ValueError(
                "num_patches phải là số chính phương để CNN khôi phục lưới ảnh, "
                f"nhận được {image_tokens.shape[1]}."
            )
