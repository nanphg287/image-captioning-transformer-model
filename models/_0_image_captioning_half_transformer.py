from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from models._4_0_language_modeling_head import LanguageModelingHead
from models._2_text_embedding import TextEmbedding
from models._3_0_text_transformer_decoder import TextTransformerDecoder


class ImageCaptioningHalfTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            vocabulary_size: int,
            max_caption_length: int,
            pad_token_id: int,
            d_model: int = 512,
            num_heads: int = 4,
            d_ff: int = 2048,
            num_decoder_layers: int = 4,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-6,
            freeze_backbone: bool = True
    ):
        super().__init__()

        patches_per_side = image_size // patch_size
        num_patches = patches_per_side ** 2

        # Khởi tạo backbone ResNet-50 pre-trained
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Bỏ đi layer global average pooling và fully connected ở cuối
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Bộ chuyển đổi số kênh đặc trưng của ResNet (2048) về d_model (512)
        self.feature_projection = nn.Linear(2048, d_model)

        # Đóng băng trọng số của backbone nếu có yêu cầu
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.text_embedding = TextEmbedding(
            vocabulary_size=vocabulary_size,
            d_model=d_model,
            max_sequence_length=max_caption_length,
            pad_token_id=pad_token_id,
            dropout=dropout
        )

        self.text_transformer_decoder = (
            TextTransformerDecoder(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_layers=num_decoder_layers,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            )
        )

        self.language_modeling_head = LanguageModelingHead(d_model=d_model, vocabulary_size=vocabulary_size)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # Trích xuất đặc trưng qua ResNet-50 -> [B, 2048, 7, 7]
        visual_features = self.backbone(images)

        # Trải phẳng chiều rộng/cao -> [B, 2048, 49]
        visual_features = visual_features.flatten(2)

        # Đảo chiều để chuyển thành dạng chuỗi -> [B, 49, 2048]
        visual_features = visual_features.transpose(1, 2)

        # Ánh xạ từ 2048 về d_model (512) -> [B, 49, d_model]
        projected_features = self.feature_projection(visual_features)

        return projected_features

    def decode_captions(
            self,
            decoder_input_ids: torch.Tensor,
            visual_features: torch.Tensor,
            decoder_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        text_embeddings = self.text_embedding(decoder_input_ids)

        decoder_features = (
            self.text_transformer_decoder(
                text_embeddings=text_embeddings,
                visual_features=visual_features,
                padding_mask=decoder_padding_mask
            )
        )

        logits = self.language_modeling_head(decoder_features)

        return logits

    def forward(
            self,
            images: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            decoder_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        visual_features = self.encode_images(images)

        logits = self.decode_captions(
            decoder_input_ids=decoder_input_ids,
            visual_features=visual_features,
            decoder_padding_mask=decoder_padding_mask
        )

        return logits
