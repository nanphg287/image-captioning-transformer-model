from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

IMAGE_DIR = RAW_DATA_DIR / "Images"
CAPTION_FILE = RAW_DATA_DIR / "captions.txt"

IMAGE_TO_CAPTIONS_FILE = (
        PROCESSED_DATA_DIR / "image_to_captions.json"
)

TRAIN_DATA_MAPPING = PROCESSED_DATA_DIR / "train.json"
VALIDATION_DATA_FILE = PROCESSED_DATA_DIR / "validation.json"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.json"

VOCABULARY_FILE = PROCESSED_DATA_DIR / "vocabulary.json"

# ImageNet statistics - dung cho ca train va inference, khong duoc lech nhau.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

@dataclass
class Config:
    image_dir: Path = IMAGE_DIR
    resize_image_dir: Path = PROCESSED_DATA_DIR / "resize_image"
    caption_file: Path = CAPTION_FILE
    image_to_captions_file: Path = IMAGE_TO_CAPTIONS_FILE
    train_data_file: Path = TRAIN_DATA_MAPPING
    validation_data_file: Path = VALIDATION_DATA_FILE
    test_data_file: Path = TEST_DATA_FILE
    vocabulary_file: Path = VOCABULARY_FILE
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Conv stem thay cho patchify tuyen tinh (them locality bias cho ViT).
    # 4 lan stride-2: 224 -> 112 -> 56 -> 28 -> 14  => 196 token, giu nguyen so token.
    use_conv_stem: bool = True
    conv_stem_channels: tuple[int, ...] = (32, 64, 128, 256)

    # ------------------------------------------------------------------
    # Transformer  (THU NHO: 38.1M -> 12.0M tham so)
    # ------------------------------------------------------------------
    d_model: int = 256
    num_heads: int = 4  # head_dim = 256 / 4 = 64
    d_ff: int = 1024
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4

    dropout: float = 0.1  # dropout chung (residual, FFN, embedding)
    attention_dropout: float = 0.1  # dropout rieng tren attention weights
    drop_path_rate: float = 0.1  # stochastic depth - regularization manh cho du lieu it
    layer_norm_eps: float = 1e-6

    # Buoc chung khong gian giua token embedding va LM head.
    # Tiet kiem ~2.06M tham so (17% model) va thuong cai thien chat luong.
    tie_word_embeddings: bool = True

    # ------------------------------------------------------------------
    # Caption
    # ------------------------------------------------------------------
    max_caption_length: int = 40
    min_word_frequency: int = 2

    # ------------------------------------------------------------------
    # Augmentation  (nguon "du lieu them" duy nhat khi train from scratch)
    # ------------------------------------------------------------------
    random_resized_crop_scale: tuple[float, float] = (0.7, 1.0)
    color_jitter: float = 0.4
    rand_augment_num_ops: int = 2
    rand_augment_magnitude: int = 7
    random_erasing_prob: float = 0.25
    horizontal_flip_prob: float = 0.5
    normalize_mean: tuple[float, ...] = IMAGENET_MEAN
    normalize_std: tuple[float, ...] = IMAGENET_STD

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 32
    num_epochs: int = 60
    learning_rate: float = 3e-4  # dung KEM warmup ben duoi
    warmup_steps: int = 1000  # ~1 epoch. AdamW betas=(0.9,0.98) BAT BUOC phai co warmup.
    weight_decay: float = 0.05  # ap dung CHI cho tensor ndim >= 2, khong ap len norm/bias/embedding
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0  # da bo field trung `gradient_clip_norm`
    adam_betas: tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-9

    num_workers: int = 4
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    # ------------------------------------------------------------------
    # Logging / validation
    # ------------------------------------------------------------------
    log_interval: int = 20
    validate_every_n_epochs: int = 1
    early_stopping_patience: int = 8

    # Sinh caption cho dung nhung anh nay sau moi epoch.
    # Neu 3 caption giong het nhau => model dang bo qua anh, dung train ngay.
    sanity_check_image_count: int = 3

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} phai chia het cho num_heads={self.num_heads}."
            )

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} phai chia het cho patch_size={self.patch_size}."
            )

        if self.use_conv_stem:
            downsample = 2 ** len(self.conv_stem_channels)
            if downsample != self.patch_size:
                raise ValueError(
                    f"conv stem ha {downsample}x nhung patch_size={self.patch_size}. "
                    f"Can dung {int(self.patch_size).bit_length() - 1} lop stride-2 "
                    f"de so token khop voi positional embedding."
                )
            if self.conv_stem_channels[-1] > self.d_model:
                raise ValueError(
                    f"kenh cuoi cua conv stem ({self.conv_stem_channels[-1]}) "
                    f"khong nen lon hon d_model ({self.d_model})."
                )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout phai nam trong [0,1), nhan duoc {self.dropout}.")

    # ------------------------------------------------------------------
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.in_channels
