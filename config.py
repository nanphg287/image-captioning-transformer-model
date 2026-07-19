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


@dataclass
class Config:
    # Dataset
    image_dir: str = IMAGE_DIR
    resize_image_dir: str = PROCESSED_DATA_DIR / "resize_image"
    caption_file: str = CAPTION_FILE
    image_to_captions_file: str = IMAGE_TO_CAPTIONS_FILE
    train_data_file: str = TRAIN_DATA_MAPPING
    validation_data_file: str = VALIDATION_DATA_FILE
    test_data_file: str = TEST_DATA_FILE

    # Image
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Transformer
    d_model: int = 512
    num_heads: int = 4
    d_ff: int = 2048
    embed_dim: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers = 4
    decoder_layers: int = 4
    encoder_heads: int = 4
    decoder_heads: int = 4
    mlp_dim: int = 1024
    mlp_ratio: float = 4.0
    attention_dropout: float = 0.1

    # Caption
    max_caption_length: int = 40
    min_word_frequency: int = 2
    vocabulary_file: str = VOCABULARY_FILE

    # Training
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm = 1.0
    num_epochs = 30
    eps: float = 1e-6
    gradient_clip_norm: float = 1.0
    num_workers: int = 4
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    checkpoint_path: str = "checkpoints/image_captioning_epoch_030.pt"
    log_interval: int = 20

