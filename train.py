from __future__ import annotations

from json import JSONDecodeError
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config
from data.collate import ImageCaptionCollator
from data.image_caption_dataset import ImageCaptionDataset
from data.tokenizer import CaptionTokenizer
from data.vocabulary import Vocabulary
from models._0_image_captioning_half_transformer import ImageCaptioningHalfTransformer
from models._0_image_captioning_transformer import ImageCaptioningTransformer

NUM_WORKERS = 4
RANDOM_SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def create_or_load_vocabulary() -> Vocabulary:
    tokenizer = CaptionTokenizer()

    vocabulary_path = Path(Config.vocabulary_file)

    if vocabulary_path.exists():
        try:
            vocabulary = Vocabulary.load(vocabulary_path=vocabulary_path, tokenizer=tokenizer)
            print(f"Đã tải vocabulary: "f"{vocabulary_path}")
            return vocabulary

        except (JSONDecodeError, KeyError, TypeError, ValueError) as error:
            print(f"Vocabulary hiện tại không hợp lệ: {error}")
            print("Tiến hành tạo lại vocabulary.")

    vocabulary = Vocabulary(tokenizer=tokenizer)
    vocabulary.build_from_json(
        json_path=Config.train_data_file,
        min_frequency=Config.min_word_frequency
    )

    vocabulary.save(output_path=vocabulary_path)
    print(f"Đã tạo vocabulary mới: {vocabulary_path}")
    return vocabulary


def create_train_dataloader(device: torch.device, vocabulary: Vocabulary) -> DataLoader:
    train_dataset = ImageCaptionDataset(
        json_path=Config.train_data_file,
        vocabulary=vocabulary,
        max_caption_length=Config.max_caption_length
    )

    collator = ImageCaptionCollator(pad_token_id=vocabulary.pad_token_id)

    return DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
        collate_fn=collator
    )


def train_one_epoch(
        model: ImageCaptioningTransformer,
        train_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: AdamW,
        device: torch.device,
        pad_token_id: int,
        epoch: int
) -> float:
    model.train()

    total_loss = 0.0
    total_valid_tokens = 0

    for batch_index, batch in enumerate(train_dataloader, start=1):
        images = batch["images"].to(device=device, non_blocking=True)
        caption_ids = batch["caption_ids"].to(device=device, non_blocking=True)
        caption_padding_mask = (batch["caption_padding_mask"].to(device=device, non_blocking=True))

        # Teacher forcing.
        decoder_input_ids = caption_ids[:, :-1]
        decoder_target_ids = caption_ids[:, 1:]

        decoder_padding_mask = caption_padding_mask[:, :-1]

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            images=images,
            decoder_input_ids=decoder_input_ids,
            decoder_padding_mask=decoder_padding_mask
        )

        vocabulary_size = logits.shape[-1]

        loss = criterion(logits.reshape(-1, vocabulary_size), decoder_target_ids.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=Config.max_grad_norm)
        optimizer.step()

        valid_token_count = (decoder_target_ids != pad_token_id).sum().item()
        total_loss += (loss.item() * valid_token_count)
        total_valid_tokens += valid_token_count

        if batch_index == 1 or batch_index % Config.log_interval == 0 or batch_index == len(train_dataloader):
            average_loss = (total_loss / total_valid_tokens)
            print(
                f"Epoch {epoch:02d} | "
                f"Batch {batch_index:04d}/"
                f"{len(train_dataloader):04d} | "
                f"Loss {loss.item():.4f} | "
                f"Average Loss {average_loss:.4f}"
            )

    return total_loss / total_valid_tokens


def save_checkpoint(
        model: ImageCaptioningTransformer,
        optimizer: AdamW,
        epoch: int,
        average_loss: float,
        vocabulary_size: int
) -> Path:
    checkpoint_directory = Path(Config.checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (checkpoint_directory / f"image_captioning_epoch_{epoch:03d}.pt")
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "average_loss": average_loss,
        "vocabulary_size": vocabulary_size
    }

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def create_validation_dataloader(
        device: torch.device,
        vocabulary: Vocabulary
) -> DataLoader:
    validation_dataset = ImageCaptionDataset(
        json_path=Config.validation_data_file,
        vocabulary=vocabulary,
        max_caption_length=Config.max_caption_length
    )

    collator = ImageCaptionCollator(pad_token_id=vocabulary.pad_token_id)

    return DataLoader(
        dataset=validation_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
        collate_fn=collator
    )


@torch.inference_mode()
def validate_one_epoch(
        model: ImageCaptioningTransformer,
        validation_dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        pad_token_id: int
) -> float:
    model.eval()

    total_loss = 0.0
    total_valid_tokens = 0

    for batch in validation_dataloader:
        images = batch["images"].to(device=device, non_blocking=True)

        caption_ids = batch["caption_ids"].to(device=device, non_blocking=True)

        caption_padding_mask = batch[
            "caption_padding_mask"
        ].to(
            device=device,
            non_blocking=True
        )

        decoder_input_ids = caption_ids[:, :-1]
        decoder_target_ids = caption_ids[:, 1:]
        decoder_padding_mask = caption_padding_mask[:, :-1]

        logits = model(images=images, decoder_input_ids=decoder_input_ids, decoder_padding_mask=decoder_padding_mask)

        vocabulary_size = logits.shape[-1]

        loss = criterion(logits.reshape(-1, vocabulary_size), decoder_target_ids.reshape(-1))

        valid_token_count = (decoder_target_ids != pad_token_id).sum().item()

        total_loss += loss.item() * valid_token_count
        total_valid_tokens += valid_token_count

    if total_valid_tokens == 0:
        raise RuntimeError(
            "Validation dataset không có token hợp lệ."
        )

    return total_loss / total_valid_tokens


def main() -> None:
    # --------------------------------------------------
    # Random seed.
    # --------------------------------------------------
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # --------------------------------------------------
    # Dataset và device.
    # --------------------------------------------------
    device = get_device()

    print(f"Device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # Vocabulary và DataLoader.
    # --------------------------------------------------
    vocabulary = create_or_load_vocabulary()
    vocabulary_size = len(vocabulary)
    print(f"Vocabulary size: {vocabulary_size}")
    train_dataloader = create_train_dataloader(device=device, vocabulary=vocabulary)
    print(f"Số batch huấn luyện: {len(train_dataloader)}")

    # --------------------------------------------------
    # Model.
    # --------------------------------------------------
    model = ImageCaptioningTransformer(
        image_size=Config.image_size,
        patch_size=Config.patch_size,
        vocabulary_size=vocabulary_size,
        max_caption_length=(
            Config.max_caption_length
        ),
        pad_token_id=vocabulary.pad_token_id,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        d_ff=Config.d_ff,
        num_decoder_layers=Config.num_decoder_layers,
        dropout=0.1,
        layer_norm_eps=1e-6
    ).to(device)

    # --------------------------------------------------
    # Loss.
    # --------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_token_id)

    # --------------------------------------------------
    # Optimizer.
    # --------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=Config.weight_decay
    )

    # --------------------------------------------------
    # Train epochs.
    # --------------------------------------------------
    print("\n===== BẮT ĐẦU HUẤN LUYỆN =====")

    validation_dataloader = create_validation_dataloader(
        device=device,
        vocabulary=vocabulary
    )

    best_validation_loss = float("inf")

    for epoch in range(1, Config.num_epochs + 1):
        average_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pad_token_id=vocabulary.pad_token_id,
            epoch=epoch
        )

        validation_loss = validate_one_epoch(
            model=model,
            validation_dataloader=validation_dataloader,
            criterion=criterion,
            device=device,
            pad_token_id=vocabulary.pad_token_id
        )

        checkpoint_path = ""
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                average_loss=validation_loss,
                vocabulary_size=len(vocabulary)
            )

        print(f"Epoch {epoch:02d} hoàn thành | Average Loss: {average_loss:.4f} | Validation Loss: {validation_loss:.4f}")
        print(f"Checkpoint: {checkpoint_path}")

    print("\n===== HUẤN LUYỆN HOÀN THÀNH =====")


if __name__ == "__main__":
    main()
