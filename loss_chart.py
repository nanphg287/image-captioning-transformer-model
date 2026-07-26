import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt


CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_FILE = Path("average_loss_60_epochs.pdf")


def get_epoch_number(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.stem)

    if match is None:
        return 0

    return int(match.group(1))


checkpoint_files = sorted(
    CHECKPOINT_DIR.glob("image_captioning_epoch_*.pt"),
    key=get_epoch_number
)

epochs = []
average_losses = []

for checkpoint_file in checkpoint_files:
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu"
    )

    epochs.append(int(checkpoint["epoch"]))
    average_losses.append(float(checkpoint["average_loss"]))

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    average_losses,
    color="blue",
    linewidth=2,
    marker="o",
    markersize=3,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Average Loss qua 60 Epoch")

plt.xticks(range(0, max(epochs) + 1, 5))
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    format="pdf",
    bbox_inches="tight"
)

plt.show()