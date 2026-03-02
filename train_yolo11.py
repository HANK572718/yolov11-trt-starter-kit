"""
YOLOv11 Training Script for Object Detection.

Trains yolo11s and yolo11m models on the local YOLO-format dataset.

Usage:
    poetry run python train_yolo11.py
    poetry run python train_yolo11.py --model s       # train only small
    poetry run python train_yolo11.py --model m       # train only medium
    poetry run python train_yolo11.py --epochs 100
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def setup_wandb() -> bool:
    """Check if wandb is installed and logged in; disable silently if not.

    Reads the stored API key (no network call). If unavailable, sets
    WANDB_MODE=disabled so ultralytics skips the integration entirely.

    Returns:
        bool: True if wandb is active, False if disabled.
    """
    try:
        import wandb  # noqa: F401

        api_key = wandb.api.api_key  # None when not logged in
        if not api_key:
            raise RuntimeError("wandb not logged in")
        print("[wandb] available and logged in — logging enabled")
        return True
    except Exception:
        os.environ["WANDB_MODE"] = "disabled"
        print("[wandb] not available or not logged in — skipping")
        return False


DATASET_YAML = Path(__file__).parent / "dataset_yolo_format" / "dataset.yaml"

TRAIN_CONFIGS = {
    "s": {
        "model": "yolo11s.pt",
        "name": "yolo11s_custom",
    },
    "m": {
        "model": "yolo11m.pt",
        "name": "yolo11m_custom",
    },
}


def train(model_key: str, epochs: int, imgsz: int, batch: int, device: str) -> None:
    """Train a single YOLO model variant.

    Args:
        model_key: Model size key, either 's' (small) or 'm' (medium).
        epochs: Number of training epochs.
        imgsz: Input image size (square).
        batch: Batch size. Use -1 for AutoBatch.
        device: Compute device, e.g. '0', 'cpu', or '0,1'.

    Returns:
        None
    """
    cfg = TRAIN_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Training: {cfg['model']}  |  epochs={epochs}  |  imgsz={imgsz}  |  batch={batch}")
    print(f"{'='*60}\n")

    model = YOLO(cfg["model"])
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=cfg["name"],
        patience=10,          # early stopping patience
        save_period=10,       # save checkpoint every N epochs
        workers=4,
        cos_lr=True,          # cosine LR scheduler
        augment=True,
        cache=False,          # set True if RAM allows, speeds up training
        exist_ok=True,        # allow overwriting existing run folder
    )
    print(f"\nDone: {cfg['name']} — results saved to runs/detect/{cfg['name']}/\n")


def main() -> None:
    """Parse arguments and run training for selected model(s).

    Returns:
        None
    """
    setup_wandb()

    parser = argparse.ArgumentParser(description="Train YOLOv11 s/m on custom dataset")
    parser.add_argument(
        "--model",
        choices=["s", "m", "both"],
        default="both",
        help="Which model size to train: s, m, or both (default: both)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size; -1 = AutoBatch (default: -1)",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu' (default: 0)")
    args = parser.parse_args()

    targets = ["s", "m"] if args.model == "both" else [args.model]
    for key in targets:
        train(
            model_key=key,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    print("All training finished.")
    print("Best weights location:")
    for key in targets:
        name = TRAIN_CONFIGS[key]["name"]
        print(f"  runs/detect/{name}/weights/best.pt")


if __name__ == "__main__":
    main()
