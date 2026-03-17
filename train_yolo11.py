"""
YOLOv11 Training Script — supports ISAT and YOLO-format datasets.

Usage:
    # ISAT annotations (auto-detect format, segment task)
    poetry run python train_yolo11.py --source C:\\my_data\\images --model s

    # ISAT annotations, detect task
    poetry run python train_yolo11.py --source C:\\my_data\\images --task detect --model s

    # Existing YOLO-format dataset
    poetry run python train_yolo11.py --source dataset_yolo_format --format yolo --task detect

    # Disable wandb
    poetry run python train_yolo11.py --source C:\\my_data\\images --no-wandb
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import psutil

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# System resource probe → auto workers
# ---------------------------------------------------------------------------

def resolve_workers() -> int:
    """Probe available RAM + paging file and return a safe DataLoader workers count.

    Windows spawns one subprocess per worker; each imports cv2/torch, which is
    heavy on virtual memory. When the system is already loaded the paging file
    can be exhausted, causing a DLL load failure. This function caps workers
    conservatively based on current free virtual memory.

    Thresholds (free RAM + free swap):
        < 4 GB  → 0  (in-process, safest)
        < 8 GB  → 2
        ≥ 8 GB  → 4  (also capped at cpu_count // 2)
    """
    vm   = psutil.virtual_memory()
    swap = psutil.swap_memory()
    free_gb = (vm.available + swap.free) / (1024 ** 3)
    cpu_cap = max(1, (os.cpu_count() or 4) // 2)

    if free_gb < 4:
        workers = 0
    elif free_gb < 8:
        workers = 2
    else:
        workers = min(4, cpu_cap)

    print(
        f"[INFO] System probe — free virtual memory: {free_gb:.1f} GB "
        f"(RAM {vm.available / (1024**3):.1f} GB + swap {swap.free / (1024**3):.1f} GB) "
        f"→ workers={workers}"
    )
    return workers


# ---------------------------------------------------------------------------
# Model config tables
# ---------------------------------------------------------------------------

DETECT_CONFIGS = {
    "n": {"model": "yolo11n.pt",     "name": "yolo11n_custom"},
    "s": {"model": "yolo11s.pt",     "name": "yolo11s_custom"},
    "m": {"model": "yolo11m.pt",     "name": "yolo11m_custom"},
    "l": {"model": "yolo11l.pt",     "name": "yolo11l_custom"},
    "x": {"model": "yolo11x.pt",     "name": "yolo11x_custom"},
}

SEGMENT_CONFIGS = {
    "n": {"model": "yolo11n-seg.pt", "name": "yolo11n_seg_custom"},
    "s": {"model": "yolo11s-seg.pt", "name": "yolo11s_seg_custom"},
    "m": {"model": "yolo11m-seg.pt", "name": "yolo11m_seg_custom"},
    "l": {"model": "yolo11l-seg.pt", "name": "yolo11l_seg_custom"},
    "x": {"model": "yolo11x-seg.pt", "name": "yolo11x_seg_custom"},
}

ALL_SIZES = ["n", "s", "m", "l", "x"]


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

def detect_format(source: Path) -> str:
    """Return 'isat' or 'yolo', or raise SystemExit on ambiguity."""
    json_files = list(source.glob("*.json"))
    if json_files:
        try:
            with open(json_files[0], encoding="utf-8") as f:
                data = json.load(f)
            if data.get("info", {}).get("description") == "ISAT":
                return "isat"
        except Exception:
            pass

    if (source / "dataset.yaml").exists():
        return "yolo"

    print(
        f"[ERROR] Cannot determine dataset format in: {source}\n"
        "  Expected either:\n"
        "    • *.json files with ISAT format  (--format isat)\n"
        "    • dataset.yaml                   (--format yolo)"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# ISAT → YOLO conversion
# ---------------------------------------------------------------------------

def collect_classes(json_files: list[Path]) -> dict[str, int]:
    """Scan all JSONs and build a sorted category → class_id mapping.

    __background__ is included if present but gets a numeric id like any other
    class; callers decide whether to skip it when writing labels.
    """
    categories: set[str] = set()
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        for obj in data.get("objects", []):
            cat = obj.get("category", "").strip()
            if cat:
                categories.add(cat)

    if not categories:
        print(
            "[ERROR] No annotated objects found across all JSON files.\n"
            "  Make sure each object has a 'category' field set (not empty)."
        )
        sys.exit(1)

    # Check that there are usable classes beyond __background__
    real_classes = categories - {"__background__"}
    if not real_classes:
        print(
            "[WARNING] All objects have category '__background__'.\n"
            "  No trainable classes found. Please assign proper category names in ISAT."
        )
        sys.exit(1)

    # Sort alphabetically; __background__ excluded from training classes
    sorted_cats = sorted(real_classes)
    return {cat: idx for idx, cat in enumerate(sorted_cats)}


def isat_json_to_label(
    json_path: Path,
    class_map: dict[str, int],
    task: str,
) -> list[str]:
    """Convert one ISAT JSON to YOLO label lines."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    info = data["info"]
    img_w: float = info["width"]
    img_h: float = info["height"]
    lines: list[str] = []

    for obj in data.get("objects", []):
        cat = obj.get("category", "").strip()
        if cat not in class_map:
            continue  # skip __background__ and unknown

        class_id = class_map[cat]

        if task == "segment":
            seg = obj.get("segmentation", [])
            if len(seg) < 3:
                continue
            coords = []
            for point in seg:
                x_norm = max(0.0, min(1.0, point[0] / img_w))
                y_norm = max(0.0, min(1.0, point[1] / img_h))
                coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
            lines.append(f"{class_id} " + " ".join(coords))
        else:  # detect
            bbox = obj.get("bbox")  # [x_min, y_min, x_max, y_max]
            if bbox is None:
                # Fall back: compute from segmentation
                seg = obj.get("segmentation", [])
                if not seg:
                    continue
                xs = [p[0] for p in seg]
                ys = [p[1] for p in seg]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) / 2 / img_w
            cy = (y_min + y_max) / 2 / img_h
            w  = (x_max - x_min) / img_w
            h  = (y_max - y_min) / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def convert_isat(source: Path, task: str, val_split: float) -> Path:
    """Convert ISAT dataset to YOLO format. Returns output dataset root."""
    json_files = sorted(source.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No JSON files found in {source}")
        sys.exit(1)

    print(f"[INFO] Found {len(json_files)} ISAT JSON files in {source}")

    class_map = collect_classes(json_files)
    print(f"[INFO] Detected classes: {list(class_map.keys())}")

    # Only keep JSONs that have a matching image
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    valid_pairs: list[tuple[Path, Path]] = []
    for jf in json_files:
        for ext in image_exts:
            img = jf.with_suffix(ext)
            if img.exists():
                valid_pairs.append((img, jf))
                break

    if not valid_pairs:
        print(
            f"[ERROR] No image+JSON pairs found in {source}\n"
            "  Make sure each *.json has a matching image file."
        )
        sys.exit(1)

    print(f"[INFO] Valid image+JSON pairs: {len(valid_pairs)}")

    # Train/val split
    random.seed(42)
    random.shuffle(valid_pairs)
    n_val = max(1, int(len(valid_pairs) * val_split))
    val_pairs   = valid_pairs[:n_val]
    train_pairs = valid_pairs[n_val:]
    print(f"[INFO] Split: {len(train_pairs)} train / {len(val_pairs)} val")

    # Output directory
    out_dir = source.parent / (source.name + "_yolo_converted")
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_split(pairs: list[tuple[Path, Path]], split: str) -> None:
        for img_path, json_path in pairs:
            # Copy image
            dst_img = out_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # Write label
            label_lines = isat_json_to_label(json_path, class_map, task)
            label_file = out_dir / "labels" / split / (img_path.stem + ".txt")
            label_file.write_text("\n".join(label_lines), encoding="utf-8")

    write_split(train_pairs, "train")
    write_split(val_pairs, "val")

    # Write dataset.yaml
    names_block = "\n".join(f"    {v}: {k}" for k, v in class_map.items())
    yaml_content = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test:\n\n"
        f"names:\n{names_block}\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"[INFO] Converted dataset written to: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_yaml: Path,
    model_key: str,
    task: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    use_wandb: bool,
) -> None:
    configs = SEGMENT_CONFIGS if task == "segment" else DETECT_CONFIGS
    cfg = configs[model_key]

    print(f"\n{'='*60}")
    print(f"Training : {cfg['model']}")
    print(f"Task     : {task}")
    print(f"Epochs   : {epochs}  |  imgsz={imgsz}  |  batch={batch}")
    print(f"Dataset  : {dataset_yaml}")
    print(f"{'='*60}\n")

    train_kwargs: dict = dict(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=cfg["name"],
        patience=10,
        save_period=10,
        workers=resolve_workers(),
        cos_lr=True,
        augment=True,
        cache=False,
        exist_ok=True,
    )

    if use_wandb:
        train_kwargs["project"] = "yolo11-training"

    model = YOLO(cfg["model"])
    model.train(**train_kwargs)

    task_dir = "segment" if task == "segment" else "detect"
    print(f"\nDone: {cfg['name']} — results saved to runs/{task_dir}/{cfg['name']}/\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run training for selected model(s).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 on ISAT or YOLO-format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset_yolo_format",
        help="Dataset directory path (default: dataset_yolo_format)",
    )
    parser.add_argument(
        "--format",
        choices=["isat", "yolo", "auto"],
        default="auto",
        help="Dataset format: isat | yolo | auto (default: auto)",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "segment"],
        default="segment",
        help="Training task: detect | segment (default: segment)",
    )
    parser.add_argument(
        "--model",
        choices=["n", "s", "m", "l", "x", "all"],
        default="s",
        help="Model size: n, s, m, l, x, or all (default: s)",
    )
    parser.add_argument("--epochs",    type=int,   default=100,  help="Max training epochs (default: 100)")
    parser.add_argument("--imgsz",     type=int,   default=640,  help="Input image size (default: 640)")
    parser.add_argument("--batch",     type=int,   default=-1,   help="Batch size; -1=AutoBatch (default: -1)")
    parser.add_argument("--device",    type=str,   default="0",  help="CUDA device id or 'cpu' (default: 0)")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation fraction for ISAT mode (default: 0.2)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb experiment tracking",
    )
    args = parser.parse_args()

    # --- wandb setup ---
    use_wandb = not args.no_wandb
    if use_wandb:
        from ultralytics import settings as ult_settings
        if not ult_settings.get("wandb"):
            ult_settings.update({"wandb": True})
            print("[INFO] ultralytics settings: wandb 已自動啟用")
        print("[INFO] wandb 追蹤已啟用，若未登入請執行 wandb login")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        from ultralytics import settings as ult_settings
        ult_settings.update({"wandb": False})
        print("[INFO] wandb 已停用 (--no-wandb)")

    # --- Resolve source path ---
    source = Path(args.source)
    if not source.is_absolute():
        source = Path(__file__).parent / source
    source = source.resolve()

    if not source.exists():
        print(f"[ERROR] Source path does not exist: {source}")
        sys.exit(1)

    # --- Format detection ---
    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(source)
        print(f"[INFO] Auto-detected format: {fmt}")

    # --- ISAT conversion ---
    if fmt == "isat":
        dataset_root = convert_isat(source, args.task, args.val_split)
    else:
        dataset_root = source

    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"[ERROR] dataset.yaml not found at: {dataset_yaml}")
        sys.exit(1)

    # --- Run training ---
    targets = ALL_SIZES if args.model == "all" else [args.model]
    for key in targets:
        train(
            dataset_yaml=dataset_yaml,
            model_key=key,
            task=args.task,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            use_wandb=use_wandb,
        )

    print("\nAll training finished.")
    configs = SEGMENT_CONFIGS if args.task == "segment" else DETECT_CONFIGS
    task_dir = "segment" if args.task == "segment" else "detect"
    print("Best weights:")
    for key in targets:
        name = configs[key]["name"]
        print(f"  runs/{task_dir}/{name}/weights/best.pt")


if __name__ == "__main__":
    main()
