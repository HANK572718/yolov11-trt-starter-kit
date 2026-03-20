"""
YOLOv11 完整流程測試：訓練 → ONNX → TensorRT → 推理
Windows 必須使用 if __name__ == '__main__': 保護多進程 DataLoader
"""
import sys
import glob
from pathlib import Path


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60, flush=True)


def main():
    import torch
    from ultralytics import YOLO

    # ── Step 0: 環境資訊 ──────────────────────────────────────────
    step("Step 0: 環境資訊")
    print(f"Python:      {sys.version.split()[0]}")
    print(f"Torch:       {torch.__version__}")
    print(f"CUDA 可用:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")

    # ── Step 1: 訓練 ─────────────────────────────────────────────
    step("Step 1: 訓練 yolo11n (coco128, 3 epochs)")
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="coco128.yaml",
        epochs=3,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,          # Windows 下避免多進程問題
        project="runs/test",
        name="pipeline",
        exist_ok=True,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"✓ 訓練完成 → {best_pt}", flush=True)

    # ── Step 2: Export ONNX ──────────────────────────────────────
    step("Step 2: Export ONNX")
    model2 = YOLO(str(best_pt))
    onnx_path = model2.export(format="onnx", imgsz=640, simplify=True)
    print(f"✓ ONNX 導出 → {onnx_path}", flush=True)

    # ── Step 3: Export TensorRT ──────────────────────────────────
    step("Step 3: Export TensorRT (FP16)")
    model3 = YOLO(str(best_pt))
    engine_path = model3.export(format="engine", imgsz=640, half=True, simplify=True)
    print(f"✓ TRT engine → {engine_path}", flush=True)

    # ── Step 4: TRT 推理 ─────────────────────────────────────────
    step("Step 4: TensorRT 推理")
    model_trt = YOLO(str(engine_path))

    # 找 coco128 的樣本圖片
    sample_imgs = glob.glob(str(Path.home() / "Documents/datasets/coco128/images/train2017/*.jpg"))
    if not sample_imgs:
        sample_imgs = glob.glob("datasets/coco128/images/train2017/*.jpg")

    if sample_imgs:
        img = sample_imgs[0]
        print(f"推理圖片: {img}")
        result = model_trt(img, verbose=False)
        boxes = result[0].boxes
        print(f"✓ 推理成功 — 偵測到 {len(boxes)} 個物件")
        if len(boxes) > 0:
            print(f"  類別: {[model_trt.names[int(c)] for c in boxes.cls[:5]]}")
    else:
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype="uint8")
        result = model_trt(dummy, verbose=False)
        print(f"✓ 推理成功（dummy 圖） — 偵測到 {len(result[0].boxes)} 個物件")

    step("全部完成！訓練 → ONNX → TensorRT → 推理 流程驗證通過 ✓")


if __name__ == "__main__":
    main()
