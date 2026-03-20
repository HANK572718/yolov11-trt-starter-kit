"""
TensorRT export + 推理測試（使用已訓練好的 best.pt）
"""
import sys
import glob
from pathlib import Path


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60, flush=True)


def main():
    import tensorrt as trt
    import torch
    from ultralytics import YOLO

    step("TensorRT 版本確認")
    print(f"TensorRT: {trt.__version__}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print(f"✓ TRT Builder 建立成功 (GPU: {torch.cuda.get_device_name(0)})")

    best_pt = Path("runs/detect/runs/test/pipeline/weights/best.pt")
    if not best_pt.exists():
        print(f"ERROR: 找不到 {best_pt}，請先執行 test_pipeline.py")
        return

    # ── Step 3: Export TensorRT ──────────────────────────────────
    step("Step 3: Export TensorRT (FP16)")
    model3 = YOLO(str(best_pt))
    engine_path = model3.export(format="engine", imgsz=640, half=True, simplify=True)
    print(f"✓ TRT engine → {engine_path}", flush=True)

    # ── Step 4: TRT 推理 ─────────────────────────────────────────
    step("Step 4: TensorRT 推理")
    model_trt = YOLO(str(engine_path))

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
        dummy = (255 * __import__('numpy').random.rand(640, 640, 3)).astype('uint8')
        result = model_trt(dummy, verbose=False)
        print(f"✓ 推理成功（dummy 圖） — 偵測到 {len(result[0].boxes)} 個物件")

    step("全部完成！TensorRT 流程驗證通過 ✓")


if __name__ == "__main__":
    main()
