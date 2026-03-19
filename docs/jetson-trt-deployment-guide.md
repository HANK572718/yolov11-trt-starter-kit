# YOLOv11 TensorRT 部署調查報告

> 調查日期：2026-03-18 | 環境：Windows + RTX 5070 Ti → 目標 Jetson Orin

---

## 一、Jetson Orin 支援版本矩陣

| JetPack | Ubuntu | Python | CUDA | TensorRT | 備註 |
|---|---|---|---|---|---|
| 5.1.2 | 20.04 | **3.8** | 11.4 | 8.5.2 | 舊裝置 |
| 6.0 GA | 22.04 | **3.10** | 12.2 | 8.6.2 | |
| **6.1 GA** | 22.04 | **3.10** | 12.6 | **10.3.0** | 推薦 |
| 7.0 | 24.04 | 3.12 | 13.0 | 10.x | 僅 Jetson Thor |

**結論：Jetson Orin 最佳選擇為 JetPack 6.1 + Python 3.10。Python 3.11 無官方支援。**

---

## 二、Windows 開發環境（已驗證）

### 套件版本

| 套件 | 版本 |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.10.0+cu128 |
| TorchVision | 0.25.0+cu128 |
| TensorRT | **10.15.1.29 (cu12)** |
| Ultralytics | 8.4.23 |
| ONNX | 1.16.1 |

### 關鍵坑：tensorrt-cu13 vs cu12

`pip install tensorrt` 預設安裝 `tensorrt-cu13`，**與 CUDA 12.8 (cu128) 不相容**，
導致 `CUDA initialization failure with error: 35`。

```toml
# pyproject.toml — 正確寫法
tensorrt-cu12 = "^10.15.1.29"  # 明確指定 cu12
```

---

## 三、.engine 跨裝置限制（最重要）

```
x86 Windows（訓練）        Jetson Orin（部署）
  yolo11n.pt     ──scp──►  yolo11n.pt
                              │
                              ▼ 必須在 Jetson 上重新 export
                           yolo11n.engine  ← 綁定裝置 GPU / TRT 版本
```

- `.pt` 可跨平台移植
- `.engine` **不可**從 x86 移植到 Jetson，也不可在不同 Jetson 之間共用

---

## 四、Jetson Orin 部署 Wheel 路徑（JetPack 6.1）

```bash
# PyTorch (Ultralytics 提供的 aarch64 預編譯版)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
pip install ultralytics[export]
```

詳見 `pyproject.toml` 的 `[tool.poetry.group.jetson.dependencies]` 區塊（已備註）。

---

## 五、Jetson 上的完整 Export + 推理流程

```python
# 在 Jetson 上執行（workers=0 for Windows，Jetson 可使用預設值）
from ultralytics import YOLO

# Export（FP16 推薦，speed/accuracy 最佳平衡）
model = YOLO("yolo11n.pt")
model.export(format="engine", half=True, imgsz=640)

# 推理
model_trt = YOLO("yolo11n.engine")
results = model_trt("image.jpg")
```

### 系統調優（必做）

```bash
sudo nvpmodel -m 0    # 最大功耗模式
sudo jetson_clocks    # 鎖定最高時脈
sudo pip install jetson-stats && jtop  # 監控工具
```

---

## 六、效能基準（Jetson AGX Orin 64GB，YOLOv11）

| 模型 | FP32 | FP16 | 加速倍率 |
|---|---|---|---|
| YOLO11n | 3.93 ms | 2.55 ms | ~1.5x |
| YOLO11x | 28.50 ms | 13.55 ms | ~2.1x |

TensorRT FP16 相比 PyTorch 整體可達 **5–15x** 加速。

---

## 七、Windows 開發注意事項

### DataLoader 多進程問題

Windows 使用 `spawn` 模式，訓練腳本必須加保護：

```python
# 必須有此保護，否則 RuntimeError: An attempt has been made to start a new process...
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(data="coco128.yaml", workers=0)  # 或使用 workers=0
```

---

## 八、替代部署方案（不用 PyTorch）

若 Jetson 不想安裝 PyTorch，可用純 TRT 方案：

```bash
# Step 1：在 x86 電腦 export ONNX（需 PyTorch）
yolo export model=yolo11n.pt format=onnx

# Step 2：複製 .onnx 到 Jetson，用 JetPack 內建工具轉換
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo11n.onnx \
  --saveEngine=yolo11n.engine \
  --fp16

# Step 3：用 TensorRT Python/C++ API 自行撰寫推論邏輯
```

> 參考：[TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)（支援 YOLOv11）

---

## 九、驗證測試結果（本機 RTX 5070 Ti）

| 步驟 | 結果 |
|---|---|
| 訓練 3 epochs (coco128) | ✓ mAP50=0.692 |
| Export ONNX | ✓ 10.2 MB |
| Export TRT FP16 engine | ✓ 7.3 MB（建立耗時 31s） |
| TRT 推理 | ✓ 偵測 6 個物件（bowl, broccoli...） |

測試腳本：`test_pipeline.py`（完整流程）、`test_trt_only.py`（純 TRT 測試）
