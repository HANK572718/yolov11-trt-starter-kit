# YOLOv11 TensorRT Starter Kit

YOLOv11n 從 PyTorch 到 TensorRT FP16 的完整最佳化流程示範，目標部署平台為 Windows NVIDIA GPU 與 Jetson Orin Nano。

---

## 支援模型類型

> 三種 COCO 預訓練模型，開箱即用，ultralytics 自動下載。

| 類型 | 模型檔 | 大小 | 用途 | Script Phase |
|------|--------|------|------|-------------|
| 多類別物件偵測 | `yolo11n.pt` | ~5.6 MB | COCO 80 類全類別偵測 | Phase 2-A |
| 人物偵測 | `yolo11n.pt` (classes=[0]) | — | 同上模型，過濾 person 類 | Phase 2-A-Person |
| 實例分割 | `yolo11n-seg.pt` | ~6.7 MB | COCO 80 類 + pixel mask | Phase 2-E |

- 多類別偵測與人物偵測共用同一個 `.pt`，不需額外下載
- 三種任務均可匯出 ONNX（跨平台）與 TensorRT `.engine`（需本機 build）

---

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `yolov11_tensorrt_jetson.py` | 主腳本：PT → ONNX → TRT FP16 三段推理流程 |
| `tensorrt_postprocess_example.py` | TRT 原始輸出後處理參考（手動 NMS） |
| `nvidia_deployment_overview.md` | NVIDIA 部署生態系說明（TRT / Triton / DeepStream） |
| `hank_laptop_env.md` | 開發者本機環境規格與 Benchmark 結果 |
| `pyproject.toml` | Poetry 依賴設定 |
| `poetry.lock` | 鎖定版本的依賴清單 |

---

## 快速開始（Windows + CUDA）

```bash
# 前提：Python 3.11.9、Poetry 2.x、NVIDIA Driver 560+、CUDA 12.4

# 1. 指定 Python 版本建立 venv
poetry env use "C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe"

# 2. 安裝依賴
poetry install

# 3. 降版 TensorRT（重要：poetry install 預設裝 10.15，需要 10.6）
poetry run pip install tensorrt==10.6.0.post2
poetry run pip install tensorrt-cu12==10.6.0.post2 \
                       tensorrt-cu12-libs==10.6.0.post2 \
                       tensorrt-cu12-bindings==10.6.0.post2

# 4. 驗證 TRT 可用
poetry run python -c "import tensorrt as trt; b=trt.Builder(trt.Logger()); print('TRT OK:', trt.__version__)"

# 5. 執行主腳本
poetry run python yolov11_tensorrt_jetson.py
```

---

## Benchmark（開發機：RTX 4060 Laptop，yolo11n，bus.jpg，640x640）

| 格式 | Inf(ms) | FPS | 加速倍率 |
|------|---------|-----|---------|
| PyTorch FP32 | 7.4 | 97.5 | 1.00x |
| ONNX Runtime (CUDA EP) | 5.6 | 105.5 | 1.08x |
| TensorRT FP16 | **3.5** | **108.5** | **1.11x** |

> RTX 4060 Laptop 基礎效能已高，TRT 加速比不明顯。
> Jetson Orin Nano（baseline 15~25 FPS）預估 TRT FP16 可達 60~100 FPS（4~6x）。

---

## pyproject.toml 平台相容性說明

> **重要：本 `pyproject.toml` 為 Windows x86_64 + CUDA 12.4 環境的全量依賴快照，移植到 Linux 或 Jetson 前需要手動修改。**

### Windows 專屬套件（Linux / Jetson 上需移除或替換）

| 套件 | 問題 | Linux 替代方案 |
|------|------|---------------|
| `torch / torchvision / torchaudio` | URL 指向 `win_amd64.whl` | 改用 `linux_x86_64.whl` URL；Jetson 用 JetPack 內建 |
| `PyQt5 / PyQt5-Qt5` | URL 指向 `win_amd64.whl` | `PyQt5 = "*"` 即可（PyPI 有 Linux wheel） |
| `detectron2` | 第三方預編譯 `win_amd64.whl` | Linux 需從原始碼 build |
| `wmi` | **Windows Only**（WMI 系統管理介面） | 直接移除 |
| `imagingcontrol4` | The Imaging Source 相機 SDK，僅 Windows wheel | 移除或換官方 Linux SDK |
| `pyueye` | IDS uEye SDK，需先裝驅動 | 移除或換官方 Linux driver |
| `ids-peak / ids-peak-ipl / ids-peak-afl` | IDS 工業相機，Linux 版需官方 SDK | 移除或依 IDS Linux 安裝文件 |

### TensorRT 跨平台注意事項

| 平台 | TRT 安裝方式 |
|------|-------------|
| Windows（Driver 560+，CUDA 12.4） | `pip install tensorrt==10.6.0.post2`（見上方快速開始） |
| x86_64 Linux（Ubuntu 22.04 + CUDA 12.x） | `pip install tensorrt`（PyPI 有 Linux wheel） |
| Jetson Orin Nano（aarch64，JetPack 6.x） | **不要 pip 裝**，JetPack 已預裝 TRT，直接 `import tensorrt` |

### OpenVINO

`openvino` 套件目前 PyPI **不支援 aarch64（Jetson）**，在 Jetson 上需移除此依賴。

---

## Jetson Orin Nano 部署注意事項

- TRT `.engine` 檔案**不可跨平台複製**，必須在 Jetson 上本機重新 export
- ONNX 檔案可以跨平台使用（從 Windows 複製到 Jetson 直接 inference 沒問題）
- 建議先用 ONNX Runtime (CUDA EP) 驗證精度，再進行 TRT export
- 詳見 `nvidia_deployment_overview.md`

---

## YOLOv8 / YOLO11 切換與預訓練模型說明

### 命名規則與切換方式

兩個版本檔名格式**不同**，不是單純替換字串：

| 版本 | 偵測 | 分割 | 姿態 |
|------|------|------|------|
| YOLO**v8** | `yolov8n.pt` | `yolov8n-seg.pt` | `yolov8n-pose.pt` |
| YOLO**11** | `yolo11n.pt` | `yolo11n-seg.pt` | `yolo11n-pose.pt` |

> YOLO11 檔名無 `v`；YOLOv8 有 `v`。切換時只改腳本頭部 6 個常數即可，其餘程式碼完全不動。

```python
# 切換至 YOLOv8
MODEL_PT         = "yolov8n.pt"
MODEL_ONNX       = "yolov8n.onnx"
MODEL_ENGINE     = "yolov8n.engine"
MODEL_SEG_PT     = "yolov8n-seg.pt"
MODEL_SEG_ONNX   = "yolov8n-seg.onnx"
MODEL_SEG_ENGINE = "yolov8n-seg.engine"
```

### Ultralytics 版本相容性

`ultralytics==8.4.7` 同時支援 v8 與 v11，**不需升降版**：

```python
from ultralytics import YOLO
YOLO("yolov8n.pt")   # v8
YOLO("yolo11n.pt")   # v11，同一個 class，同一套 API
```

### 為什麼建議維持 YOLO11？

| 指標 | YOLOv8n | YOLO11n |
|------|---------|---------|
| 參數量 | 3.2M | 2.6M（少 19%） |
| COCO mAP | 37.3 | 39.5（高 2.2） |
| 模型大小 | ~6.3 MB | ~5.6 MB |

除非需要與舊系統相容，否則維持 YOLO11。

### 預訓練模型下載說明

#### 方式 A：Ultralytics 自動下載（推薦）

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")   # 第一次執行自動下載，後續從快取載入
```

快取位置：
- Windows：`C:\Users\<user>\.config\Ultralytics\`
- Linux / Jetson：`~/.config/Ultralytics/`

#### 方式 B：程式內批次下載（離線環境預先準備）

```python
from ultralytics.utils.downloads import attempt_download_asset

for name in ["yolo11n.pt", "yolo11n-seg.pt"]:
    attempt_download_asset(name)
```

#### 方式 C：GitHub 手動下載

下載頁面：`https://github.com/ultralytics/assets/releases`

**物件偵測（Detection）**

| 模型 | 大小 | mAP | 速度(T4 ms) |
|------|------|-----|------------|
| `yolo11n.pt` | 5.6 MB | 39.5 | 1.5 |
| `yolo11s.pt` | 19.8 MB | 47.0 | 2.5 |
| `yolo11m.pt` | 67.6 MB | 51.5 | 5.0 |
| `yolo11l.pt` | 87.0 MB | 53.4 | 6.2 |
| `yolo11x.pt` | 136.7 MB | 54.7 | 11.3 |

**實例分割（Segmentation）**

| 模型 | 大小 | Box mAP | Mask mAP |
|------|------|---------|---------|
| `yolo11n-seg.pt` | 6.7 MB | 38.9 | 32.0 |
| `yolo11s-seg.pt` | 21.4 MB | 46.6 | 38.8 |
| `yolo11m-seg.pt` | 70.2 MB | 51.5 | 42.0 |
| `yolo11l-seg.pt` | 90.4 MB | 53.4 | 43.2 |
| `yolo11x-seg.pt` | 140.7 MB | 54.7 | 44.0 |

### Jetson 離線環境的建議流程

```
Windows（有網路）              Jetson（無網路）
─────────────────              ────────────────
1. 自動下載 .pt                scp .pt 到 Jetson
2. 匯出 .onnx                 scp .onnx 到 Jetson（可選）
3. 不要複製 .engine            在 Jetson 本機 build .engine
```

```bash
scp yolo11n.pt user@jetson-ip:/home/user/
scp yolo11n-seg.pt user@jetson-ip:/home/user/
scp yolo11n.onnx user@jetson-ip:/home/user/
```

### 模型選擇建議

| 需求 | 建議模型 |
|------|---------|
| 一般多類別偵測 | `yolo11n.pt` |
| 人物偵測 | `yolo11n.pt` + `classes=[0]`（不需額外下載） |
| 實例分割 | `yolo11n-seg.pt` |
| Jetson / 嵌入式優先速度 | `yolo11n` 系列（nano） |
| 需要較高精度 | `yolo11s` 或 `yolo11m` |

### 後處理程式碼相容性

v8 與 v11 輸出 tensor 格式完全相同，`tensorrt_postprocess_example.py` **無需修改**直接適用：

```
Detection    輸出: [1, 84, 8400]                          # 4(bbox) + 80(cls)
Segmentation 輸出: [1, 116, 8400] + [1, 32, 160, 160]    # +32(mask_coeff) + prototype
```

---

## 開發環境

詳見 `hank_laptop_env.md`。

- OS: Windows 11 Build 26100
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU（8GB VRAM，Ada Lovelace SM 8.9）
- Driver: 566.14 / CUDA: 12.4 / TensorRT: 10.6.0.post2
- Python: 3.11.9 / PyTorch: 2.4.1+cu124 / Ultralytics: 8.4.7
