# NVIDIA 模型部署生態系統概覽
## 從 YOLOv11 學習推理引擎與最佳化框架

---

## 一、核心問題：為什麼部署比訓練複雜？

深度學習模型從訓練到實際部署，面對的根本問題是**硬體差異**：

- 不同廠商的 GPU / NPU 有完全不同的指令集與記憶體架構
- 沒有任何一種推理格式能在所有硬體上同時達到最佳效能
- NVIDIA / Intel / Apple / ARM 各自開發了針對自家硬體的推理引擎

**核心矛盾：可移植性（Portability） vs. 效能（Performance）**

```
高可移植性                                              高效能
    |                                                      |
PyTorch .pt  ------>  ONNX  ------>  TensorRT .engine
(任何機器)           (跨平台)        (綁定 NVIDIA 特定硬體)
```

這個矛盾沒有「完美解」，只有針對你的部署目標選擇最合適的方案。

---

## 二、推理格式全景圖

### 格式轉換路徑

```
訓練格式
.pt (PyTorch)
    |
    +---[model.export()]---> .onnx              通用中間格式
                                |
                                +---> TensorRT .engine     NVIDIA GPU 專屬
                                |       (最快，但只能在 build 的那台機器跑)
                                |
                                +---> ONNX Runtime          跨平台 CPU/GPU
                                |       (次快，可直接複製到任意機器)
                                |
                                +---> OpenVINO .xml/.bin    Intel CPU/GPU 專屬
                                |
                                +---> CoreML .mlmodel       Apple Silicon 專屬
                                |
                                +---> TFLite .tflite        手機 / 嵌入式
```

### 各推理引擎比較

| 推理引擎 | 開發商 | 目標硬體 | 可移植性 | 效能增益 | 安裝難度 |
|---------|--------|---------|---------|---------|---------|
| PyTorch（原生）| Meta | 任何 | 最高 | 基準線 | 低 |
| ONNX Runtime | Microsoft | CPU/NVIDIA/AMD/ARM | 高 | +10~30% | 低 |
| OpenVINO | Intel | Intel CPU/iGPU/VPU | 中（Intel 生態）| 高（Intel 上）| 中 |
| TensorRT | NVIDIA | NVIDIA GPU | 低（NVIDIA 專屬）| +50~300% | 高 |
| CoreML | Apple | Apple Silicon | 低（Apple 專屬）| 高（Apple 上）| 低（macOS）|
| TFLite | Google | 手機 / 嵌入式 | 中 | 中 | 低 |

---

## 三、TensorRT 為什麼不能跨平台？

這是初學者最常見的疑問。TensorRT 在 build（export）時做了以下事情：

### 1. Layer Fusion（層融合）
把多個運算合併成一個 CUDA kernel。例如：
```
Conv2D → BatchNorm → ReLU
          ↓ (fusion)
    ConvBNReLU（單一 kernel）
```
融合的方式依 GPU 架構不同而不同。

### 2. Kernel Auto-Tuning（核心自動調優）
TensorRT 會對每一層**枚舉所有可能的 CUDA kernel 實作**，然後在當前 GPU 上實際跑過一遍，選出最快的那個。

這意味著：
- RTX 4070（Ada Lovelace，SM 8.9）選出的 kernel
- 和 Jetson Orin Nano（Ampere，SM 8.7）選出的 kernel 完全不同

### 3. Precision Calibration（精度校準）
FP16 / INT8 量化的截斷範圍，是根據當前 GPU 的數值行為校準的。

### 4. Device-Specific Serialization（裝置特定序列化）
最終的 `.engine` 是針對特定 GPU 的 binary，包含只有該 GPU 能解讀的指令。

**結論：**
- RTX 4070 build 的 `.engine` → 無法在 Jetson Orin Nano 執行
- 同廠牌不同型號通常也不互通（偶爾相同架構版本可能相容，但不保證）
- 解法：**在每台目標裝置上，從同一份 `.onnx` 或 `.pt` 各自 build 一次 engine**

---

## 四、NVIDIA 官方推薦的工具鏈

### 4-1. TensorRT 本體

**定位：NVIDIA GPU 推理最佳化核心函式庫**

- 核心是 C++ library，有 Python binding（`import tensorrt`）
- 安裝方式：
  - Windows / Linux x86：`pip install tensorrt`（需先安裝 CUDA Toolkit）
  - Jetson（JetPack 6.x）：預裝，不需另外安裝
- Ultralytics 整合：`model.export(format='engine')` 底層就是呼叫 TensorRT API

量化精度選擇：

| 精度 | 速度 | 精度損失 | 需要校準資料 |
|------|------|---------|------------|
| FP32 | 基準 | 無 | 否 |
| FP16 | ~2x | 極小 | 否（推薦入門）|
| INT8 | ~3~4x | 小 | 需要（校準集）|

---

### 4-2. TensorRT 官方 Docker 映像檔（NGC）

**為什麼用 Docker？**

TensorRT 安裝步驟繁瑣：CUDA 版本、cuDNN 版本、TensorRT 版本三者需要精確對齊，稍有不符就會出現莫名其妙的錯誤。NVIDIA 的解法是提供官方預裝 Docker image。

```bash
# 從 NVIDIA GPU Cloud (NGC) 拉取官方 TensorRT container
# 映像版本格式：YY.MM-py3 (e.g. 24.12 = 2024年12月)
docker pull nvcr.io/nvidia/tensorrt:24.12-py3

# 啟動容器（掛載工作目錄到 /workspace）
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/tensorrt:24.12-py3 bash

# 容器內執行模型轉換
cd /workspace
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

**NGC（NVIDIA GPU Cloud）是什麼？**
- NVIDIA 的官方容器登錄倉庫，類似 Docker Hub 但專注 GPU 工作負載
- 提供：TensorRT、Triton、CUDA、PyTorch、JAX、NeMo 等官方預裝映像
- 部分映像需要免費註冊 NGC 帳號才能 pull

---

### 4-3. NVIDIA Triton Inference Server

**定位：生產環境的多模型推理服務框架**

Triton 是 NVIDIA 設計用來在雲端 / 伺服器上**同時服務多個模型**的框架。

```
HTTP/gRPC 客戶端
       ↓
  Triton Server
       ↓
  ┌──────────┬──────────┬──────────┐
  │TensorRT  │ONNX RT   │PyTorch   │  ← 多後端同時支援
  │Model A   │Model B   │Model C   │
  └──────────┴──────────┴──────────┘
```

主要功能：
- **Dynamic Batching**：自動把多個同時到來的請求合批推理，提升 GPU 利用率
- **Model Ensemble**：把前處理 → 推理 → 後處理串成 pipeline
- **A/B Testing**：同時部署多版本模型進行比較
- **Metrics**：Prometheus 指標，方便監控推理延遲與吞吐量

```bash
# Triton Server 快速啟動（需要準備 model repository 目錄結構）
docker run --gpus all -it --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/model_repo:/models \
    nvcr.io/nvidia/tritonserver:24.12-py3 \
    tritonserver --model-repository=/models

# model_repo 目錄結構:
# model_repo/
# └── yolov11n/
#     ├── config.pbtxt      <- 模型設定（後端、輸入輸出格式）
#     └── 1/
#         └── model.engine  <- TensorRT engine 檔案
```

**適用場景：**
- 雲端 API 推理服務（需要低延遲 + 高吞吐）
- 多模型路由（根據請求類型選擇不同模型）
- 不適合：邊緣裝置、Jetson 單機推理（overhead 過大）

---

### 4-4. DeepStream SDK（Jetson 重點）

**定位：Jetson 上的視訊分析流水線框架**

DeepStream 是 NVIDIA 專為**多路視訊流 AI 分析**設計的 SDK，基於 GStreamer 管線。

```
RTSP 攝影機 × 4
      ↓
 DeepStream Pipeline
      ↓
 ┌────────────┐
 │TensorRT    │  ← 直接整合，零拷貝 GPU 記憶體
 │Detection   │
 └────────────┘
      ↓
 結果輸出（RTSP 串流 / 檔案 / Redis / Kafka）
```

特點：
- 多路攝影機同時處理（Jetson Orin Nano 支援 2~4 路 Full HD）
- 使用 NVMM（NVIDIA Memory Model）零拷貝，避免 CPU↔GPU 記憶體來回複製
- 適合工廠監控、交通偵測、零售分析等場景

**學習曲線較高**，建議先熟悉 TensorRT 直接推理後再學 DeepStream。

---

### 4-5. TAO Toolkit（Training + 模型剪枝）

**定位：訓練最佳化工具，輸出可直接接 TensorRT**

- 提供預訓練模型（Detection、Segmentation、Re-ID、License Plate 等）
- 支援 Pruning（結構化剪枝）：在不明顯損失精度的情況下縮小模型
- 支援 QAT（Quantization-Aware Training）：在訓練時模擬 INT8 量化，比訓練後量化更準確
- 輸出格式直接可接 TensorRT

如果你的 yolo11n 精度還不夠，但 yolo11s 太慢，TAO 的 pruning 可以找到中間甜蜜點。

---

## 五、ONNX Runtime：最佳平衡點（初學者推薦入門）

ONNX Runtime（ORT）是最適合初學者的部署起點：

**優點：**
- 比 PyTorch 快 10~30%（CUDA ExecutionProvider）
- 同一份 `.onnx` 在 Windows / Jetson / Intel 上都能跑，無需 re-build
- 安裝簡單：`pip install onnxruntime-gpu`
- 不需要 TensorRT 環境

```python
import onnxruntime as ort

# 優先使用 CUDA，fallback 到 CPU
sess = ort.InferenceSession(
    "yolo11n.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 確認使用哪個 Provider
print(sess.get_providers())  # ['CUDAExecutionProvider']

# 推理
output = sess.run(None, {"images": input_blob})
```

---

## 六、本次學習路徑建議

### 階段路線（由簡到難）

```
Step 1  PyTorch .pt
        確認模型功能正確，建立 FPS 基準線
        ↓
Step 2  ONNX Export + ONNX Runtime
        體驗跨平台格式，同一份 .onnx 在 Windows 和 Jetson 都能跑
        ↓
Step 3  TensorRT FP16（各自機器 build）
        體驗最大加速，理解平台綁定的代價
        ↓
Step 4  TensorRT INT8（進階，需校準資料集）
        適合 Jetson 記憶體受限 + 追求最高效能的場景
        ↓
Step 5  Triton Inference Server（選修）
        用 Docker 提供 REST API 推理服務，適合雲端或多裝置管理
```

### 預期效能（Jetson Orin Nano，yolo11n，640×640）

| 步驟 | 格式 | 預估 FPS |
|------|------|---------|
| Step 1 | PyTorch FP32 | 15~25 FPS |
| Step 2 | ONNX + CUDA EP | 30~50 FPS |
| Step 3 | TensorRT FP16 | 60~100 FPS |
| Step 4 | TensorRT INT8 | 80~150 FPS |

### 對應 Ultralytics export 指令

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")

# Step 2: ONNX
model.export(format='onnx', imgsz=640, simplify=True)

# Step 3: TensorRT FP16（必須在目標機器上執行）
model.export(format='engine', imgsz=640, half=True, batch=1, device=0)

# Step 4: TensorRT INT8（需要校準資料）
model.export(format='engine', int8=True, data='coco8.yaml', imgsz=640, device=0)
```

---

## 七、選擇推理引擎的決策樹

```
你的部署目標是?
│
├── NVIDIA GPU（雲端伺服器）
│     └── 需要服務 API？
│           ├── 是 → Triton Inference Server + TensorRT
│           └── 否 → 直接 TensorRT engine
│
├── Jetson 邊緣裝置
│     ├── 單路攝影機，簡單部署 → TensorRT engine（本腳本流程）
│     └── 多路視訊流，生產環境 → DeepStream + TensorRT
│
├── Intel CPU / iGPU（例如 NUC、工業 PC）
│     └── OpenVINO（比 ONNX Runtime 在 Intel 上更快）
│
├── 手機 / 嵌入式 ARM（無 NVIDIA GPU）
│     └── TFLite 或 ONNX Runtime（ARM CPU）
│
└── 跨平台（同一份模型要在多種硬體上跑）
      └── ONNX Runtime（效能次之，但可移植性最佳）
```

---

## 八、常見問題（FAQ）

**Q: 我在 Windows 上 export 的 .engine 可以複製到 Jetson 上用嗎？**
A: 不行。TensorRT engine 是針對特定 GPU 架構 build 的 binary，必須在目標機器上各自 build。

**Q: 那 .onnx 可以複製嗎？**
A: 可以。ONNX 是跨平台格式，複製到任何機器都能直接用 ONNX Runtime 推理。TensorRT 也是先把 .onnx 轉換成 .engine，所以你只需要維護一份 .onnx。

**Q: FP16 和 INT8 的精度差多少？**
A: 以 COCO mAP 為例：
- FP32 → FP16：通常 < 0.1% mAP 下降，幾乎無感
- FP32 → INT8：通常 0.5~2% mAP 下降，對大多數應用可接受

**Q: Jetson Orin Nano 支援 FP16 嗎？**
A: 支援。Jetson Orin Nano 使用 Ampere 架構（SM 8.7），原生支援 FP16 Tensor Core。

**Q: TensorRT Docker 在 Jetson 上能用嗎？**
A: 可以，但 Jetson 有自己的 JetPack 容器（`nvcr.io/nvidia/l4t-tensorrt`），使用 ARM 版本。一般的 x86 TensorRT Docker 映像無法在 Jetson 上執行。

---

## 九、參考資源

- TensorRT 官方文件：developer.nvidia.com/tensorrt
- NGC 容器倉庫：catalog.ngc.nvidia.com
- Triton Inference Server：github.com/triton-inference-server/server
- DeepStream SDK：developer.nvidia.com/deepstream-sdk
- Ultralytics 部署文件：docs.ultralytics.com/modes/export
- Jetson 社群論壇：forums.developer.nvidia.com/c/agx-autonomous-machines/jetson
