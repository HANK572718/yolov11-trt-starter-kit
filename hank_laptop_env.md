# 開發環境規格記錄
## Hank Laptop - MSI (本次腳本驗證與開發環境)

> 本檔案記錄 yolov11_tensorrt_jetson.py 開發與測試時所使用的本機環境。
> 作為參與者複製此雛形專案時的版本比對基準。

---

## 系統資訊

| 項目 | 數值 |
|------|------|
| OS | Windows 11 (Build 26100) |
| 架構 | AMD64 (x86_64) |
| 機器名稱 | MSI |

---

## CPU

| 項目 | 數值 |
|------|------|
| 型號 | Intel Core (Family 6, Model 186 - Raptor Lake 13th Gen) |
| 實體核心 | 10 核 |
| 邏輯執行緒 | 16 執行緒 |
| 最高頻率 | 2.4 GHz（基礎，Laptop 節流版） |

---

## 記憶體

| 項目 | 數值 |
|------|------|
| 系統 RAM | 63.7 GB |

---

## GPU

| 項目 | 數值 |
|------|------|
| 型號 | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8.0 GB |
| GPU 架構 | Ada Lovelace (SM 8.9) |
| CUDA Multiprocessors | 24 SM |
| NVIDIA Driver | 566.14 |

> **注意：** Driver 566.14 最高支援 CUDA Runtime 12.7。
> TensorRT 10.15 以上需要 Driver 570+（CUDA 12.8），會出現 CUDA error 35。
> 本環境使用 **TensorRT 10.6.0.post2**（CUDA 12.6 相容）。

---

## CUDA 環境

| 項目 | 數值 |
|------|------|
| CUDA Toolkit 版本 | 12.4 |
| CUDA_PATH | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` |
| cuDNN 版本 | 9.1.0 (build 90100) |

---

## Python 環境（Poetry 虛擬環境）

| 套件 | 版本 |
|------|------|
| Python | 3.11.9 (MSC v.1938 64bit) |
| Poetry | 2.1.1 |
| PyTorch | 2.4.1+cu124 |
| TensorRT | **10.6.0.post2** |
| ONNX Runtime | 1.23.2 |
| Ultralytics | 8.4.7 |

### ONNX Runtime 可用 Provider（此環境）

```
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

---

## TRT DLL 搜尋路徑（腳本 Cell 0 自動加入）

腳本啟動時會自動偵測並加入以下目錄（Windows 限定）：

```
.venv/Lib/site-packages/tensorrt_libs/          (nvinfer_10.dll 等 11 個 DLL)
.venv/Lib/site-packages/nvidia/cuda_runtime/bin/ (cudart64_12.dll 等)
C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/
```

---

## Benchmark 結果（此環境，yolo11n，bus.jpg，640x640）

| 格式 | Pre(ms) | Inf(ms) | Post(ms) | Total(ms) | FPS | 加速倍率 |
|------|---------|---------|---------|---------|-----|---------|
| PyTorch PT (FP32) | 1.5 | 7.4 | 1.4 | 10.3 | 97.5 | 1.00x |
| ONNX Runtime (CUDA EP) | 2.3 | 5.6 | 1.5 | 9.5 | 105.5 | 1.08x |
| TensorRT FP16 | 3.8 | **3.5** | 2.0 | 9.2 | **108.5** | **1.11x** |

> RTX 4060 Laptop 基礎效能已高，TRT 加速比 Jetson 不明顯。
> 在 Jetson Orin Nano（15~25 FPS baseline）上 TRT FP16 預估可達 60~100 FPS（4~6x 加速）。

---

## 環境複製指令

```bash
# 前提：已安裝 Python 3.11.x 與 Poetry 2.x

# 1. 安裝依賴
poetry install

# 2. 修正 TensorRT 版本（重要：pip 預設裝 10.15 但需 10.6）
poetry run pip install tensorrt==10.6.0.post2
poetry run pip install tensorrt-cu12==10.6.0.post2 \
                       tensorrt-cu12-libs==10.6.0.post2 \
                       tensorrt-cu12-bindings==10.6.0.post2

# 3. 確認 TRT Builder 可正常運作
poetry run python -c "import tensorrt as trt; b=trt.Builder(trt.Logger()); print('TRT OK' if b else 'TRT FAIL')"

# 4. 執行主腳本
poetry run python yolov11_tensorrt_jetson.py
```
