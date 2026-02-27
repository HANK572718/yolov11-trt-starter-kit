# YOLOv11 TensorRT Starter Kit

YOLOv11n 從 PyTorch 到 TensorRT FP16 的完整最佳化流程示範，目標部署平台為 Windows NVIDIA GPU 與 Jetson Orin Nano。

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

## 開發環境

詳見 `hank_laptop_env.md`。

- OS: Windows 11 Build 26100
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU（8GB VRAM，Ada Lovelace SM 8.9）
- Driver: 566.14 / CUDA: 12.4 / TensorRT: 10.6.0.post2
- Python: 3.11.9 / PyTorch: 2.4.1+cu124 / Ultralytics: 8.4.7
