# YOLOv11 訓練指南

本指南適合剛開始使用 ISAT 標注工具並想訓練 YOLOv11 模型的初學者。

---

## 目錄

1. [快速開始](#1-快速開始)
2. [ISAT 標注工作流](#2-isat-標注工作流)
3. [資料集格式說明](#3-資料集格式說明)
4. [完整 CLI 參數說明](#4-完整-cli-參數說明)
5. [訓練超參數說明](#5-訓練超參數說明)
6. [wandb 監控設定](#6-wandb-監控設定)
7. [訓練結果](#7-訓練結果)
8. [下一步：TensorRT 部署](#8-下一步tensorrt-部署)

---

## 1. 快速開始

### 環境安裝

```bash
poetry install
```

### 最小訓練指令

**ISAT 模式**（自動偵測格式、segment 任務）：
```bash
poetry run python train_yolo11.py --source C:\my_data\images --model s
```

**YOLO 模式**（現有 YOLO 格式資料集）：
```bash
poetry run python train_yolo11.py --source dataset_yolo_format --format yolo --task detect
```

---

## 2. ISAT 標注工作流

### 啟動 ISAT

```bash
poetry run isat-sam
```

### 標注步驟

1. **開啟資料夾**：File → Open Folder，選擇放有圖片的資料夾
2. **新增類別**：在右側 Category 欄位輸入類別名稱（例如 `device`、`connector`）
3. **畫多邊形輪廓**：
   - 按 `Q` 開始畫多邊形
   - 沿著物件邊緣逐點點擊
   - 按右鍵或 `Enter` 完成
4. **設定 category**：每個標注區域必須選擇正確的 category（**不能留在 `__background__`**）
5. **儲存**：按 `Ctrl+S` 儲存，ISAT 會產生與圖片同名的 `.json` 檔

> **重要**：每個物件都要設定有意義的 category 名稱。若 category 全部都是 `__background__`，訓練腳本會報錯並中止。

### 設定 isat.yaml 類別清單（可選）

在 ISAT 安裝目錄下編輯 `isat.yaml`，預先定義好類別清單，之後標注時可直接從下拉選單選擇，不需要手動輸入。

```yaml
label:
  - device
  - connector
```

---

## 3. 資料集格式說明

### ISAT 模式

只需一個資料夾，同時放置圖片和 JSON 檔（扁平結構）：

```
my_data/images/
├── image_001.jpg
├── image_001.json    ← ISAT 自動產生
├── image_002.jpg
├── image_002.json
└── ...
```

訓練腳本會自動：
- 讀取所有 JSON，萃取類別名稱
- 轉換為 YOLO 格式
- 切分 train / val（預設比例 0.2）
- 輸出至 `my_data/images_yolo_converted/`

### YOLO 模式

需符合 YOLO 標準結構，且根目錄要有 `dataset.yaml`：

```
dataset_yolo_format/
├── dataset.yaml
├── images/
│   ├── train/
│   │   └── *.jpg
│   └── val/
│       └── *.jpg
└── labels/
    ├── train/
    │   └── *.txt
    └── val/
        └── *.txt
```

`dataset.yaml` 範例：

```yaml
path: C:\Users\suser\Documents\yolov11-trt-starter-kit\dataset_yolo_format
train: images/train
val: images/val
test:

names:
    0: device
    1: connector
```

---

## 4. 完整 CLI 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--source` | 資料集路徑 | `dataset_yolo_format` |
| `--format` | `isat` / `yolo` / `auto` | `auto` |
| `--task` | `detect` / `segment` | `segment` |
| `--model` | `n` / `s` / `m` / `l` / `x` / `all` | `s` |
| `--epochs` | 最大訓練輪次 | `100` |
| `--imgsz` | 輸入圖片尺寸（像素） | `640` |
| `--batch` | Batch size，`-1` = AutoBatch | `-1` |
| `--device` | CUDA 裝置編號或 `cpu` | `0` |
| `--val-split` | 驗證集比例（ISAT 模式） | `0.2` |
| `--no-wandb` | 停用 wandb 追蹤 | 預設開啟 |

### 使用範例

```bash
# ISAT segment（預設）
poetry run python train_yolo11.py --source C:\my_data\images --model s

# ISAT detect
poetry run python train_yolo11.py --source C:\my_data\images --task detect --model s

# YOLO 格式 detect
poetry run python train_yolo11.py --source dataset_yolo_format --format yolo --task detect

# 訓練所有尺寸
poetry run python train_yolo11.py --source C:\my_data\images --model all

# 不啟用 wandb
poetry run python train_yolo11.py --source C:\my_data\images --no-wandb

# 指定 GPU 2
poetry run python train_yolo11.py --source C:\my_data\images --device 2
```

---

## 5. 訓練超參數說明

以下參數已內建於訓練腳本，無需手動指定：

| 參數 | 值 | 說明 |
|------|----|------|
| `patience` | `10` | Early Stopping：連續 10 個 epoch val 指標無改善則自動停止 |
| `cos_lr` | `True` | 餘弦學習率排程，收斂更穩定 |
| `augment` | `True` | 啟用 Mosaic、Mixup、HSV、Flip 等資料增強 |
| `cache` | `False` | 不快取圖片到 RAM。若 RAM > 16 GB 可改 `True` 加速 |
| `save_period` | `10` | 每 10 個 epoch 儲存一次 checkpoint |
| `workers` | `4` | 資料載入的 worker 執行緒數 |

---

## 6. wandb 監控設定

[wandb](https://wandb.ai) 可視化 loss 曲線、mAP、混淆矩陣等訓練指標。

### 一次性設定（登入）

```bash
wandb login
```

按提示貼上 API key（在 wandb.ai 帳號設定頁取得）。

### 訓練過程

- 登入後，每次訓練會自動上傳到 wandb project `yolo11-training`
- 若未登入，Ultralytics 會靜默跳過 wandb，訓練仍正常進行
- 加上 `--no-wandb` 明確停用

---

## 7. 訓練結果

訓練完成後，結果存放於：

- **Detect 任務**：`runs/detect/<name>/`
- **Segment 任務**：`runs/segment/<name>/`

重要檔案：

| 檔案 | 說明 |
|------|------|
| `weights/best.pt` | 驗證指標最佳的模型 |
| `weights/last.pt` | 最後一個 epoch 的模型 |
| `results.csv` | 每個 epoch 的 loss 與 mAP 數值 |
| `confusion_matrix.png` | 混淆矩陣 |
| `val_batch0_pred.jpg` | 驗證集預測範例 |

---

## 8. 下一步：TensorRT 部署

訓練好 `best.pt` 之後，可以轉換為 TensorRT 引擎部署到 Jetson 裝置：

```bash
# 參考腳本
poetry run python yolov11_tensorrt_jetson.py
```

詳情請參考專案中的 `yolov11_tensorrt_jetson.py`。
