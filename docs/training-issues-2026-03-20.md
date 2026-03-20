# 訓練議題記錄 — 2026-03-20

## 背景

使用 `train_yolo11.py` 以同一份資料集（`0317_merged_yolo_converted`）訓練 YOLOv11n-seg 模型。
先前已有 YOLOv11s-seg 的成功訓練紀錄（30 epochs，Box mAP50 ≈ 0.953）。

---

## 議題一：重複訓練資料夾

### 現象

`runs/segment/` 下存在兩個名稱相似的資料夾：

```
runs/segment/yolo11s_seg_custom/           ← 無 wandb
runs/segment/yolo11-training/yolo11s_seg_custom/  ← 有 wandb
```

### 原因

兩次訓練使用完全相同的超參數（`seed=0` + `deterministic=True`），
唯一差異是 `project` 參數：第一次為 `null`（預設路徑），第二次指定 `project=yolo11-training`。
結果數值完全一致，但後者多了 wandb 實驗追蹤資料。

### 處置

刪除無 wandb 的舊版資料夾 `runs/segment/yolo11s_seg_custom/`，
保留 `runs/segment/yolo11-training/yolo11s_seg_custom/` 為正式版本。

---

## 議題二：訓練 n 模型失敗（Windows 分頁檔不足）

### 指令

```bash
poetry run python train_yolo11.py --source ... --format yolo --model n
```

### 錯誤訊息

```
OSError: [WinError 1455] 頁面檔太小，無法完成操作
Error loading "cublas64_12.dll" or one of its dependencies.
```

### 原因

`resolve_workers()` 根據可用虛擬記憶體（25.8 GB）選擇 `workers=4`。
Windows 多進程 spawn 時每個 worker 各自載入 `torch`、`cublas64_12.dll`，
實際分頁檔使用量超過系統上限。

### 修復

加入 `--workers 0` 旗標，強制使用主進程（single-process）模式：

```bash
poetry run python train_yolo11.py ... --workers 0
```

---

## 議題三：CUDA OOM（AMP 檢查階段）

### 錯誤訊息

```
RuntimeError: CUDA error: out of memory
assert amp_allclose(YOLO("yolo26n.pt"), im)
```

### 原因

ultralytics 8.4.23 在 `_setup_train` 初始化時，會額外下載並載入 `yolo26n.pt`
進行 AMP（Automatic Mixed Precision）驗證推論，導致 GPU 顯存不足。
（前次 workers=4 失敗後的殘留 CUDA 狀態可能加劇此問題）

### 修復

新增 `--no-amp` CLI 旗標至 `train_yolo11.py`，允許跳過 AMP 初始化：

**`train_yolo11.py` 修改內容：**

1. `train()` 函數新增 `amp: bool = True` 參數
2. `train_kwargs` 加入 `amp=amp`
3. CLI 新增 `--no-amp` 旗標（`action="store_true"`）
4. `main()` 呼叫 `train()` 時傳入 `amp=not args.no_amp`

```bash
poetry run python train_yolo11.py ... --workers 0 --no-amp
```

---

## 議題四：cuDNN 執行失敗（CUDNN_STATUS_EXECUTION_FAILED）

### 錯誤訊息

```
fatal   : Memory allocation failure
cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
```

### 原因

RTX 5070 Ti（Blackwell 架構）搭配 torch-2.7.0+cu128 時，
cuDNN v8 API 在 BatchNorm forward pass 出現執行失敗。
AutoBatch 階段也因相同原因無法正常探測，回退至 batch=16 後仍失敗。
根本原因推測為多次失敗 run 累積的 CUDA context 損壞，
以及 cuDNN v8 API 與此 GPU/driver 版本的相容性問題。

### 修復

設定環境變數 `TORCH_CUDNN_V8_API_DISABLED=1`，停用 cuDNN v8 API；
同時固定 `--batch 16` 跳過 AutoBatch：

```bash
TORCH_CUDNN_V8_API_DISABLED=1 poetry run python train_yolo11.py \
  --source ... --format yolo --model n \
  --workers 0 --no-amp --batch 16
```

> **注意：** 若未來在相同環境重新訓練 n 模型，建議先重新開機清除 CUDA context，
> 再評估是否仍需 `TORCH_CUDNN_V8_API_DISABLED=1`。

---

## 最終成功指令

```bash
TORCH_CUDNN_V8_API_DISABLED=1 poetry run python train_yolo11.py \
  --source "C:\Users\suser\Documents\yolov11-trt-starter-kit\dataset\0317_merged_yolo_converted" \
  --format yolo \
  --model n \
  --workers 0 \
  --no-amp \
  --batch 16
```

---

## 訓練結果（YOLOv11n-seg）

| 指標 | n 模型 | s 模型（參考） |
|------|--------|--------------|
| Box mAP50 | 0.956 | 0.953 |
| Box mAP50-95 | 0.867 | 0.887 |
| Mask mAP50 | 0.907 | 0.925 |
| Mask mAP50-95 | 0.806 | 0.840 |
| 參數量 | 2.84M | 9.42M |
| 推論速度 | 2.2ms | — |

權重：`runs/segment/yolo11-training/yolo11n_seg_custom/weights/best.pt`

---

## 修改的程式碼檔案

| 檔案 | 修改內容 |
|------|---------|
| `train_yolo11.py` | 新增 `--no-amp` 旗標與 `amp` 參數傳遞 |
