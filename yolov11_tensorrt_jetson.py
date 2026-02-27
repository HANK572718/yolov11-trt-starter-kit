# %% [Cell 0] 匯入模組與平台偵測
"""
YOLOv11n COCO 跨平台推理與 TensorRT 最佳化腳本
=========================================================
支援平台:
  - Windows + NVIDIA GPU (RTX 4070)
  - Jetson Orin Nano (JetPack 6.x)

推理格式進化路徑:
  PT (.pt) --> ONNX (.onnx) --> TensorRT (.engine)

重要提示:
  TensorRT .engine 必須在目標機器上各自 build，不可跨平台複製。
  RTX 4070 build 的 .engine 無法在 Jetson 上執行，反之亦然。
  請在每台目標裝置上各自執行 Phase 3-A。

使用方式:
  - 在 VS Code / PyCharm 中以 #%% 分區塊逐段執行
  - 或 python yolov11_tensorrt_jetson.py 全部執行

依賴套件:
  pip install ultralytics onnxruntime-gpu
  TensorRT: Windows 需另裝，Jetson JetPack 已內建
"""

import os
import platform
import site
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

# ============ 平台偵測 ============
IS_JETSON    = platform.machine() == 'aarch64'
HAS_CUDA     = torch.cuda.is_available()
DEVICE       = 0 if HAS_CUDA else 'cpu'
PLATFORM     = "Jetson Orin Nano" if IS_JETSON else platform.node()

# ============ 顯示控制旗標 ============
# False = 跳過所有 cv2.imshow 視窗（適合全自動執行、ssh 無桌面環境）
# True  = 每個推理階段結束後彈出視窗顯示結果
SHOW_DISPLAY  = False

# False = 跳過所有攝影機即時推理 block（適合無攝影機或自動化測試）
# True  = 執行攝影機推理（需要實體攝影機）
RUN_CAMERA    = False

# ============ Windows TensorRT DLL 搜尋路徑修正 ============
# pip install tensorrt 在 Windows 上將 DLL 放在 site-packages 的子目錄
# 但 Windows 不會自動把這些路徑加入 DLL 搜尋清單，導致 trt.Builder() 回傳 null
# 需在任何 TensorRT 呼叫前，用 os.add_dll_directory() 手動加入
if sys.platform == 'win32':
    _trt_dll_dirs_found = []
    for _sp in site.getsitepackages():
        # TRT 10.x pip 套件將 DLL 放在這些子目錄
        for _sub in ['tensorrt_libs', 'tensorrt', 'nvidia\\cuda_runtime\\bin',
                     'nvidia\\cublas\\bin', 'nvidia\\cudnn\\bin']:
            _candidate = Path(_sp) / _sub
            if _candidate.exists() and list(_candidate.glob('*.dll')):
                os.add_dll_directory(str(_candidate))
                _trt_dll_dirs_found.append(str(_candidate))
    # 也嘗試 CUDA_PATH 環境變數指向的 bin 目錄
    _cuda_path = os.environ.get('CUDA_PATH', '')
    if _cuda_path:
        _cuda_bin = Path(_cuda_path) / 'bin'
        if _cuda_bin.exists():
            os.add_dll_directory(str(_cuda_bin))
            _trt_dll_dirs_found.append(str(_cuda_bin))

print("=" * 60)
print("環境資訊")
print("=" * 60)
print(f"  Platform  : {PLATFORM}")
print(f"  Python    : {sys.version.split()[0]}")
print(f"  PyTorch   : {torch.__version__}")
print(f"  CUDA 可用  : {HAS_CUDA}")
if HAS_CUDA:
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  推理裝置   : {'GPU (device 0)' if HAS_CUDA else 'CPU'}")
if sys.platform == 'win32':
    print(f"  TRT DLL 目錄數: {len(_trt_dll_dirs_found)}")
    for _d in _trt_dll_dirs_found:
        print(f"    {_d}")

# ============ 全域設定 ============
MODEL_PT      = "yolo11n.pt"
MODEL_ONNX    = "yolo11n.onnx"
MODEL_ENGINE  = "yolo11n.engine"

IMGSZ         = 640            # 推理圖片尺寸
CONF_THRESH   = 0.5            # 信心度閾值
IOU_THRESH    = 0.45           # NMS IOU 閾值
PERSON_ONLY   = [0]            # COCO class 0 = person (人物偵測)
WARMUP_RUNS   = 3              # 暖機次數（排除 CUDA 初始化誤差）
MEASURE_RUNS  = 20             # 計時次數

# 測試圖片：ultralytics 內建 bus.jpg（含多名行人，方便測試）
TEST_IMAGE    = str(ASSETS / "bus.jpg")

# 速度紀錄（各階段結果存入此 dict，最後統一比較）
BENCH: dict   = {}

print(f"\n  測試圖片  : {TEST_IMAGE}")
print(f"  偵測類別  : COCO class {PERSON_ONLY} (person)")
print(f"  SHOW_DISPLAY: {SHOW_DISPLAY}  |  RUN_CAMERA: {RUN_CAMERA}")


# =====================================================================
# %% [Phase 1] 下載 YOLOv11n COCO 預訓練權重
"""
Phase 1 - 下載模型
------------------
ultralytics 自動從 CDN 下載 yolo11n.pt 到:
  Windows : C:/Users/<user>/.config/Ultralytics/
  Linux   : ~/.config/Ultralytics/

若本機已有快取則直接載入，不會重複下載。

模型資訊:
  yolo11n: nano 版（最小最快），約 2.6M 參數
  COCO 80 類：person, car, bicycle, dog ...
  yolo11n.pt 約 5.6 MB
"""

print("\n" + "=" * 60)
print("Phase 1 : 載入 YOLOv11n COCO 預訓練模型")
print("=" * 60)

pt_model = YOLO(MODEL_PT)

param_count = sum(p.numel() for p in pt_model.model.parameters())
print(f"  模型檔   : {MODEL_PT}")
print(f"  參數量   : {param_count:,}")
print(f"  類別數   : {len(pt_model.names)}")
print(f"  class 0  : {pt_model.names[0]}")   # person


# =====================================================================
# %% [Phase 2-A] PyTorch 基礎推理（Baseline）
"""
Phase 2-A - PyTorch 原始推理
-----------------------------
以 .pt 模型進行推理，記錄推理速度作為後續加速的基準線。

classes=[0] 僅偵測行人（COCO person）。

SOURCE 可替換為:
  TEST_IMAGE          -> 使用 ultralytics 內建測試圖 (bus.jpg)
  "path/to/image.jpg" -> 自訂圖片路徑
  "path/to/video.mp4" -> 影片檔
  0                   -> 攝影機（USB 第一顆）
"""

print("\n" + "=" * 60)
print("Phase 2-A : PyTorch 推理（Baseline）")
print("=" * 60)

SOURCE = TEST_IMAGE   # <-- 可改為攝影機 0 或影片路徑

# 暖機（讓 CUDA 完成初始化，避免第一次計時失準）
print(f"  Warmup {WARMUP_RUNS} runs ...")
for _ in range(WARMUP_RUNS):
    pt_model.predict(
        SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
        imgsz=IMGSZ, device=DEVICE, verbose=False
    )

# 計時
pre_lst, inf_lst, post_lst = [], [], []
print(f"  Measuring {MEASURE_RUNS} runs ...")
for _ in range(MEASURE_RUNS):
    res = pt_model.predict(
        SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
        imgsz=IMGSZ, device=DEVICE, verbose=False
    )
    spd = res[0].speed   # {'preprocess': ms, 'inference': ms, 'postprocess': ms}
    pre_lst.append(spd['preprocess'])
    inf_lst.append(spd['inference'])
    post_lst.append(spd['postprocess'])

BENCH['PyTorch PT'] = {
    'pre' : float(np.mean(pre_lst)),
    'inf' : float(np.mean(inf_lst)),
    'post': float(np.mean(post_lst)),
}
_total = sum(BENCH['PyTorch PT'].values())
print(f"  Pre={BENCH['PyTorch PT']['pre']:.1f}ms  "
      f"Inf={BENCH['PyTorch PT']['inf']:.1f}ms  "
      f"Post={BENCH['PyTorch PT']['post']:.1f}ms  "
      f"Total={_total:.1f}ms  FPS={1000 / _total:.1f}")

# 顯示偵測結果
if SHOW_DISPLAY:
    _res_pt = pt_model.predict(
        SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
        imgsz=IMGSZ, device=DEVICE, verbose=False
    )
    _annotated = _res_pt[0].plot()
    cv2.imshow("Phase 2-A : PyTorch Detection", _annotated)
    print("  按任意鍵關閉視窗 ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================================
# %% [Phase 2-A Camera] PyTorch 即時攝影機推理（選用）
"""
Phase 2-A-Camera - 攝影機即時推理
-----------------------------------
修改 CAMERA_ID 為攝影機編號（0=內建，1=USB 外接）。
Jetson 若使用 CSI camera，請將 cap 改為 GStreamer pipeline（參見 Jetson block）。
按 'q' 退出。
"""

CAMERA_ID = 0

if RUN_CAMERA:
    print("\n[攝影機] PyTorch 即時推理（RUN_CAMERA=True）...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"  無法開啟攝影機 {CAMERA_ID}")
    else:
        print("  推理中 ... 按 'q' 退出")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            cam_res = pt_model.predict(
                frame, classes=PERSON_ONLY, conf=CONF_THRESH,
                imgsz=IMGSZ, device=DEVICE, verbose=False
            )
            fps_val = 1.0 / (time.perf_counter() - t0)
            annotated_frame = cam_res[0].plot()
            cv2.putText(annotated_frame, f"PT FPS: {fps_val:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera - PyTorch Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
else:
    print("  [Phase 2-A Camera] 略過（RUN_CAMERA=False）")


# =====================================================================
# %% [Phase 2-B] 匯出 ONNX 格式
"""
Phase 2-B - 匯出 ONNX
-----------------------
ONNX 是通用中間格式，幾乎所有推理引擎都支援。
同一份 .onnx 可複製到任何機器（Windows / Jetson / Intel CPU）直接使用。

匯出選項說明:
  simplify=True  -> 使用 onnx-simplifier 移除冗餘節點，縮小模型並加速
  dynamic=False  -> 固定 batch size，Jetson 使用固定 batch=1 更穩定
  half=False     -> 保持 FP32 精度（量化留給 TensorRT 階段）

輸出檔: yolo11n.onnx（可複製到任意機器使用）
"""

print("\n" + "=" * 60)
print("Phase 2-B : 匯出 ONNX")
print("=" * 60)

_onnx_path = Path(MODEL_ONNX)
if _onnx_path.exists():
    print(f"  已存在，跳過匯出: {MODEL_ONNX} ({_onnx_path.stat().st_size / 1e6:.1f} MB)")
else:
    print("  匯出中 ...")
    pt_model.export(
        format='onnx',
        imgsz=IMGSZ,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f"  匯出完成: {MODEL_ONNX} ({_onnx_path.stat().st_size / 1e6:.1f} MB)")


# =====================================================================
# %% [Phase 2-B Inference] ONNX Runtime 推理（via Ultralytics）
"""
Phase 2-B Inference - ONNX Runtime 推理
-----------------------------------------
使用 Ultralytics 的 ONNX 接口，保留相同的 predict() API。
底層自動使用 onnxruntime，並在 NVIDIA GPU 上啟用 CUDAExecutionProvider。

優點:
  - 同一份 .onnx 在 Windows / Jetson / Intel 上都能跑（可移植）
  - 比 PyTorch 快約 10~30%（視 GPU 而定）
  - 不需要安裝 TensorRT，依賴更簡單
"""

print("\n" + "=" * 60)
print("Phase 2-B : ONNX Runtime 推理（via Ultralytics）")
print("=" * 60)

onnx_model = YOLO(MODEL_ONNX)

print(f"  Warmup {WARMUP_RUNS} runs ...")
for _ in range(WARMUP_RUNS):
    onnx_model.predict(
        SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
        imgsz=IMGSZ, verbose=False
    )

pre_lst, inf_lst, post_lst = [], [], []
print(f"  Measuring {MEASURE_RUNS} runs ...")
for _ in range(MEASURE_RUNS):
    res = onnx_model.predict(
        SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
        imgsz=IMGSZ, verbose=False
    )
    spd = res[0].speed
    pre_lst.append(spd['preprocess'])
    inf_lst.append(spd['inference'])
    post_lst.append(spd['postprocess'])

BENCH['ONNX Runtime'] = {
    'pre' : float(np.mean(pre_lst)),
    'inf' : float(np.mean(inf_lst)),
    'post': float(np.mean(post_lst)),
}
_total = sum(BENCH['ONNX Runtime'].values())
print(f"  Pre={BENCH['ONNX Runtime']['pre']:.1f}ms  "
      f"Inf={BENCH['ONNX Runtime']['inf']:.1f}ms  "
      f"Post={BENCH['ONNX Runtime']['post']:.1f}ms  "
      f"Total={_total:.1f}ms  FPS={1000 / _total:.1f}")


# =====================================================================
# %% [Phase 2-B Raw] 原始 ONNX Runtime 推理（教學：顯示完整流程）
"""
Phase 2-B Raw - 原始 ONNX Runtime 推理
----------------------------------------
不透過 Ultralytics wrapper，直接使用 onnxruntime.InferenceSession。
展示完整的前處理 → 推理 → 後處理流程。

了解此流程的用途:
  - 移植到 C++ / Rust 等語言時需要此邏輯
  - 理解 TensorRT 自訂後處理的原理
  - 部署到不支援 Ultralytics 的嵌入式環境

yolo11n detection 輸出格式:
  shape: [1, 84, 8400]
    - 84 = 4 (x, y, w, h 中心點格式) + 80 (COCO class scores)
    - 8400 = 80x80 + 40x40 + 20x20 三種尺度的預測格點數
"""

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    print("  onnxruntime 未安裝，跳過 Raw ORT 示範")
    print("  安裝指令: pip install onnxruntime-gpu  (NVIDIA GPU)")
    print("            pip install onnxruntime       (CPU only)")


def _letterbox(img: np.ndarray, new_shape: tuple = (640, 640)) -> tuple:
    """
    將圖片縮放至目標尺寸，保持長寬比，空白處填灰色（letterbox）。

    Args:
        img      (np.ndarray): BGR 原始圖片
        new_shape (tuple)    : 目標尺寸 (height, width)

    Returns:
        img_lb (np.ndarray): letterbox 後的圖片
        ratio  (float)     : 縮放比例（用於還原座標）
        pad    (tuple)     : (pad_w, pad_h) 填充量（像素）
    """
    h0, w0 = img.shape[:2]
    h1, w1 = new_shape
    ratio   = min(h1 / h0, w1 / w0)
    new_h   = int(round(h0 * ratio))
    new_w   = int(round(w0 * ratio))
    pad_h   = (h1 - new_h) / 2
    pad_w   = (w1 - new_w) / 2

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_lb      = np.full((h1, w1, 3), 114, dtype=np.uint8)
    top  = int(round(pad_h - 0.1))
    left = int(round(pad_w - 0.1))
    img_lb[top:top + new_h, left:left + new_w] = img_resized
    return img_lb, ratio, (pad_w, pad_h)


def _preprocess_ort(img_bgr: np.ndarray, imgsz: int = 640) -> tuple:
    """
    ONNX Runtime 前處理：BGR → RGB → letterbox → normalize → NCHW float32。

    Args:
        img_bgr (np.ndarray): BGR 原始圖片
        imgsz   (int)       : 模型輸入邊長（預設 640）

    Returns:
        blob  (np.ndarray): float32，shape (1, 3, imgsz, imgsz)
        ratio (float)     : letterbox 縮放比例
        pad   (tuple)     : (pad_w, pad_h) 填充量
    """
    img_lb, ratio, pad = _letterbox(img_bgr, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    blob    = img_rgb.astype(np.float32) / 255.0
    blob    = blob.transpose(2, 0, 1)[np.newaxis]   # HWC -> NCHW
    return blob, ratio, pad


def _postprocess_ort(
    output: np.ndarray,
    original_shape: tuple,
    ratio: float,
    pad: tuple,
    conf_thresh: float = 0.5,
    iou_thresh: float  = 0.45,
    target_classes: list = None,
) -> list:
    """
    解析 yolo11n 輸出 [1, 84, 8400]，還原到原圖座標並套用 NMS。

    Args:
        output         (np.ndarray): 模型輸出，shape (1, 84, 8400)
        original_shape (tuple)     : 原圖尺寸 (h, w)
        ratio          (float)     : letterbox 縮放比例
        pad            (tuple)     : (pad_w, pad_h) letterbox 填充量
        conf_thresh    (float)     : 信心度閾值
        iou_thresh     (float)     : NMS IOU 閾值
        target_classes (list)      : 只保留的類別 ID，None 表示全部

    Returns:
        detections (list): list of dict，每個元素:
                           {'box': [x1,y1,x2,y2], 'score': float, 'class_id': int}
    """
    pred       = output[0].T                              # [8400, 84]
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]                              # [8400, 80]
    class_ids  = np.argmax(cls_scores, axis=1)
    confs      = cls_scores[np.arange(len(class_ids)), class_ids]

    # 信心度過濾
    mask = confs >= conf_thresh
    if target_classes is not None:
        mask &= np.isin(class_ids, target_classes)

    boxes_xywh = boxes_xywh[mask]
    confs      = confs[mask]
    class_ids  = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # xywh（letterbox 座標）-> xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # NMS
    boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confs.tolist(), conf_thresh, iou_thresh)
    indices = indices.flatten() if len(indices) > 0 else []

    # 還原到原圖座標
    oh, ow = original_shape[:2]
    pad_w, pad_h = pad
    detections = []
    for i in indices:
        rx1 = max(0,  (x1[i] - pad_w) / ratio)
        ry1 = max(0,  (y1[i] - pad_h) / ratio)
        rx2 = min(ow, (x2[i] - pad_w) / ratio)
        ry2 = min(oh, (y2[i] - pad_h) / ratio)
        detections.append({
            'box'     : [int(rx1), int(ry1), int(rx2), int(ry2)],
            'score'   : float(confs[i]),
            'class_id': int(class_ids[i]),
        })
    return detections


if _ORT_AVAILABLE:
    print("\n" + "=" * 60)
    print("Phase 2-B Raw : 原始 ONNX Runtime 推理（教學）")
    print("=" * 60)

    _providers = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider'] if HAS_CUDA
        else ['CPUExecutionProvider']
    )
    _sess       = ort.InferenceSession(MODEL_ONNX, providers=_providers)
    _input_name = _sess.get_inputs()[0].name

    print(f"  Provider  : {_sess.get_providers()[0]}")
    print(f"  Input     : {_input_name}  {_sess.get_inputs()[0].shape}")
    print(f"  Output    : {_sess.get_outputs()[0].name}  {_sess.get_outputs()[0].shape}")

    _img_raw             = cv2.imread(TEST_IMAGE)
    _blob, _ratio, _pad  = _preprocess_ort(_img_raw, IMGSZ)

    # 暖機
    for _ in range(WARMUP_RUNS):
        _sess.run(None, {_input_name: _blob})

    # 計時（只計算 session.run，排除前後處理以凸顯純推理差異）
    _t0 = time.perf_counter()
    for _ in range(MEASURE_RUNS):
        _ort_out = _sess.run(None, {_input_name: _blob})
    _ms_per_img = (time.perf_counter() - _t0) * 1000 / MEASURE_RUNS

    _dets = _postprocess_ort(
        _ort_out[0], _img_raw.shape, _ratio, _pad,
        conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH,
        target_classes=PERSON_ONLY,
    )
    print(f"  session.run: {_ms_per_img:.2f} ms/image（不含前後處理）")
    print(f"  偵測到 {len(_dets)} 個 person:")
    for d in _dets:
        print(f"    box={d['box']}  score={d['score']:.3f}")

    # 視覺化
    if SHOW_DISPLAY:
        _vis = _img_raw.copy()
        for d in _dets:
            bx1, by1, bx2, by2 = d['box']
            cv2.rectangle(_vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(_vis, f"person {d['score']:.2f}", (bx1, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Phase 2-B Raw : ORT Detection", _vis)
        print("  按任意鍵關閉視窗 ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =====================================================================
# %% [Phase 3-A] 匯出 TensorRT Engine（必須在目標機器上執行！）
"""
Phase 3-A - TensorRT Engine 匯出
----------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! 此 block 必須在每台目標機器上各自執行                   !!
!! .engine 檔案不可跨平台複製                              !!
!! RTX 4070 build 的 engine 無法在 Jetson 上執行           !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TensorRT build 過程（Ultralytics 底層自動完成）:
  1. PT -> ONNX -> TensorRT Parser
  2. Layer Fusion（Conv+BN+ReLU 合一）
  3. FP16 Kernel Selection（針對當前 GPU 架構選最快的 kernel）
  4. Engine 序列化為 .engine 二進位檔

匯出選項說明:
  half=True      -> FP16 量化，速度約 2x，Ampere / Ada 架構均支援
  simplify=True  -> 先透過 onnx-simplifier 精簡計算圖
  batch=1        -> Jetson 記憶體有限，固定 batch=1
  workspace=4    -> TRT builder 最大記憶體（GB），Jetson 建議改為 2

INT8 量化（更快，但需要校準資料集，選用）:
  pt_model.export(
      format='engine', int8=True, data='coco8.yaml',
      imgsz=640, device=0, half=False,
  )
"""

print("\n" + "=" * 60)
print("Phase 3-A : 匯出 TensorRT Engine")
print(f"  目標平台 : {PLATFORM}")
print("=" * 60)

if not HAS_CUDA:
    print("  警告: 未偵測到 CUDA，無法匯出 TensorRT engine")
    print("  請確認 NVIDIA 驅動、CUDA Toolkit 與 TensorRT 已安裝")
else:
    _engine_path = Path(MODEL_ENGINE)
    if _engine_path.exists():
        print(f"  已存在: {MODEL_ENGINE} ({_engine_path.stat().st_size / 1e6:.1f} MB)，跳過匯出")
        print("  若要在此機器上重新 build，請先刪除 .engine 檔案再執行此 block")
    else:
        # Windows 特有問題: trt.Builder() 需要 CUDA context 已存在才能初始化
        # 若 context 尚未建立會拋出 pybind11 factory returned nullptr
        # 解法: 在呼叫 export 前先執行一次 CUDA op 強制建立 context
        print("  初始化 CUDA context ...")
        _ctx_init = torch.zeros(1, device=f'cuda:{DEVICE}')
        torch.cuda.synchronize(DEVICE)
        del _ctx_init

        print("  開始 build（首次需要 2~10 分鐘，Jetson 可能需要更久）...")
        _t_export = time.perf_counter()
        pt_model.export(
            format='engine',
            imgsz=IMGSZ,
            half=True,       # FP16 量化（推薦，速度快且精度損失小）
            simplify=True,   # 先過 onnx-simplifier
            device=DEVICE,
            batch=1,
            workspace=4,     # Jetson 記憶體少，建議改為 2
        )
        _elapsed = time.perf_counter() - _t_export
        print(f"  Build 完成: {MODEL_ENGINE} ({_elapsed:.0f} 秒)")
        print(f"  檔案大小: {_engine_path.stat().st_size / 1e6:.1f} MB")
        print(f"  此 engine 只能在本機（{PLATFORM}）使用")


# =====================================================================
# %% [Phase 3-B] TensorRT 推理
"""
Phase 3-B - TensorRT 推理
---------------------------
載入 .engine 進行推理，Ultralytics 自動處理:
  - Engine 反序列化
  - CUDA 記憶體分配（固定記憶體 pinned memory）
  - FP16 輸入轉換
  - 後處理（NMS、座標縮放）

注意: 若 .engine 是在其他機器 build 的，此步驟會直接報錯。
      必須先在本機執行 Phase 3-A 才能使用此 block。
"""

print("\n" + "=" * 60)
print("Phase 3-B : TensorRT 推理")
print("=" * 60)

if not Path(MODEL_ENGINE).exists():
    print(f"  找不到 {MODEL_ENGINE}")
    print("  請先執行 Phase 3-A 在本機 build TensorRT engine")
elif not HAS_CUDA:
    print("  警告: TensorRT 需要 CUDA，目前環境無法執行")
else:
    trt_model = YOLO(MODEL_ENGINE)

    print(f"  Warmup {WARMUP_RUNS} runs ...")
    for _ in range(WARMUP_RUNS):
        trt_model.predict(
            SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
            imgsz=IMGSZ, verbose=False
        )

    pre_lst, inf_lst, post_lst = [], [], []
    print(f"  Measuring {MEASURE_RUNS} runs ...")
    for _ in range(MEASURE_RUNS):
        res = trt_model.predict(
            SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
            imgsz=IMGSZ, verbose=False
        )
        spd = res[0].speed
        pre_lst.append(spd['preprocess'])
        inf_lst.append(spd['inference'])
        post_lst.append(spd['postprocess'])

    BENCH['TensorRT FP16'] = {
        'pre' : float(np.mean(pre_lst)),
        'inf' : float(np.mean(inf_lst)),
        'post': float(np.mean(post_lst)),
    }
    _total = sum(BENCH['TensorRT FP16'].values())
    print(f"  Pre={BENCH['TensorRT FP16']['pre']:.1f}ms  "
          f"Inf={BENCH['TensorRT FP16']['inf']:.1f}ms  "
          f"Post={BENCH['TensorRT FP16']['post']:.1f}ms  "
          f"Total={_total:.1f}ms  FPS={1000 / _total:.1f}")

    # 顯示偵測結果
    if SHOW_DISPLAY:
        _res_trt = trt_model.predict(
            SOURCE, classes=PERSON_ONLY, conf=CONF_THRESH,
            imgsz=IMGSZ, verbose=False
        )
        _annotated_trt = _res_trt[0].plot()
        cv2.imshow("Phase 3-B : TensorRT Detection", _annotated_trt)
        print("  按任意鍵關閉視窗 ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =====================================================================
# %% [Phase 3-B Camera] TensorRT 攝影機即時推理（選用）
"""
Phase 3-B-Camera - TensorRT 攝影機推理
----------------------------------------
與 Phase 2-A Camera 相同流程，但使用 TensorRT engine。
可直接比較同一個攝影機源的 PT vs TRT FPS 差異。
"""

if Path(MODEL_ENGINE).exists() and HAS_CUDA and RUN_CAMERA:
    print("\n[攝影機] TensorRT 即時推理（RUN_CAMERA=True）...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"  無法開啟攝影機 {CAMERA_ID}")
    else:
        print("  推理中 ... 按 'q' 退出")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            cam_res = trt_model.predict(
                frame, classes=PERSON_ONLY, conf=CONF_THRESH,
                imgsz=IMGSZ, verbose=False
            )
            fps_val = 1.0 / (time.perf_counter() - t0)
            annotated_frame = cam_res[0].plot()
            cv2.putText(annotated_frame, f"TRT FPS: {fps_val:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Camera - TensorRT Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
else:
    print("  [Phase 3-B Camera] 略過（RUN_CAMERA=False 或 engine 未存在）")


# =====================================================================
# %% [Benchmark] 速度比較報告
"""
Benchmark - 速度比較報告
-------------------------
彙整各推理格式的速度，計算相對於 PyTorch baseline 的加速倍率。

指標說明:
  Pre  = 前處理時間 (ms)  圖片讀取、resize、normalize
  Inf  = 推理時間 (ms)    神經網路前向傳播
  Post = 後處理時間 (ms)  NMS、座標縮放
  FPS  = 1000 / (Pre + Inf + Post)
  加速  = PyTorch Total / 當前格式 Total
"""

print("\n" + "=" * 60)
print("Benchmark : 速度比較報告")
print("=" * 60)

if not BENCH:
    print("  尚未收集任何資料，請先執行 Phase 2-A 與 Phase 3-B")
else:
    _hdr = (f"  {'格式':<18} {'Pre(ms)':>8} {'Inf(ms)':>8} "
            f"{'Post(ms)':>9} {'Total(ms)':>10} {'FPS':>7} {'加速倍率':>9}")
    print(_hdr)
    print("  " + "-" * (len(_hdr) - 2))

    _baseline = None
    for name, data in BENCH.items():
        _total = data['pre'] + data['inf'] + data['post']
        _fps   = 1000 / _total
        if _baseline is None:
            _baseline = _total
            _speedup  = 1.0
        else:
            _speedup = _baseline / _total
        print(f"  {name:<18} {data['pre']:>8.1f} {data['inf']:>8.1f} "
              f"{data['post']:>9.1f} {_total:>10.1f} {_fps:>7.1f} {_speedup:>8.2f}x")

    print()
    print(f"  平台 : {PLATFORM}")
    if HAS_CUDA:
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  圖片 : {Path(SOURCE).name}  IMGSZ={IMGSZ}  Conf={CONF_THRESH}")


# =====================================================================
# %% [Jetson] Jetson Orin Nano 設定建議
"""
Jetson Orin Nano 部署前設定
------------------------------
在 Jetson 上執行本腳本前，建議完成以下設定。

效能預估（yolo11n，640x640）:
  PyTorch FP32  : ~15~25 FPS
  ONNX Runtime  : ~30~50 FPS
  TRT FP16      : ~60~100 FPS
  TRT INT8      : ~80~150 FPS

部署步驟:
  1. scp yolov11_tensorrt_jetson.py user@jetson:/home/user/
  2. ssh jetson
  3. pip install ultralytics onnxruntime-gpu
  4. python3 yolov11_tensorrt_jetson.py  (Phase 1~2 先測試)
  5. 設定最大效能後執行 Phase 3
"""

print("\n" + "=" * 60)
print("Jetson Orin Nano 部署建議")
print("=" * 60)

_jetson_cmds = [
    ("確認 JetPack 版本",   "cat /etc/nv_tegra_release"),
    ("解鎖最大效能",        "sudo nvpmodel -m 0 && sudo jetson_clocks"),
    ("確認 TensorRT",       'python3 -c "import tensorrt; print(tensorrt.__version__)"'),
    ("安裝 ultralytics",    "pip install ultralytics"),
    ("安裝 onnxruntime",    "pip install onnxruntime-gpu"),
    ("確認 CUDA",           'python3 -c "import torch; print(torch.cuda.is_available())"'),
]
for desc, cmd in _jetson_cmds:
    print(f"  [{desc}]")
    print(f"    $ {cmd}")

print()
print("  CSI Camera（Jetson 原廠攝影機）GStreamer Pipeline:")
_gst = (
    '"nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! '
    'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"'
)
print(f"    cap = cv2.VideoCapture({_gst}, cv2.CAP_GSTREAMER)")
print()
print("  完成設定後，在 Jetson 上重新執行 Phase 3-A 以 build TensorRT engine")
print("  Phase 3-A build 完成後的 .engine 只能在 Jetson 上使用，無法複製到 Windows")
