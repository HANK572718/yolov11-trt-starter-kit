# TensorRT 部署後的實際後處理範例
"""
本文件展示如何直接從 TensorRT 模型輸出取得結果

使用場景:
1. 使用 TensorRT Python API 直接推理
2. 使用 ONNX Runtime with TensorRT EP
3. 使用 C++/CUDA 部署時的參考
"""

import numpy as np
import cv2
from typing import Tuple, List


class TensorRTDetectionPostProcessor:
    """
    TensorRT Detection 模型後處理器

    模型輸出格式:
    - YOLO v8/v11 Detection: [batch, 84, 8400]
      - 84 = 4 (bbox) + 80 (classes)
      - 8400 = 80x80 + 40x40 + 20x20 的預測點數量
      - 需要轉置為 [batch, 8400, 84]
    """

    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 num_classes: int = 80, input_shape: Tuple[int, int] = (640, 640)):
        """
        Args:
            conf_threshold: 信心度閾值
            iou_threshold: NMS IOU 閾值
            num_classes: 類別數量
            input_shape: 模型輸入尺寸 (height, width)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.input_shape = input_shape

    def __call__(self, output: np.ndarray, original_shape: Tuple[int, int]) -> dict:
        """
        後處理 TensorRT 輸出

        Args:
            output: numpy array, shape (1, 84, 8400) 或 (1, 8400, 84)
            original_shape: 原始圖片尺寸 (height, width)

        Returns:
            dict:
                'boxes': numpy array, shape (N, 4) - [x1, y1, x2, y2] 原圖座標
                'scores': numpy array, shape (N,) - 信心分數
                'class_ids': numpy array, shape (N,) - 類別 ID
        """
        # 確保輸出形狀為 [batch, num_predictions, bbox+classes]
        if output.shape[1] == 84:  # [1, 84, 8400]
            output = output.transpose(0, 2, 1)  # [1, 8400, 84]

        predictions = output[0]  # [8400, 84]

        # 分離邊界框和類別分數
        boxes_xywh = predictions[:, :4]  # [8400, 4] - [x_center, y_center, width, height]
        class_scores = predictions[:, 4:]  # [8400, 80]

        # 獲取每個預測的最高類別分數和類別 ID
        max_scores = np.max(class_scores, axis=1)  # [8400]
        class_ids = np.argmax(class_scores, axis=1)  # [8400]

        # 過濾低信心度的預測
        mask = max_scores > self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return {'boxes': np.array([]), 'scores': np.array([]), 'class_ids': np.array([])}

        # 轉換 xywh (中心點格式) 為 xyxy
        boxes_xyxy = self._xywh2xyxy(boxes_xywh)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            boxes_xyxy = boxes_xyxy[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]

            # 縮放座標到原圖尺寸
            boxes_xyxy = self._scale_boxes(boxes_xyxy, self.input_shape, original_shape)

        return {
            'boxes': boxes_xyxy,
            'scores': scores,
            'class_ids': class_ids
        }

    def _xywh2xyxy(self, boxes_xywh: np.ndarray) -> np.ndarray:
        """
        轉換中心點格式為左上右下格式

        Args:
            boxes_xywh: [N, 4] - [x_center, y_center, width, height]

        Returns:
            boxes_xyxy: [N, 4] - [x1, y1, x2, y2]
        """
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        return boxes_xyxy

    def _scale_boxes(self, boxes: np.ndarray, from_shape: Tuple[int, int],
                    to_shape: Tuple[int, int]) -> np.ndarray:
        """
        縮放座標從輸入尺寸到原圖尺寸

        Args:
            boxes: [N, 4] - [x1, y1, x2, y2] 在 from_shape 中的座標
            from_shape: (height, width) - 模型輸入尺寸
            to_shape: (height, width) - 原圖尺寸

        Returns:
            boxes: [N, 4] - 縮放後的座標
        """
        # 計算縮放比例 (letterbox 考慮)
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])

        # 計算 padding
        pad_h = (from_shape[0] - to_shape[0] * gain) / 2
        pad_w = (from_shape[1] - to_shape[1] * gain) / 2

        # 移除 padding 並縮放
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain  # x 座標
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain  # y 座標

        # 限制在圖片範圍內
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, to_shape[1])  # x
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, to_shape[0])  # y

        return boxes


class TensorRTSegmentationPostProcessor:
    """
    TensorRT Segmentation 模型後處理器

    模型輸出格式:
    - 輸出 0: [batch, 116, 8400] - detection + mask coefficients
      - 116 = 4 (bbox) + 80 (classes) + 32 (mask coeffs)
    - 輸出 1: [batch, 32, 160, 160] - mask prototypes
    """

    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 num_classes: int = 80, input_shape: Tuple[int, int] = (640, 640)):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.det_processor = TensorRTDetectionPostProcessor(
            conf_threshold, iou_threshold, num_classes, input_shape
        )

    def __call__(self, det_output: np.ndarray, proto_output: np.ndarray,
                original_shape: Tuple[int, int]) -> dict:
        """
        後處理 Segmentation 輸出

        Args:
            det_output: [1, 116, 8400] - detection + mask coefficients
            proto_output: [1, 32, 160, 160] - mask prototypes
            original_shape: (height, width) - 原圖尺寸

        Returns:
            dict:
                'boxes': [N, 4]
                'scores': [N]
                'class_ids': [N]
                'masks': [N, H, W] - 原圖尺寸的 binary masks
        """
        # 轉置輸出
        if det_output.shape[1] == 116:
            det_output = det_output.transpose(0, 2, 1)  # [1, 8400, 116]

        predictions = det_output[0]  # [8400, 116]

        # 分離 detection 和 mask coefficients
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:84]
        mask_coeffs = predictions[:, 84:]  # [8400, 32]

        # Detection 後處理
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores > self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]
        mask_coeffs = mask_coeffs[mask]  # [N, 32]

        if len(boxes_xywh) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'class_ids': np.array([]),
                'masks': np.array([])
            }

        # 轉換座標
        boxes_xyxy = self.det_processor._xywh2xyxy(boxes_xywh)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )

        if len(indices) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'class_ids': np.array([]),
                'masks': np.array([])
            }

        indices = indices.flatten()
        boxes_xyxy = boxes_xyxy[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        mask_coeffs = mask_coeffs[indices]  # [N, 32]

        # 生成 masks
        proto = proto_output[0]  # [32, 160, 160]
        masks = self._process_masks(mask_coeffs, proto, boxes_xyxy, original_shape)

        # 縮放座標
        boxes_xyxy = self.det_processor._scale_boxes(boxes_xyxy, self.input_shape, original_shape)

        return {
            'boxes': boxes_xyxy,
            'scores': scores,
            'class_ids': class_ids,
            'masks': masks
        }

    def _process_masks(self, mask_coeffs: np.ndarray, proto: np.ndarray,
                      boxes: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        從 mask coefficients 和 prototypes 生成最終 masks

        Args:
            mask_coeffs: [N, 32] - mask 係數
            proto: [32, 160, 160] - mask 原型
            boxes: [N, 4] - 邊界框 (用於裁剪)
            original_shape: (height, width) - 原圖尺寸

        Returns:
            masks: [N, H, W] - binary masks
        """
        # 矩陣乘法: [N, 32] @ [32, 160*160] = [N, 160*160]
        c, mh, mw = proto.shape
        masks = (mask_coeffs @ proto.reshape(c, -1)).reshape(-1, mh, mw)  # [N, 160, 160]

        # Sigmoid
        masks = 1 / (1 + np.exp(-masks))

        # 上採樣到輸入尺寸
        masks_resized = []
        for mask in masks:
            mask_resized = cv2.resize(mask, self.input_shape[::-1])  # (W, H)
            masks_resized.append(mask_resized)
        masks = np.array(masks_resized)  # [N, 640, 640]

        # 縮放到原圖尺寸
        gain = min(self.input_shape[0] / original_shape[0], self.input_shape[1] / original_shape[1])
        pad_h = (self.input_shape[0] - original_shape[0] * gain) / 2
        pad_w = (self.input_shape[1] - original_shape[1] * gain) / 2

        # 裁剪 padding 區域
        top = int(pad_h)
        left = int(pad_w)
        bottom = int(self.input_shape[0] - pad_h)
        right = int(self.input_shape[1] - pad_w)

        masks = masks[:, top:bottom, left:right]

        # 縮放到原圖尺寸
        masks_final = []
        for mask in masks:
            mask_final = cv2.resize(mask, (original_shape[1], original_shape[0]))
            masks_final.append(mask_final > 0.5)  # 二值化
        masks = np.array(masks_final)

        return masks


class TensorRTPosePostProcessor:
    """
    TensorRT Pose 模型後處理器

    模型輸出格式:
    - [batch, 56, 8400] - detection + keypoints
      - 56 = 4 (bbox) + 1 (person class) + 51 (17 keypoints * 3)
    """

    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 input_shape: Tuple[int, int] = (640, 640)):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape

    def __call__(self, output: np.ndarray, original_shape: Tuple[int, int]) -> dict:
        """
        後處理 Pose 輸出

        Args:
            output: [1, 56, 8400]
            original_shape: (height, width)

        Returns:
            dict:
                'boxes': [N, 4]
                'scores': [N]
                'keypoints': [N, 17, 3] - [x, y, confidence]
        """
        if output.shape[1] == 56:
            output = output.transpose(0, 2, 1)  # [1, 8400, 56]

        predictions = output[0]  # [8400, 56]

        # 分離組件
        boxes_xywh = predictions[:, :4]  # [8400, 4]
        scores = predictions[:, 4]  # [8400]
        keypoints = predictions[:, 5:].reshape(-1, 17, 3)  # [8400, 17, 3]

        # 過濾
        mask = scores > self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        keypoints = keypoints[mask]

        if len(boxes_xywh) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'keypoints': np.array([])
            }

        # 轉換座標
        boxes_xyxy = self._xywh2xyxy(boxes_xywh)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )

        if len(indices) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'keypoints': np.array([])
            }

        indices = indices.flatten()
        boxes_xyxy = boxes_xyxy[indices]
        scores = scores[indices]
        keypoints = keypoints[indices]

        # 縮放座標
        boxes_xyxy = self._scale_boxes(boxes_xyxy, self.input_shape, original_shape)
        keypoints = self._scale_keypoints(keypoints, self.input_shape, original_shape)

        return {
            'boxes': boxes_xyxy,
            'scores': scores,
            'keypoints': keypoints
        }

    def _xywh2xyxy(self, boxes_xywh: np.ndarray) -> np.ndarray:
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        return boxes_xyxy

    def _scale_boxes(self, boxes: np.ndarray, from_shape: Tuple[int, int],
                    to_shape: Tuple[int, int]) -> np.ndarray:
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad_h = (from_shape[0] - to_shape[0] * gain) / 2
        pad_w = (from_shape[1] - to_shape[1] * gain) / 2

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, to_shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, to_shape[0])

        return boxes

    def _scale_keypoints(self, keypoints: np.ndarray, from_shape: Tuple[int, int],
                        to_shape: Tuple[int, int]) -> np.ndarray:
        """
        縮放關鍵點座標

        Args:
            keypoints: [N, 17, 3] - [x, y, conf]
            from_shape: (height, width)
            to_shape: (height, width)

        Returns:
            keypoints: [N, 17, 3] - 縮放後的關鍵點
        """
        gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
        pad_h = (from_shape[0] - to_shape[0] * gain) / 2
        pad_w = (from_shape[1] - to_shape[1] * gain) / 2

        keypoints[..., 0] = (keypoints[..., 0] - pad_w) / gain  # x
        keypoints[..., 1] = (keypoints[..., 1] - pad_h) / gain  # y

        return keypoints


# ==================== 使用範例 ====================

def example_tensorrt_detection():
    """
    Detection 模型 TensorRT 推理範例
    """
    # 假設已經有 TensorRT 引擎的輸出
    # 這裡用隨機數據模擬
    batch_size = 1
    num_predictions = 8400
    num_classes = 80

    # 模擬 TensorRT 輸出 (實際應從引擎獲取)
    raw_output = np.random.randn(batch_size, 4 + num_classes, num_predictions).astype(np.float32)

    # 創建後處理器
    postprocessor = TensorRTDetectionPostProcessor(
        conf_threshold=0.25,
        iou_threshold=0.45,
        num_classes=80,
        input_shape=(640, 640)
    )

    # 假設原圖尺寸
    original_shape = (1080, 1920)

    # 後處理
    results = postprocessor(raw_output, original_shape)

    print("Detection 結果:")
    print(f"  檢測到 {len(results['boxes'])} 個物件")
    for i in range(len(results['boxes'])):
        x1, y1, x2, y2 = results['boxes'][i]
        score = results['scores'][i]
        cls_id = results['class_ids'][i]
        print(f"  物件 {i}: 類別 {cls_id}, 信心度 {score:.2f}, "
              f"座標 ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")


if __name__ == "__main__":
    print("TensorRT 後處理範例")
    print("=" * 60)
    example_tensorrt_detection()
