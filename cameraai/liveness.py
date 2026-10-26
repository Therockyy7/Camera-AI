"""Chống giả mạo bằng MiniFASNetV2 (ONNX).

Tiền xử lý bám đúng mã gốc của repo model (facenox/face-antispoof-onnx): crop vuông ×1,5 quanh khung
mặt (thiếu biên thì pad phản chiếu), ảnh RGB, letterbox 128×128, chia 255, CHW float32.
Đầu ra là 2 logit [real, spoof]; xác suất mặt thật = sigmoid(real - spoof).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = 128
BBOX_EXPANSION = 1.5


def crop_square(image: np.ndarray, box, expansion: float = BBOX_EXPANSION) -> np.ndarray:
    x, y, w, h = box
    size = int(max(w, h) * expansion)
    if w <= 0 or h <= 0 or size <= 0:
        raise ValueError("Khung mặt không hợp lệ")
    x1 = int(x + w / 2 - size / 2)
    y1 = int(y + h / 2 - size / 2)
    img_h, img_w = image.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(img_w, x1 + size), min(img_h, y1 + size)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("Khung mặt nằm ngoài ảnh")
    crop = image[cy1:cy2, cx1:cx2]
    return cv2.copyMakeBorder(
        crop, cy1 - y1, y1 + size - cy2, cx1 - x1, x1 + size - cx2, cv2.BORDER_REFLECT_101
    )


def to_input(face_rgb: np.ndarray) -> np.ndarray:
    h, w = face_rgb.shape[:2]
    ratio = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(face_rgb, (new_w, new_h), interpolation=interpolation)
    dh, dw = INPUT_SIZE - new_h, INPUT_SIZE - new_w
    img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_REFLECT_101)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def real_probability(real_logit: float, spoof_logit: float) -> float:
    diff = real_logit - spoof_logit
    if diff >= 0:
        return 1.0 / (1.0 + math.exp(-diff))
    e = math.exp(diff)
    return e / (1.0 + e)


class LivenessChecker:
    def __init__(self, model_path: Path, threshold: float = 0.5):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.threshold = threshold

    def score(self, frame_bgr: np.ndarray, box) -> float:
        """Xác suất khuôn mặt trong box là mặt thật (0..1)."""
        face = cv2.cvtColor(crop_square(frame_bgr, box), cv2.COLOR_BGR2RGB)
        logits = self._session.run(None, {self._input_name: to_input(face)[np.newaxis]})[0][0]
        return real_probability(float(logits[0]), float(logits[1]))

    def is_real(self, probability: float) -> bool:
        return probability >= self.threshold
