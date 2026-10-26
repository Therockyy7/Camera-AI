"""Tiện ích ảnh: cắt quanh khung mặt, thu nhỏ, encode JPEG, vẽ chú thích.

Font Hershey của OpenCV chỉ có ASCII nên chữ trên hình được bỏ dấu tiếng Việt.
"""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .observation import Box

# Màu BGR
GREEN = (80, 200, 80)
YELLOW = (0, 200, 255)
RED = (60, 60, 230)
GRAY = (160, 160, 160)


def crop_around(frame: np.ndarray, box: Box, scale: float = 2.0) -> np.ndarray:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half_w, half_h = w * scale / 2, h * scale / 2
    img_h, img_w = frame.shape[:2]
    x1, y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x2, y2 = min(img_w, int(cx + half_w)), min(img_h, int(cy + half_h))
    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    return frame[y1:y2, x1:x2].copy()


def fit_within(image: np.ndarray, max_side: int = 400) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Không encode được JPEG")
    return buf.tobytes()


def ascii_label(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return stripped.encode("ascii", "ignore").decode()


def draw_box(frame: np.ndarray, box: Box, label: str, color: tuple[int, int, int]) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = ascii_label(label)
    if not text:
        return
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top = max(0, y - th - baseline - 6)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, top + th + 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
