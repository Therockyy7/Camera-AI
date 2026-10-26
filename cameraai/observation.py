"""Kết quả phân tích một khuôn mặt trên một frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gallery import Match

Box = tuple[int, int, int, int]  # x, y, w, h trên frame gốc


@dataclass
class FaceObservation:
    box: Box
    det_score: float
    sharpness: float = 0.0
    turn: float = 0.0  # độ quay ngang (0 = nhìn thẳng)
    reject_reason: str | None = None  # "small" | "edge" | "turned" | "blurry"
    liveness: float | None = None  # xác suất mặt thật; None = chưa kiểm (tắt, hoặc mặt nhỏ/nghiêng)
    is_real: bool | None = None
    embedding: np.ndarray | None = None
    match: Match | None = None

    @property
    def usable(self) -> bool:
        return self.reject_reason is None
