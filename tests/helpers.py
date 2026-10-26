"""Dữ liệu giả dùng chung cho test."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from cameraai.camera import CameraStatus
from cameraai.gallery import Match
from cameraai.observation import FaceObservation

TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def at(hour: int, minute: int = 0, second: int = 0, day: int = 26) -> datetime:
    return datetime(2026, 10, day, hour, minute, second, tzinfo=TZ)


def make_obs(
    employee_id: int | None = 1,
    *,
    score: float = 0.7,
    real: bool | None = True,
    box=(100, 100, 120, 120),
    det_score: float = 0.95,
    reject_reason: str | None = None,
    nearest: int | None = None,
    liveness: float | None = None,
) -> FaceObservation:
    if liveness is not None:
        real = liveness >= 0.5
    elif real is not None:
        liveness = 0.9 if real else 0.1
    return FaceObservation(
        box=box,
        det_score=det_score,
        sharpness=100.0,
        reject_reason=reject_reason,
        liveness=liveness,
        is_real=real,
        embedding=np.ones(128, dtype=np.float32),
        match=Match(employee_id, score, employee_id if employee_id is not None else nearest),
    )


class FakeEngine:
    def __init__(self, faces=None, liveness_enabled: bool = True, model_name: str = "sface", rows=None):
        self.faces = list(faces or [])
        self.liveness_enabled = liveness_enabled
        self.model_name = model_name
        self.rows = np.array([[10, 10, 50, 50] + [0] * 10 + [0.99]], np.float32) if rows is None else rows

    def analyze(self, frame, min_face_px=None):
        return list(self.faces)

    def detect(self, frame):
        return self.rows

    def embed(self, frame, row):
        return np.full(128, 0.5, np.float32)


class VaryingEngine(FakeEngine):
    """Mỗi frame trả 1 mặt với embedding ngẫu nhiên (các mẫu khác nhau, như khi xoay đầu)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(0)

    def analyze(self, frame, min_face_px=None):
        face = make_obs(None)
        face.embedding = self._rng.standard_normal(128).astype(np.float32)
        return [face]


class FakeSource:
    """Mỗi lần gọi latest() trả về một frame mới (empty=True: chưa có hình)."""

    def __init__(self, empty: bool = False):
        self.frame = None if empty else np.zeros((480, 640, 3), np.uint8)
        self.frame_id = 0

    def latest(self):
        if self.frame is None:
            return self.frame_id, None
        self.frame_id += 1
        return self.frame_id, self.frame

    def status(self):
        return CameraStatus(True, False, "0", 640, 480, 30.0, None)
