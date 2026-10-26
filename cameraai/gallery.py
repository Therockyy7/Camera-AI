"""Danh sách embedding khuôn mặt đã đăng ký và so khớp cosine."""
from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    employee_id: int | None  # người khớp (điểm >= ngưỡng), None nếu không ai đạt
    score: float  # cosine cao nhất, kể cả khi dưới ngưỡng
    best_employee_id: int | None  # người gần nhất, dù dưới ngưỡng


def normalize(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


class Gallery:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._lock = threading.Lock()
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._owners = np.zeros(0, dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self._owners)

    def load(self, samples: Iterable[tuple[int, np.ndarray]]) -> None:
        items = list(samples)
        matrix = np.stack([normalize(e) for _, e in items]) if items else np.zeros((0, 0), dtype=np.float32)
        owners = np.array([employee_id for employee_id, _ in items], dtype=np.int64)
        with self._lock:
            self._matrix, self._owners = matrix, owners

    def match(self, embedding) -> Match:
        with self._lock:
            matrix, owners = self._matrix, self._owners
        if len(owners) == 0:
            return Match(None, 0.0, None)
        scores = matrix @ normalize(embedding)
        best = int(np.argmax(scores))
        score, employee_id = float(scores[best]), int(owners[best])
        return Match(employee_id if score >= self.threshold else None, score, employee_id)
