"""Bộ nhận diện khuôn mặt đổi được: SFace (mặc định) và ArcFace (chỉ để đánh giá — xem model_store).

Mỗi bộ nhận một dòng YuNet (x, y, w, h, 5 điểm mốc, score) theo toạ độ frame gốc và trả embedding.
Embedding của hai model khác nhau (128 vs 512 chiều) nên mẫu đăng ký lưu embedding theo tên model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from . import model_store
from .config import AppConfig

# Toạ độ 5 điểm (mắt phải, mắt trái, mũi, khoé miệng phải, khoé miệng trái) trên ảnh 112×112 — template của
# ArcFace (InsightFace), cũng là template OpenCV dùng trong FaceRecognizerSF.alignCrop.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


class Recognizer(Protocol):
    name: str
    dim: int
    threshold: float

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray: ...


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Ma trận 2×3 của phép đồng dạng (xoay + tỉ lệ + tịnh tiến) bình phương tối thiểu đưa src về dst (Umeyama)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mean_src, dst - mean_dst
    u, s, vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    d = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1, 1] = -1
    rotation = u @ d @ vt
    scale = np.trace(np.diag(s) @ d) / ((src_c ** 2).sum() / len(src))
    shift = mean_dst - scale * rotation @ mean_src
    return np.hstack([scale * rotation, shift[:, None]]).astype(np.float32)


def align_112(frame: np.ndarray, row: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(row[4:14], np.float32).reshape(5, 2)
    return cv2.warpAffine(frame, similarity_transform(landmarks, ARCFACE_TEMPLATE), (112, 112), borderValue=0)


class SFaceRecognizer:
    name = "sface"
    dim = 128

    def __init__(self, models_dir: Path, threshold: float):
        self.threshold = threshold
        self._model = cv2.FaceRecognizerSF.create(
            str(model_store.model_path(models_dir, model_store.RECOGNIZER)), ""
        )

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        return self._model.feature(self._model.alignCrop(frame, row)).reshape(-1).copy()


class ArcFaceRecognizer:
    name = "arcface"
    dim = 512

    def __init__(self, models_dir: Path, threshold: float):
        import onnxruntime as ort

        self.threshold = threshold
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self._session = ort.InferenceSession(
            str(model_store.model_path(models_dir, model_store.ARCFACE)),
            sess_options=options, providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        # Graph không tự chuẩn hoá (lớp đầu là Conv) → (x - 127.5) / 127.5, ảnh RGB, như InsightFace.
        blob = cv2.dnn.blobFromImage(
            align_112(frame, row), 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        return self._session.run(None, {self._input_name: blob})[0].reshape(-1).copy()


def threshold_for(cfg: AppConfig, name: str) -> float:
    return cfg.recognition.arcface_threshold if name == "arcface" else cfg.recognition.sface_threshold


def make_recognizer(cfg: AppConfig, name: str | None = None) -> Recognizer:
    name = name or cfg.recognition.model
    if name == "sface":
        return SFaceRecognizer(cfg.models_path, threshold_for(cfg, name))
    if name == "arcface":
        return ArcFaceRecognizer(cfg.models_path, threshold_for(cfg, name))
    raise ValueError(f"Không có bộ nhận diện '{name}'")
