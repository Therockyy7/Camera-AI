"""Phát hiện (YuNet) + chống giả (MiniFASNet) + nhận diện (SFace / ArcFace) trên một frame.

Hai cổng lọc: cổng NHẬN DIỆN nới (camera trần nhìn mặt nhỏ, xiên) và cổng CHỐNG GIẢ chặt hơn (model chỉ tin cậy
với mặt to, gần thẳng). Mặt qua cổng nhận diện nhưng không qua cổng chống giả có liveness = None ("chưa kiểm").
Đối tượng OpenCV DNN không an toàn khi dùng chung giữa các thread: mỗi thread tạo FaceEngine riêng.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import model_store
from .config import AppConfig, LivenessConfig, RecognitionConfig
from .gallery import Gallery
from .liveness import LivenessChecker
from .observation import Box, FaceObservation
from .recognizers import Recognizer, make_recognizer

SHARPNESS_SIZE = 112
EDGE_MARGIN_PX = 5  # như hàm detect() của repo model chống giả mạo


def sharpness(frame: np.ndarray, box: Box) -> float:
    """Phương sai Laplacian trên ảnh mặt xám đưa về 112×112 (ít phụ thuộc khoảng cách)."""
    x, y, w, h = box
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    interpolation = cv2.INTER_AREA if gray.shape[1] > SHARPNESS_SIZE else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (SHARPNESS_SIZE, SHARPNESS_SIZE), interpolation=interpolation)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def touches_edge(box: Box, frame_shape, margin: int = EDGE_MARGIN_PX) -> bool:
    """Mặt chạm/tràn mép khung hình: điểm mốc lệch và crop chống giả bị lấp phần phản chiếu → bỏ qua."""
    x, y, w, h = box
    img_h, img_w = frame_shape[:2]
    return min(x, y, img_w - (x + w), img_h - (y + h)) < margin


def turn_offset(row: np.ndarray) -> float:
    """Độ quay ngang của mặt từ điểm mốc YuNet: 0 = nhìn thẳng, >= 0,5 = quay hẳn sang bên.

    Đo vị trí mũi giữa hai mắt (mắt phải của người nằm bên trái ảnh).
    """
    right_eye_x, left_eye_x, nose_x = float(row[4]), float(row[6]), float(row[8])
    span = left_eye_x - right_eye_x
    if span <= 0:
        return 1.0
    return abs((nose_x - right_eye_x) / span - 0.5)


def reject_reason(
    box: Box, turn: float, sharp: float, frame_shape, min_face_px: int, cfg: RecognitionConfig
) -> str | None:
    """Cổng lọc nhận diện. None = dùng được."""
    if min(box[2], box[3]) < min_face_px:
        return "small"
    if touches_edge(box, frame_shape):
        return "edge"
    if turn > cfg.max_turn:
        return "turned"
    if sharp < cfg.min_sharpness:
        return "blurry"
    return None


def liveness_checkable(box: Box, turn: float, cfg: LivenessConfig) -> bool:
    """Cổng lọc chống giả mạo: model MiniFASNet cần mặt >= 64 px và nhìn gần thẳng."""
    return min(box[2], box[3]) >= cfg.min_face_px and turn <= cfg.max_turn


class FaceEngine:
    def __init__(self, cfg: AppConfig, gallery: Gallery | None = None, recognizer: Recognizer | None = None):
        models_dir = cfg.models_path
        self._cfg = cfg
        self._gallery = gallery
        self._detector = cv2.FaceDetectorYN.create(
            str(model_store.model_path(models_dir, model_store.DETECTOR)), "", (320, 320),
            cfg.recognition.detect_score, 0.3, 5000,
        )
        self._recognizer = recognizer or make_recognizer(cfg)
        self._liveness = (
            LivenessChecker(model_store.model_path(models_dir, model_store.LIVENESS), cfg.liveness.threshold)
            if cfg.liveness.mode != "off"
            else None
        )

    @property
    def model_name(self) -> str:
        return self._recognizer.name

    @property
    def liveness_enabled(self) -> bool:
        return self._liveness is not None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Các dòng YuNet (x, y, w, h, 5 điểm mốc, score) theo toạ độ frame gốc."""
        img_h, img_w = frame.shape[:2]
        target = self._cfg.processing.detect_width
        scale, image = 1.0, frame
        if target and img_w > target:
            scale = target / img_w
            image = cv2.resize(frame, (target, max(1, round(img_h * scale))), interpolation=cv2.INTER_AREA)
        self._detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self._detector.detect(image)
        if faces is None:
            return np.zeros((0, 15), dtype=np.float32)
        faces = faces.astype(np.float32, copy=True)
        faces[:, :14] /= scale
        return faces

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        return self._recognizer.embed(frame, row)

    def analyze(self, frame: np.ndarray, min_face_px: int | None = None) -> list[FaceObservation]:
        min_px = self._cfg.recognition.min_face_px if min_face_px is None else min_face_px
        observations = []
        for row in self.detect(frame):
            box = tuple(int(round(v)) for v in row[:4])
            turn = turn_offset(row)
            obs = FaceObservation(box=box, det_score=float(row[14]), sharpness=sharpness(frame, box), turn=turn)
            obs.reject_reason = reject_reason(box, turn, obs.sharpness, frame.shape, min_px, self._cfg.recognition)
            if obs.reject_reason is None:
                if self._liveness is not None and liveness_checkable(box, turn, self._cfg.liveness):
                    obs.liveness = self._liveness.score(frame, box)
                    obs.is_real = self._liveness.is_real(obs.liveness)
                obs.embedding = self.embed(frame, row)
                if self._gallery is not None:
                    obs.match = self._gallery.match(obs.embedding)
            observations.append(obs)
        return observations
