"""Đăng ký mẫu khuôn mặt: chụp từ camera đang chạy hoặc từ ảnh tải lên.

Chụp phía server (lấy frame mới nhất của VideoSource) vì trên Windows webcam bị khoá cho một tiến trình,
trình duyệt không mở song song được.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from .config import AppConfig
from .gallery import normalize
from .imaging import crop_around, encode_jpeg, fit_within
from .observation import FaceObservation
from .storage import Storage

log = logging.getLogger(__name__)

REASON_TEXT = {
    "no_face": "Không thấy khuôn mặt",
    "many_faces": "Có nhiều hơn một khuôn mặt",
    "small": "Mặt quá nhỏ — lại gần camera hơn",
    "edge": "Mặt bị cắt ở mép khung hình — lùi ra hoặc vào giữa",
    "blurry": "Ảnh mờ — giữ yên",
    "turned": "Mặt quay nghiêng — nhìn thẳng camera",
    "low_score": "Khuôn mặt không rõ",
    "spoof": "Nghi mặt giả (ảnh / màn hình)",
    "not_image": "Không đọc được file ảnh",
    "no_camera": "Chưa có hình từ camera",
    "similar": "Giống mẫu đã có — xoay đầu sang góc khác",
}


class EnrollmentError(ValueError):
    pass


@dataclass
class CaptureResult:
    saved: int
    attempts: int
    reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "saved": self.saved,
            "attempts": self.attempts,
            "reasons": [{"code": c, "count": n, "text": REASON_TEXT.get(c, c)} for c, n in self.reasons.items()],
        }


class EnrollmentService:
    def __init__(
        self,
        cfg: AppConfig,
        engine_factory: Callable[[], object],
        source,
        storage: Storage,
        on_change: Callable[[], None],
        clock: Callable[[], datetime],
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._cfg = cfg
        self._engine_factory = engine_factory
        self._engine = None
        self._source = source
        self._storage = storage
        self._on_change = on_change
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def _engine_instance(self):
        if self._engine is None:  # tạo lười: model chỉ nạp khi lần đầu cần
            self._engine = self._engine_factory()
        return self._engine

    def check_frame(self, frame: np.ndarray, require_real: bool) -> tuple[FaceObservation | None, str | None]:
        faces = self._engine_instance().analyze(frame, min_face_px=self._cfg.enrollment.min_face_px)
        if not faces:
            return None, "no_face"
        large = [f for f in faces if f.reject_reason != "small"]  # mặt nhỏ = người ở hậu cảnh, không tính
        if not large:
            return None, "small"
        if len(large) > 1:
            return None, "many_faces"
        face = large[0]
        if face.reject_reason:
            return None, face.reject_reason
        if face.det_score < self._cfg.enrollment.min_det_score:
            return None, "low_score"
        if require_real and face.is_real is False:
            return None, "spoof"
        return face, None

    def capture(self, employee_id: int) -> CaptureResult:
        self._require_employee(employee_id)
        cfg = self._cfg.enrollment
        reasons: Counter[str] = Counter()
        saved = attempts = 0
        last_id = -1
        with self._lock:
            model = self._engine_instance().model_name
            existing = [normalize(e) for e in self._storage.employee_embeddings(employee_id, model)]
            deadline = time.monotonic() + cfg.capture_timeout_s
            while saved < cfg.samples_per_capture and time.monotonic() < deadline:
                frame_id, frame = self._source.latest()
                if frame is None or frame_id == last_id:
                    self._sleep(0.1)
                    continue
                last_id = frame_id
                attempts += 1
                face, reason = self.check_frame(frame, require_real=True)
                if reason is None:
                    embedding = normalize(face.embedding)
                    # Chỉ nhận mẫu khác đủ so với mọi mẫu đã có → mẫu trải nhiều góc khi người dùng xoay đầu.
                    if existing and max(float(e @ embedding) for e in existing) >= cfg.max_similarity:
                        reason = "similar"
                if reason:
                    reasons[reason] += 1
                else:
                    self._save(employee_id, frame, face, "camera")
                    existing.append(embedding)
                    saved += 1
                self._sleep(0.3)
        if attempts == 0:
            reasons["no_camera"] += 1
        if saved:
            self._on_change()
        return CaptureResult(saved, attempts, dict(reasons))

    def add_upload(self, employee_id: int, data: bytes) -> str | None:
        """Lưu mẫu từ ảnh tải lên. Trả về None nếu thành công, ngược lại là mã lý do (xem REASON_TEXT)."""
        self._require_employee(employee_id)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if image is None:
            return "not_image"
        with self._lock:
            face, reason = self.check_frame(image, require_real=False)
            if reason:
                return reason
            self._save(employee_id, image, face, "upload")
        self._on_change()
        return None

    def reindex(self) -> int:
        """Tính embedding còn thiếu cho model đang dùng từ ảnh mẫu đã lưu (sau khi đổi model).

        Trả về số mẫu đã tính. Mẫu không có ảnh hoặc không phát hiện lại được mặt bị bỏ qua (ghi log).
        """
        with self._lock:
            engine = self._engine_instance()
            done = 0
            for sample in self._storage.samples_missing_embedding(engine.model_name):
                path = self._storage.resolve_file(sample["image_path"]) if sample["image_path"] else None
                image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if path else None
                rows = engine.detect(image) if image is not None else []
                if len(rows) == 0:
                    log.warning("Mẫu #%s: không tính được embedding %s (thiếu ảnh hoặc không thấy mặt)",
                                sample["id"], engine.model_name)
                    continue
                row = max(rows, key=lambda r: r[2])
                self._storage.add_embedding(sample["id"], engine.model_name, engine.embed(image, row))
                done += 1
        if done:
            log.info("Đã tính embedding %s cho %d mẫu", engine.model_name, done)
        return done

    def _save(self, employee_id: int, frame: np.ndarray, face: FaceObservation, source: str) -> None:
        jpeg = encode_jpeg(fit_within(crop_around(frame, face.box, 1.6)))
        model = self._engine_instance().model_name
        self._storage.add_sample(employee_id, {model: face.embedding}, jpeg, source, self._clock())

    def _require_employee(self, employee_id: int) -> None:
        if self._storage.get_employee(employee_id) is None:
            raise EnrollmentError("Không tìm thấy nhân viên")
