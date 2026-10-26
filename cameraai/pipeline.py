"""Vòng xử lý nền: frame mới nhất → FaceEngine → tracker/decider → lưu; giữ frame có chú thích cho web."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timedelta

import numpy as np

from .attendance import AttendanceDecider, Decision, FaceTracker, Track
from .config import AppConfig
from .gallery import Gallery
from .imaging import GRAY, GREEN, RED, YELLOW, crop_around, draw_box, encode_jpeg, fit_within
from .observation import FaceObservation
from .storage import Storage

log = logging.getLogger(__name__)

REASON_LABELS = {
    "small": "Lại gần hơn",
    "edge": "Lùi ra / vào giữa khung hình",
    "blurry": "Giữ yên",
    "turned": "Nhìn thẳng camera",
}


class Pipeline:
    def __init__(
        self, cfg: AppConfig, source, engine, storage: Storage, gallery: Gallery, clock: Callable[[], datetime]
    ):
        self._cfg = cfg
        self._source = source
        self._engine = engine
        self._storage = storage
        self._gallery = gallery
        self._clock = clock
        self._tracker = FaceTracker(window=cfg.attendance.window_frames)
        self._decider = AttendanceDecider(cfg.attendance, cfg.liveness.mode, cfg.liveness.spoof_threshold)
        self._names: dict[int, str] = {}
        self._recent: deque[dict] = deque(maxlen=20)
        self._jpeg: bytes | None = None
        self._jpeg_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._processed_fps = 0.0
        self._last_cleanup: date | None = None

    # ---------- vòng đời ----------
    def prime(self) -> None:
        """Nạp gallery và lần chấm gần nhất từ DB (để cooldown đúng sau khi khởi động lại)."""
        since = self._clock() - timedelta(minutes=self._cfg.attendance.cooldown_minutes)
        self._decider.seed(self._storage.last_attendance_times(self._cfg.camera.id, since))
        self.reload_gallery()

    def start(self) -> None:
        self.prime()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def reload_gallery(self) -> None:
        self._gallery.load(self._storage.gallery_samples(self._engine.model_name))
        self._names = {e["id"]: e["name"] for e in self._storage.list_employees()}
        log.info("Đã nạp %d mẫu khuôn mặt của %d nhân viên", self._gallery.size, len(self._names))

    # ---------- cho web ----------
    def latest_jpeg(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._jpeg_id, self._jpeg

    def recent_events(self) -> list[dict]:
        with self._lock:
            return list(self._recent)

    @property
    def processed_fps(self) -> float:
        return round(self._processed_fps, 1)

    # ---------- xử lý ----------
    def _run(self) -> None:
        interval = 1.0 / max(self._cfg.processing.fps, 0.1)
        last_id = -1
        counted, window_start = 0, time.monotonic()
        while not self._stop.is_set():
            started = time.monotonic()
            frame_id, frame = self._source.latest()
            if frame is None or frame_id == last_id:
                self._stop.wait(0.02)
                continue
            last_id = frame_id
            try:
                self.process(frame, self._clock())
                self._maybe_cleanup()
            except Exception:
                log.exception("Lỗi xử lý frame — bỏ qua frame này")
            counted += 1
            now = time.monotonic()
            if now - window_start >= 2.0:
                self._processed_fps = counted / (now - window_start)
                counted, window_start = 0, now
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def process(self, frame: np.ndarray, now: datetime) -> list[Decision]:
        observations = self._engine.analyze(frame)
        pairs = self._tracker.update(observations, now.timestamp())
        decisions = []
        for track, _ in pairs:
            decision = self._decider.evaluate(track, now)
            if decision is not None:
                self._record(frame, decision, now)
                decisions.append(decision)
        annotated = frame.copy()
        for track, obs in pairs:
            label, color = self._label(track, obs)
            draw_box(annotated, obs.box, label, color)
        jpeg = encode_jpeg(annotated, quality=75)
        with self._lock:
            self._jpeg = jpeg
            self._jpeg_id += 1
        return decisions

    def _record(self, frame: np.ndarray, decision: Decision, now: datetime) -> None:
        snapshot = encode_jpeg(fit_within(crop_around(frame, decision.observation.box, 2.0)))
        name = self._names.get(decision.employee_id, "") if decision.employee_id is not None else ""
        camera_id = self._cfg.camera.id
        if decision.kind == "attendance":
            self._storage.add_attendance(
                decision.employee_id, camera_id, now, decision.score, decision.liveness, snapshot
            )
            log.info("Chấm công: %s (điểm khớp %.2f)", name, decision.score)
        else:
            self._storage.add_rejected(
                camera_id, now, decision.kind, decision.employee_id, decision.score, decision.liveness, snapshot
            )
            log.info("Từ chối (%s), gần nhất: %s (điểm %.2f)", decision.kind, name or "-", decision.score)
        with self._lock:
            self._recent.appendleft({
                "kind": decision.kind,
                "employee_id": decision.employee_id,
                "name": name,
                "ts": now.isoformat(timespec="seconds"),
                "score": round(decision.score, 3),
                "liveness": None if decision.liveness is None else round(decision.liveness, 3),
            })

    def _label(self, track: Track, obs: FaceObservation) -> tuple[str, tuple[int, int, int]]:
        if not obs.usable:
            return REASON_LABELS.get(obs.reject_reason or "", ""), GRAY
        # Nhãn theo cửa sổ hiện tại, không theo cờ đã ghi: một lần bị nghi giả (vd lúc mới bước vào)
        # không được che trạng thái đã chấm công về sau.
        if self._decider.looks_spoof(track):
            return "Nghi giả mạo", RED
        if track.employee_id is not None:
            last = self._decider.last_recorded(track.employee_id)
            suffix = f" - đã chấm {last:%H:%M}" if last else ""
            return f"{self._names.get(track.employee_id, '?')}{suffix}", GREEN
        if obs.is_real is False:
            return "Đang kiểm tra...", YELLOW
        if obs.match is not None and obs.match.employee_id is not None:
            return f"{self._names.get(obs.match.employee_id, '?')}...", YELLOW
        if "unknown" in track.done:
            return "Người lạ", RED
        return "Đang nhận diện...", YELLOW

    def _maybe_cleanup(self) -> None:
        now = self._clock()
        if self._last_cleanup == now.date():
            return
        self._last_cleanup = now.date()
        cutoff = now - timedelta(days=self._cfg.storage.snapshot_retention_days)
        removed = self._storage.cleanup(cutoff)
        if removed:
            log.info("Đã xoá %d ảnh chụp cũ hơn %s", removed, cutoff.date())
