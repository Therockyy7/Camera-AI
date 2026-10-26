"""Nguồn video (webcam / RTSP / file). Thread nền đọc liên tục, chỉ giữ frame mới nhất, tự kết nối lại.

Giữ frame mới nhất (thay vì xếp hàng) tránh trễ tích luỹ của RTSP khi xử lý chậm hơn tốc độ camera.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraConfig, SourceSpec, mask_source, parse_source

log = logging.getLogger(__name__)

# RTSP qua TCP ổn định hơn UDP khi mạng nội bộ mất gói.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

MAX_READ_FAILURES = 10


@dataclass(frozen=True)
class CameraStatus:
    connected: bool
    ended: bool
    source: str
    width: int
    height: int
    fps: float
    last_error: str | None


def open_capture(spec: SourceSpec, cfg: CameraConfig) -> cv2.VideoCapture:
    if spec.kind == "webcam":
        # DirectShow mở webcam trên Windows nhanh hơn backend mặc định (MSMF).
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(spec.value, backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if cfg.width and cfg.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        return cap
    if spec.kind == "stream":
        cap = cv2.VideoCapture(spec.value, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(spec.value)


class VideoSource:
    def __init__(
        self,
        cfg: CameraConfig,
        opener: Callable[[SourceSpec, CameraConfig], object] = open_capture,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
    ):
        self._cfg = cfg
        self._spec = parse_source(cfg.source)
        self._opener = opener
        self._initial_backoff_s = initial_backoff_s
        self._max_backoff_s = max_backoff_s
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._connected = False
        self._ended = False
        self._fps = 0.0
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_file(self) -> bool:
        return self._spec.kind == "file"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="video-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._frame_id, self._frame

    def status(self) -> CameraStatus:
        with self._lock:
            height, width = self._frame.shape[:2] if self._frame is not None else (0, 0)
            return CameraStatus(
                self._connected, self._ended, mask_source(self._cfg.source),
                width, height, round(self._fps, 1), self._last_error,
            )

    def _run(self) -> None:
        backoff = self._initial_backoff_s
        while not self._stop.is_set():
            cap = None
            try:
                cap = self._opener(self._spec, self._cfg)
                if cap is not None and cap.isOpened():
                    backoff = self._initial_backoff_s
                    self._read_loop(cap)
                else:
                    self._set_error(f"Không mở được nguồn video {mask_source(self._cfg.source)}")
            except Exception as exc:  # lỗi bất ngờ từ backend: ghi lại rồi thử lại
                log.exception("Lỗi nguồn video")
                self._set_error(str(exc))
            finally:
                if cap is not None:
                    cap.release()
                with self._lock:
                    self._connected = False
            if self.is_file:
                with self._lock:
                    self._ended = True
                return
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self._max_backoff_s)

    def _read_loop(self, cap) -> None:
        interval = 0.0
        if self.is_file:  # phát file theo đúng fps của file
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = 1.0 / fps if fps and fps > 0 else 1.0 / 25
        with self._lock:
            self._connected = True
            self._last_error = None
        log.info("Đã kết nối nguồn video %s", mask_source(self._cfg.source))
        failures, counted, window_start = 0, 0, time.monotonic()
        while not self._stop.is_set():
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.is_file:
                    return
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    self._set_error("Mất tín hiệu camera (đọc frame lỗi liên tiếp)")
                    return
                time.sleep(0.05)
                continue
            failures = 0
            counted += 1
            now = time.monotonic()
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                if now - window_start >= 1.0:
                    self._fps = counted / (now - window_start)
                    counted, window_start = 0, now
            if interval:
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)

    def _set_error(self, message: str) -> None:
        log.warning(message)
        with self._lock:
            self._last_error = message
