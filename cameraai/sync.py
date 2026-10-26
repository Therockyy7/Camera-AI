"""Chỗ cắm để gửi lần chấm công sang hệ thống ngoài (vd Fastdo Timekeeping).

Bản này chưa có sink thật nên SyncWorker không được khởi động. Nối Fastdo sau này = viết một lớp
có hàm send() theo AttendanceSink rồi khởi động SyncWorker trong __main__, không sửa phần nhận diện.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from .storage import Storage

log = logging.getLogger(__name__)


class AttendanceSink(Protocol):
    def send(self, events: list[dict]) -> list[int]:
        """Gửi các lần chấm (dict từ Storage.pending_events). Trả về id đã gửi thành công."""
        ...


class SyncWorker:
    def __init__(
        self, storage: Storage, sink: AttendanceSink, clock: Callable[[], datetime],
        interval_s: float = 30.0, batch_size: int = 100,
    ):
        self._storage = storage
        self._sink = sink
        self._clock = clock
        self._interval_s = interval_s
        self._batch_size = batch_size
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> int:
        events = self._storage.pending_events(self._batch_size)
        if not events:
            return 0
        try:
            sent = list(self._sink.send(events))
        except Exception:
            log.exception("Gửi lần chấm công thất bại, sẽ thử lại sau")
            return 0
        self._storage.mark_synced(sent, self._clock())
        return len(sent)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self._interval_s)
