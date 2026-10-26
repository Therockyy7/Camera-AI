"""Lưu trữ SQLite + file ảnh trong thư mục data/.

Giờ lưu dạng ISO 8601 kèm offset của múi giờ cấu hình; vì cùng một offset nên so sánh chuỗi = so sánh thời gian.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS face_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    image_path TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sample_embeddings (
    sample_id INTEGER NOT NULL REFERENCES face_samples(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (sample_id, model)
);
CREATE TABLE IF NOT EXISTS attendance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    camera_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    score REAL NOT NULL,
    liveness_score REAL,
    snapshot_path TEXT,
    synced_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_attendance_ts ON attendance_events(ts);
CREATE TABLE IF NOT EXISTS rejected_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    reason TEXT NOT NULL,
    best_employee_id INTEGER,
    score REAL,
    liveness_score REAL,
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS ix_rejected_ts ON rejected_events(ts);
"""

EMPLOYEE_SELECT = """
SELECT e.id, e.code, e.name, e.created_at, COUNT(s.id) AS sample_count
FROM employees e LEFT JOIN face_samples s ON s.employee_id = e.id
"""

ATTENDANCE_SELECT = """
SELECT a.id, a.employee_id, e.code, e.name, a.camera_id, a.ts, a.score, a.liveness_score,
       a.snapshot_path, a.synced_at
FROM attendance_events a JOIN employees e ON e.id = a.employee_id
"""

FILE_DIRS = ("snapshots", "faces")
SCHEMA_VERSION = 1


class DuplicateEmployeeError(ValueError):
    pass


class Storage:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        for sub in FILE_DIRS:
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.data_dir / "cameraai.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        """v0 → v1: embedding (SFace) nằm trong face_samples → bảng sample_embeddings theo model."""
        if self._db.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
            return
        with self._db:
            columns = [r["name"] for r in self._db.execute("PRAGMA table_info(face_samples)")]
            if "embedding" in columns:
                self._db.execute(
                    "INSERT OR IGNORE INTO sample_embeddings(sample_id, model, embedding) "
                    "SELECT id, 'sface', embedding FROM face_samples"
                )
                self._db.execute("ALTER TABLE face_samples DROP COLUMN embedding")
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---------- nhân viên ----------
    def add_employee(self, code: str, name: str, now: datetime) -> dict:
        code, name = code.strip(), name.strip()
        if not code or not name:
            raise ValueError("Cần nhập mã và tên nhân viên")
        with self._lock, self._db:
            row = self._db.execute("SELECT id, active FROM employees WHERE code = ?", (code,)).fetchone()
            if row and row["active"]:
                raise DuplicateEmployeeError(f"Mã nhân viên {code} đã tồn tại")
            if row:  # mã từng bị xoá: kích hoạt lại, giữ lịch sử cũ
                self._db.execute("UPDATE employees SET name = ?, active = 1 WHERE id = ?", (name, row["id"]))
                employee_id = row["id"]
            else:
                employee_id = self._db.execute(
                    "INSERT INTO employees(code, name, active, created_at) VALUES (?, ?, 1, ?)",
                    (code, name, _iso(now)),
                ).lastrowid
        return self.get_employee(employee_id)

    def get_employee(self, employee_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute(
                EMPLOYEE_SELECT + " WHERE e.id = ? AND e.active = 1 GROUP BY e.id", (employee_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_employees(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(EMPLOYEE_SELECT + " WHERE e.active = 1 GROUP BY e.id ORDER BY e.code").fetchall()
        return [dict(r) for r in rows]

    def deactivate_employee(self, employee_id: int) -> bool:
        """Ngừng chấm công: giữ lịch sử, xoá cứng mẫu khuôn mặt (dữ liệu sinh trắc)."""
        with self._lock, self._db:
            changed = self._db.execute(
                "UPDATE employees SET active = 0 WHERE id = ? AND active = 1", (employee_id,)
            ).rowcount
            if not changed:
                return False
            paths = [
                r["image_path"]
                for r in self._db.execute("SELECT image_path FROM face_samples WHERE employee_id = ?", (employee_id,))
            ]
            self._db.execute("DELETE FROM face_samples WHERE employee_id = ?", (employee_id,))
        for path in paths:
            self._delete_file(path)
        return True

    # ---------- mẫu khuôn mặt ----------
    def add_sample(
        self, employee_id: int, embeddings: Mapping[str, np.ndarray], image_jpeg: bytes | None,
        source: str, now: datetime,
    ) -> int:
        """Lưu mẫu + embedding của (các) model đã tính. Model khác được tính sau từ ảnh mẫu."""
        image_path = self._write_file(f"faces/{employee_id}", image_jpeg) if image_jpeg else None
        with self._lock, self._db:
            sample_id = self._db.execute(
                "INSERT INTO face_samples(employee_id, image_path, source, created_at) VALUES (?, ?, ?, ?)",
                (employee_id, image_path, source, _iso(now)),
            ).lastrowid
            self._db.executemany(
                "INSERT INTO sample_embeddings(sample_id, model, embedding) VALUES (?, ?, ?)",
                [(sample_id, model, _blob(e)) for model, e in embeddings.items()],
            )
        return sample_id

    def add_embedding(self, sample_id: int, model: str, embedding: np.ndarray) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO sample_embeddings(sample_id, model, embedding) VALUES (?, ?, ?)",
                (sample_id, model, _blob(embedding)),
            )

    def samples_missing_embedding(self, model: str) -> list[dict]:
        """Mẫu (của nhân viên đang hoạt động) chưa có embedding cho model này."""
        with self._lock:
            rows = self._db.execute(
                "SELECT s.id, s.employee_id, s.image_path FROM face_samples s "
                "JOIN employees e ON e.id = s.employee_id "
                "WHERE e.active = 1 AND NOT EXISTS "
                "(SELECT 1 FROM sample_embeddings m WHERE m.sample_id = s.id AND m.model = ?) ORDER BY s.id",
                (model,),
            ).fetchall()
        return [dict(r) for r in rows]

    def employee_embeddings(self, employee_id: int, model: str) -> list[np.ndarray]:
        with self._lock:
            rows = self._db.execute(
                "SELECT m.embedding FROM sample_embeddings m JOIN face_samples s ON s.id = m.sample_id "
                "WHERE s.employee_id = ? AND m.model = ?",
                (employee_id, model),
            ).fetchall()
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]

    def list_samples(self, employee_id: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, image_path, source, created_at FROM face_samples WHERE employee_id = ? ORDER BY id",
                (employee_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_sample(self, sample_id: int) -> bool:
        with self._lock, self._db:
            row = self._db.execute("SELECT image_path FROM face_samples WHERE id = ?", (sample_id,)).fetchone()
            if row is None:
                return False
            self._db.execute("DELETE FROM face_samples WHERE id = ?", (sample_id,))
        self._delete_file(row["image_path"])
        return True

    def gallery_samples(self, model: str) -> list[tuple[int, np.ndarray]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.employee_id, m.embedding FROM sample_embeddings m "
                "JOIN face_samples s ON s.id = m.sample_id JOIN employees e ON e.id = s.employee_id "
                "WHERE e.active = 1 AND m.model = ?",
                (model,),
            ).fetchall()
        return [(r["employee_id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows]

    # ---------- sự kiện ----------
    def add_attendance(
        self, employee_id: int, camera_id: str, ts: datetime, score: float,
        liveness: float | None, snapshot_jpeg: bytes | None,
    ) -> int:
        path = self._write_file(f"snapshots/{ts.date().isoformat()}", snapshot_jpeg) if snapshot_jpeg else None
        with self._lock, self._db:
            return self._db.execute(
                "INSERT INTO attendance_events(employee_id, camera_id, ts, score, liveness_score, snapshot_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (employee_id, camera_id, _iso(ts), score, liveness, path),
            ).lastrowid

    def add_rejected(
        self, camera_id: str, ts: datetime, reason: str, best_employee_id: int | None,
        score: float | None, liveness: float | None, snapshot_jpeg: bytes | None,
    ) -> int:
        path = self._write_file(f"snapshots/{ts.date().isoformat()}", snapshot_jpeg) if snapshot_jpeg else None
        with self._lock, self._db:
            return self._db.execute(
                "INSERT INTO rejected_events(camera_id, ts, reason, best_employee_id, score, liveness_score, snapshot_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (camera_id, _iso(ts), reason, best_employee_id, score, liveness, path),
            ).lastrowid

    def attendance_on(self, day: date) -> list[dict]:
        return self.attendance_between(day, day)

    def attendance_between(self, start: date, end: date) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                ATTENDANCE_SELECT + " WHERE a.ts >= ? AND a.ts < ? ORDER BY a.ts", _day_range(start, end)
            ).fetchall()
        return [dict(r) for r in rows]

    def daily_summary(self, day: date) -> list[dict]:
        """Mỗi nhân viên: lần đầu (giờ vào), lần cuối (giờ ra), số lần chấm trong ngày."""
        with self._lock:
            rows = self._db.execute(
                "SELECT e.id AS employee_id, e.code, e.name, MIN(a.ts) AS first_ts, MAX(a.ts) AS last_ts, "
                "COUNT(*) AS count FROM attendance_events a JOIN employees e ON e.id = a.employee_id "
                "WHERE a.ts >= ? AND a.ts < ? GROUP BY e.id ORDER BY first_ts",
                _day_range(day, day),
            ).fetchall()
        return [dict(r) for r in rows]

    def rejected_on(self, day: date) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT r.*, e.code AS best_code, e.name AS best_name FROM rejected_events r "
                "LEFT JOIN employees e ON e.id = r.best_employee_id "
                "WHERE r.ts >= ? AND r.ts < ? ORDER BY r.ts DESC",
                _day_range(day, day),
            ).fetchall()
        return [dict(r) for r in rows]

    def last_attendance_times(self, camera_id: str, since: datetime) -> dict[int, datetime]:
        with self._lock:
            rows = self._db.execute(
                "SELECT employee_id, MAX(ts) AS ts FROM attendance_events "
                "WHERE camera_id = ? AND ts >= ? GROUP BY employee_id",
                (camera_id, _iso(since)),
            ).fetchall()
        return {r["employee_id"]: datetime.fromisoformat(r["ts"]) for r in rows}

    # ---------- đồng bộ ra ngoài ----------
    def pending_events(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT a.id, a.employee_id, e.code AS employee_code, e.name AS employee_name, a.camera_id, "
                "a.ts, a.score, a.liveness_score FROM attendance_events a JOIN employees e ON e.id = a.employee_id "
                "WHERE a.synced_at IS NULL ORDER BY a.id LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_synced(self, event_ids: Iterable[int], now: datetime) -> None:
        ids = list(event_ids)
        if not ids:
            return
        with self._lock, self._db:
            self._db.executemany(
                "UPDATE attendance_events SET synced_at = ? WHERE id = ?", [(_iso(now), i) for i in ids]
            )

    # ---------- dọn dẹp + file ----------
    def cleanup(self, cutoff: datetime) -> int:
        """Xoá ảnh chụp và sự kiện bị từ chối cũ hơn cutoff. Lần chấm công giữ lại, chỉ bỏ ảnh."""
        limit = _iso(cutoff)
        with self._lock, self._db:
            paths = [
                r["snapshot_path"]
                for table in ("attendance_events", "rejected_events")
                for r in self._db.execute(
                    f"SELECT snapshot_path FROM {table} WHERE ts < ? AND snapshot_path IS NOT NULL", (limit,)
                )
            ]
            self._db.execute("UPDATE attendance_events SET snapshot_path = NULL WHERE ts < ?", (limit,))
            self._db.execute("DELETE FROM rejected_events WHERE ts < ?", (limit,))
        for path in paths:
            self._delete_file(path)
        return len(paths)

    def resolve_file(self, rel_path: str) -> Path | None:
        """Đường dẫn file ảnh trong data/snapshots hoặc data/faces; None nếu ra ngoài hoặc không tồn tại."""
        path = (self.data_dir / rel_path).resolve()
        if not path.is_relative_to(self.data_dir) or not path.is_file():
            return None
        if path.relative_to(self.data_dir).parts[0] not in FILE_DIRS:
            return None
        return path

    def _write_file(self, subdir: str, data: bytes) -> str:
        folder = self.data_dir / subdir
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex[:12]}.jpg"
        (folder / name).write_bytes(data)
        return f"{subdir}/{name}"

    def _delete_file(self, rel_path: str | None) -> None:
        path = self.resolve_file(rel_path) if rel_path else None
        if path is None:
            return
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()  # chỉ xoá được khi thư mục ngày / nhân viên đã rỗng
        except OSError:
            pass


def _blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def _day_range(start: date, end: date) -> tuple[str, str]:
    return start.isoformat(), (end + timedelta(days=1)).isoformat()
