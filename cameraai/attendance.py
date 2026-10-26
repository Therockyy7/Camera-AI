"""Bám khuôn mặt qua các frame và quyết định khi nào ghi chấm công / từ chối.

Logic thuần, không phụ thuộc OpenCV — test được bằng dữ liệu giả.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import AttendanceConfig
from .observation import Box, FaceObservation


@dataclass
class Track:
    id: int
    box: Box
    last_seen: float
    history: deque[FaceObservation]
    done: set[str] = field(default_factory=set)  # các loại quyết định đã đưa ra cho track này
    employee_id: int | None = None  # nhân viên đã xác nhận trên track này


@dataclass(frozen=True)
class Decision:
    kind: str  # "attendance" | "spoof" | "unknown"
    track_id: int
    observation: FaceObservation  # quan sát đại diện, dùng để chụp ảnh
    employee_id: int | None  # người khớp (attendance) hoặc người gần nhất (spoof/unknown)
    score: float
    liveness: float | None


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    def __init__(self, window: int = 5, max_age_s: float = 1.5, iou_threshold: float = 0.3):
        self._window = window
        self._max_age_s = max_age_s
        self._iou_threshold = iou_threshold
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, observations: list[FaceObservation], now: float) -> list[tuple[Track, FaceObservation]]:
        """Gán mỗi quan sát vào track (IoU tham lam), tạo track mới nếu cần, xoá track quá hạn."""
        candidates = sorted(
            (
                (iou(track.box, obs.box), track_id, index)
                for track_id, track in self._tracks.items()
                for index, obs in enumerate(observations)
            ),
            reverse=True,
        )
        assigned: dict[int, Track] = {}
        used_tracks: set[int] = set()
        for overlap, track_id, index in candidates:
            if overlap < self._iou_threshold:
                break
            if track_id in used_tracks or index in assigned:
                continue
            used_tracks.add(track_id)
            assigned[index] = self._tracks[track_id]

        result = []
        for index, obs in enumerate(observations):
            track = assigned.get(index)
            if track is None:
                track = Track(self._next_id, obs.box, now, deque(maxlen=self._window))
                self._tracks[track.id] = track
                self._next_id += 1
            track.box = obs.box
            track.last_seen = now
            if obs.usable:
                track.history.append(obs)
            result.append((track, obs))

        expired = [track_id for track_id, track in self._tracks.items() if now - track.last_seen > self._max_age_s]
        for track_id in expired:
            del self._tracks[track_id]
        return result


class AttendanceDecider:
    def __init__(self, cfg: AttendanceConfig, liveness_mode: str, spoof_threshold: float = 0.3):
        self._cfg = cfg
        self._mode = liveness_mode  # off | opportunistic | required
        # Mặt thật đôi khi tụt điểm ở vài frame (ánh sáng, cử động); ảnh/màn hình cho điểm thấp liên tục.
        # Vì vậy chỉ ghi giả mạo khi đủ cửa sổ và điểm TRUNG BÌNH thấp hơn ngưỡng này.
        self._spoof_threshold = spoof_threshold
        self._last_recorded: dict[int, datetime] = {}

    def seed(self, last_recorded: dict[int, datetime]) -> None:
        """Nạp lần chấm gần nhất (từ DB) để cooldown đúng cả sau khi khởi động lại."""
        self._last_recorded.update(last_recorded)

    def last_recorded(self, employee_id: int) -> datetime | None:
        return self._last_recorded.get(employee_id)

    def looks_spoof(self, track: Track) -> bool:
        """Cửa sổ HIỆN TẠI có đủ frame ĐÃ KIỂM và điểm mặt thật trung bình dưới ngưỡng giả mạo."""
        if self._mode == "off":
            return False
        scores = [o.liveness for o in track.history if o.liveness is not None]
        return len(scores) >= self._cfg.window_frames and sum(scores) / len(scores) < self._spoof_threshold

    def _counts(self, obs: FaceObservation) -> bool:
        """Frame có được tính để bỏ phiếu chấm công không."""
        if self._mode == "required":
            return obs.is_real is True
        if self._mode == "opportunistic":
            return obs.is_real is not False  # chưa kiểm (None) vẫn tính
        return True

    def evaluate(self, track: Track, now: datetime) -> Decision | None:
        history = list(track.history)
        if not history:
            return None
        confirm = self._cfg.confirm_frames

        if "spoof" not in track.done and self.looks_spoof(track):
            track.done.add("spoof")
            rep = min(history, key=lambda o: o.liveness if o.liveness is not None else 1.0)
            return Decision("spoof", track.id, rep, _nearest(rep), _score(rep), rep.liveness)

        real = [o for o in history if self._counts(o)]
        if "attendance" not in track.done:
            votes = Counter(o.match.employee_id for o in real if o.match and o.match.employee_id is not None)
            if votes:
                employee_id, count = votes.most_common(1)[0]
                if count >= confirm:
                    track.done.add("attendance")
                    track.employee_id = employee_id
                    last = self._last_recorded.get(employee_id)
                    if last is not None and now - last < timedelta(minutes=self._cfg.cooldown_minutes):
                        return None
                    self._last_recorded[employee_id] = now
                    voters = [o for o in real if o.match and o.match.employee_id == employee_id]
                    best = max(voters, key=_score)
                    checked = [o.liveness for o in voters if o.liveness is not None]
                    liveness = sum(checked) / len(checked) if checked else None  # None = chưa kiểm được
                    return Decision("attendance", track.id, best, employee_id, _score(best), liveness)

        if (
            "unknown" not in track.done
            and track.employee_id is None
            and len(history) >= self._cfg.window_frames
            and len(real) == len(history)
            and all(o.match is not None and o.match.employee_id is None for o in history)
        ):
            track.done.add("unknown")
            rep = max(history, key=lambda o: o.det_score)
            return Decision("unknown", track.id, rep, _nearest(rep), _score(rep), rep.liveness)
        return None


def _score(obs: FaceObservation) -> float:
    return obs.match.score if obs.match else 0.0


def _nearest(obs: FaceObservation) -> int | None:
    return obs.match.best_employee_id if obs.match else None
