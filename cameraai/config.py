"""Cấu hình: đọc config.yaml rồi cho biến môi trường ghi đè."""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class CameraConfig:
    source: str = "0"
    id: str = "laptop"
    width: int = 1280
    height: int = 720


@dataclass
class ProcessingConfig:
    fps: float = 8.0
    detect_width: int = 640


RECOGNITION_MODELS = ("sface", "arcface")
LIVENESS_MODES = ("off", "opportunistic", "required")


@dataclass
class RecognitionConfig:
    model: str = "sface"
    sface_threshold: float = 0.363
    arcface_threshold: float = 0.30
    detect_score: float = 0.8
    # Cổng lọc để NHẬN DIỆN — nới hơn cổng chống giả mạo vì camera trần nhìn mặt nhỏ và xiên.
    min_face_px: int = 40
    min_sharpness: float = 12.0
    max_turn: float = 0.45  # đo trên webcam: quay 0,30–0,45 vẫn nhận ra (trung vị cosine 0,82)


@dataclass
class LivenessConfig:
    mode: str = "opportunistic"  # off | opportunistic (kiểm khi đủ điều kiện) | required
    threshold: float = 0.5
    spoof_threshold: float = 0.3
    # Model chống giả chỉ tin cậy khi mặt đủ lớn và gần thẳng; ngoài vùng này frame là "chưa kiểm".
    min_face_px: int = 64
    max_turn: float = 0.25


@dataclass
class AttendanceConfig:
    window_frames: int = 5
    confirm_frames: int = 3
    cooldown_minutes: float = 5.0


@dataclass
class EnrollmentConfig:
    samples_per_capture: int = 8
    capture_timeout_s: float = 15.0
    min_face_px: int = 80
    min_det_score: float = 0.9
    max_similarity: float = 0.92  # mẫu mới phải khác mọi mẫu cũ (cosine nhỏ hơn) để trải nhiều góc


@dataclass
class StorageConfig:
    data_dir: str = "data"
    snapshot_retention_days: int = 30


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    admin_password: str = ""


@dataclass
class AppConfig:
    timezone: str = "Asia/Ho_Chi_Minh"
    models_dir: str = "models"
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    liveness: LivenessConfig = field(default_factory=LivenessConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    enrollment: EnrollmentConfig = field(default_factory=EnrollmentConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @property
    def data_path(self) -> Path:
        return _resolve(self.storage.data_dir)

    @property
    def models_path(self) -> Path:
        return _resolve(self.models_dir)


# Biến môi trường -> (mục, khoá). Giá trị rỗng bị bỏ qua.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "CAMERA_SOURCE": ("camera", "source"),
    "CAMERA_ID": ("camera", "id"),
    "CAMERAAI_HOST": ("web", "host"),
    "CAMERAAI_PORT": ("web", "port"),
    "ADMIN_PASSWORD": ("web", "admin_password"),
    "CAMERAAI_DATA_DIR": ("storage", "data_dir"),
    "LIVENESS_MODE": ("liveness", "mode"),
    "RECOGNITION_MODEL": ("recognition", "model"),
}

CHOICES: dict[tuple[str, str], tuple[str, ...]] = {
    ("recognition", "model"): RECOGNITION_MODELS,
    ("liveness", "mode"): LIVENESS_MODES,
}


def load_config(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> AppConfig:
    env = os.environ if env is None else env
    cfg = AppConfig()
    config_path = Path(path) if path else ROOT_DIR / "config.yaml"
    if path and not config_path.exists():
        raise FileNotFoundError(f"Không thấy file cấu hình: {config_path}")
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        _apply(cfg, raw, "")
    for name, (section, key) in ENV_OVERRIDES.items():
        value = env.get(name)
        if value:
            _set_field(getattr(cfg, section), key, value, f"{section}.{key}")
    for (section, key), allowed in CHOICES.items():
        value = getattr(getattr(cfg, section), key)
        if value not in allowed:
            raise ValueError(f"Giá trị không hợp lệ cho {section}.{key}: {value!r} (chọn một trong {', '.join(allowed)})")
    return cfg


def _apply(target: Any, raw: Any, prefix: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Mục cấu hình '{prefix.rstrip('.')}' phải là bảng khoá/giá trị")
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if not hasattr(target, key):
            raise ValueError(f"Khoá cấu hình không hợp lệ: {name}")
        if is_dataclass(getattr(target, key)):
            _apply(getattr(target, key), value or {}, f"{name}.")
        else:
            _set_field(target, key, value, name)


def _set_field(target: Any, key: str, value: Any, name: str) -> None:
    if not hasattr(target, key):
        raise ValueError(f"Khoá cấu hình không hợp lệ: {name}")
    current = getattr(target, key)
    try:
        if isinstance(current, bool):
            converted: Any = _to_bool(value)
        elif isinstance(current, int):
            converted = int(value)
        elif isinstance(current, float):
            converted = float(value)
        elif isinstance(value, bool):  # YAML 1.1 đọc `off`/`on` thành bool
            converted = "on" if value else "off"
        else:
            converted = "" if value is None else str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Giá trị không hợp lệ cho {name}: {value!r}") from exc
    setattr(target, key, converted)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(value)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # "webcam" | "stream" | "file"
    value: int | str


def parse_source(source: str | int) -> SourceSpec:
    text = str(source).strip()
    if text.isdigit():
        return SourceSpec("webcam", int(text))
    if re.match(r"^(rtsps?|https?)://", text, re.IGNORECASE):
        return SourceSpec("stream", text)
    return SourceSpec("file", text)


def mask_source(source: str | int) -> str:
    """Che mật khẩu trong URL camera trước khi hiển thị hoặc ghi log."""
    return re.sub(r"(://[^:/@]+):[^@/]*@", r"\1:***@", str(source))


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def make_clock(timezone: str) -> Callable[[], datetime]:
    tz = ZoneInfo(timezone)
    return lambda: datetime.now(tz).replace(microsecond=0)
