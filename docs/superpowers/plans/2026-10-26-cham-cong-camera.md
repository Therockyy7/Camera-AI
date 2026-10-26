# Chấm công bằng camera (CameraAI) — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** Ứng dụng Python chạy trên laptop (webcam) nhận diện khuôn mặt để ghi lần chấm công, đổi sang IP camera RTSP chỉ bằng cấu hình.

**Kiến trúc:** Thread đọc camera giữ frame mới nhất → thread pipeline chạy YuNet (phát hiện) + MiniFASNet (chống giả) + SFace (nhận diện) → tracker/decider logic thuần quyết định ghi → SQLite + ảnh trong `data/`. FastAPI phục vụ trang quản trị HTML/JS thuần, MJPEG, API JSON.

**Công nghệ:** Python 3.12, opencv-python-headless 4.12, onnxruntime 1.22 (CPU), FastAPI + uvicorn, SQLite (stdlib), pytest. Spec: `docs/superpowers/specs/2026-10-26-cham-cong-camera-design.md`.

**Quy ước:** người dùng tự commit (không tự chạy `git commit`); `D:\CameraAI` chưa là git repo. Mọi lệnh chạy trong PowerShell tại `D:\CameraAI`, gọi thẳng `.\.venv\Scripts\python.exe` (không cần activate venv).

> **Ghi chú:** code nhúng trong plan này là trạng thái sau đợt 1 (webcam laptop). Đợt hỗ trợ camera trần nhiều góc đã đổi config, storage, face_engine, attendance, enrollment… — xem `2026-10-26-camera-tran-nhieu-goc.md`.

---

## Cấu trúc file

| File | Trách nhiệm |
|---|---|
| `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore` | Môi trường |
| `config.yaml` | Cấu hình mặc định cho laptop |
| `cameraai/config.py` | Dataclass cấu hình, đọc YAML + env, parse/che nguồn video, đồng hồ theo múi giờ |
| `cameraai/logsetup.py` | Console UTF-8 + logging |
| `cameraai/model_store.py` | Danh sách model ghim commit + SHA256, tải/kiểm |
| `cameraai/gallery.py` | `Gallery`, `Match`: so khớp cosine |
| `cameraai/observation.py` | `FaceObservation` |
| `cameraai/liveness.py` | Tiền xử lý + `LivenessChecker` (ONNX) |
| `cameraai/attendance.py` | `FaceTracker`, `AttendanceDecider`, `Decision` (logic thuần) |
| `cameraai/storage.py` | `Storage` SQLite + file ảnh |
| `cameraai/sync.py` | `AttendanceSink` (Protocol), `SyncWorker` |
| `cameraai/camera.py` | `VideoSource`, `CameraStatus` |
| `cameraai/imaging.py` | Cắt/thu nhỏ/encode ảnh, vẽ chú thích |
| `cameraai/face_engine.py` | `FaceEngine`: YuNet + SFace + liveness trên 1 frame |
| `cameraai/pipeline.py` | `Pipeline`: vòng xử lý nền, ghi sự kiện, frame chú thích |
| `cameraai/enrollment.py` | `EnrollmentService`: đăng ký từ camera / ảnh tải lên |
| `cameraai/web/app.py`, `cameraai/web/static/*` | FastAPI + giao diện |
| `cameraai/__main__.py` | Lắp ráp và chạy |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`, `README.md` | Triển khai |
| `tests/helpers.py`, `tests/test_*.py` | Test |

---

## Task 0: Môi trường và model

**Files:** Create `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`, `cameraai/__init__.py`, `cameraai/logsetup.py`

**`requirements.txt`**

```text
opencv-python-headless==4.12.0.88
onnxruntime==1.22.1
numpy>=2.0,<2.3
fastapi==0.116.1
uvicorn==0.35.0
python-multipart==0.0.20
PyYAML==6.0.2
tzdata==2025.2
```
**`requirements-dev.txt`**

```text
-r requirements.txt
pytest==8.4.1
httpx==0.28.1
```
**`pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = . tests
addopts = -q
```
**`.gitignore`**

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/
models/*.onnx
models/*.part
.env
```
**`cameraai/__init__.py`**

```python
"""CameraAI — chấm công bằng nhận diện khuôn mặt qua camera."""

__version__ = "0.1.0"
```
**`cameraai/logsetup.py`**

```python
"""Console UTF-8 (tránh lỗi in tiếng Việt khi stdout bị chuyển hướng trên Windows) + logging."""
from __future__ import annotations

import logging
import sys


def setup_console(level: int = logging.INFO) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
```

- [ ] **Step 1:** Tạo venv và cài thư viện:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```
Expected: cài xong không lỗi.

## Task 1: Cấu hình

**Files:** Create `cameraai/config.py`, `config.yaml`; Test `tests/test_config.py`

- [ ] **Step 1: Viết test (đỏ)**
**`tests/test_config.py`**

```python
import pytest

from cameraai.config import AppConfig, is_loopback, load_config, make_clock, mask_source, parse_source


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_file_empty(tmp_path):
    assert load_config(write(tmp_path, ""), env={}) == AppConfig()


def test_yaml_then_env_override(tmp_path):
    path = write(tmp_path, "camera:\n  source: 0\n  id: cong-a\nprocessing:\n  fps: 5\nliveness:\n  enabled: false\n")
    env = {
        "CAMERA_SOURCE": "rtsp://u:p@10.0.0.2/s",
        "LIVENESS_ENABLED": "true",
        "CAMERAAI_PORT": "9000",
        "ADMIN_PASSWORD": "",
    }
    cfg = load_config(path, env=env)
    assert cfg.camera.source == "rtsp://u:p@10.0.0.2/s"
    assert cfg.camera.id == "cong-a"
    assert cfg.processing.fps == 5.0 and isinstance(cfg.processing.fps, float)
    assert cfg.liveness.enabled is True
    assert cfg.web.port == 9000
    assert cfg.web.admin_password == ""


def test_yaml_number_source_becomes_string(tmp_path):
    assert load_config(write(tmp_path, "camera:\n  source: 1\n"), env={}).camera.source == "1"


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="camera.sourse"):
        load_config(write(tmp_path, "camera:\n  sourse: 0\n"), env={})


def test_bad_value_rejected(tmp_path):
    with pytest.raises(ValueError, match="processing.fps"):
        load_config(write(tmp_path, "processing:\n  fps: nhanh\n"), env={})


def test_missing_explicit_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "khong-co.yaml", env={})


@pytest.mark.parametrize(
    "raw, kind, value",
    [
        ("0", "webcam", 0),
        (" 2 ", "webcam", 2),
        ("rtsp://a:b@1.2.3.4:554/Streaming/Channels/102", "stream", "rtsp://a:b@1.2.3.4:554/Streaming/Channels/102"),
        ("RTSP://cam/x", "stream", "RTSP://cam/x"),
        ("http://cam/video.mjpg", "stream", "http://cam/video.mjpg"),
        ("D:/videos/clip.mp4", "file", "D:/videos/clip.mp4"),
    ],
)
def test_parse_source(raw, kind, value):
    spec = parse_source(raw)
    assert (spec.kind, spec.value) == (kind, value)


def test_mask_source():
    assert mask_source("rtsp://admin:Abc123@192.168.1.64:554/x") == "rtsp://admin:***@192.168.1.64:554/x"
    assert mask_source("rtsp://192.168.1.64:554/x") == "rtsp://192.168.1.64:554/x"
    assert mask_source("0") == "0"


def test_is_loopback():
    assert is_loopback("127.0.0.1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0")


def test_clock_uses_timezone():
    now = make_clock("Asia/Ho_Chi_Minh")()
    assert now.utcoffset().total_seconds() == 7 * 3600
    assert now.microsecond == 0
```
- [ ] **Step 2:** `.\.venv\Scripts\python.exe -m pytest tests/test_config.py` → FAIL (`ModuleNotFoundError: cameraai.config`)
- [ ] **Step 3: Cài đặt**
**`cameraai/config.py`**

```python
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


@dataclass
class RecognitionConfig:
    threshold: float = 0.363
    detect_score: float = 0.8
    min_face_px: int = 64
    min_sharpness: float = 25.0
    max_turn: float = 0.25


@dataclass
class LivenessConfig:
    enabled: bool = True
    threshold: float = 0.5
    spoof_threshold: float = 0.3


@dataclass
class AttendanceConfig:
    window_frames: int = 5
    confirm_frames: int = 3
    cooldown_minutes: float = 5.0


@dataclass
class EnrollmentConfig:
    samples_per_capture: int = 5
    capture_timeout_s: float = 10.0
    min_face_px: int = 80
    min_det_score: float = 0.9


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
    "LIVENESS_ENABLED": ("liveness", "enabled"),
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
```
**`config.yaml`**

```yaml
# Cấu hình CameraAI. Biến môi trường ghi đè được: CAMERA_SOURCE, CAMERA_ID, CAMERAAI_HOST,
# CAMERAAI_PORT, ADMIN_PASSWORD, CAMERAAI_DATA_DIR, LIVENESS_ENABLED.
timezone: Asia/Ho_Chi_Minh
models_dir: models

camera:
  # "0" = webcam laptop. IP camera: rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102
  # Hoặc đường dẫn file video .mp4 để phát lại.
  source: "0"
  id: laptop            # tên camera, ghi vào mỗi lần chấm
  width: 1280           # chỉ áp dụng cho webcam
  height: 720

processing:
  fps: 8                # số frame xử lý mỗi giây
  detect_width: 640     # thu nhỏ frame về bề rộng này để phát hiện mặt (nhanh hơn)

recognition:
  threshold: 0.363      # ngưỡng cosine SFace; tăng để chặt hơn (ít nhận nhầm, dễ không nhận ra)
  detect_score: 0.8     # độ tin cậy tối thiểu của bộ phát hiện mặt
  min_face_px: 64       # cạnh mặt tối thiểu (pixel trên frame gốc); model chống giả cần >= 64
  min_sharpness: 25     # độ nét tối thiểu (phương sai Laplacian)
  max_turn: 0.25        # độ quay ngang tối đa (0 = nhìn thẳng); mặt nghiêng hơn bị bỏ qua

liveness:
  enabled: true         # chống giả mạo bằng ảnh / màn hình
  threshold: 0.5        # frame có xác suất mặt thật >= ngưỡng này mới được tính để chấm công
  spoof_threshold: 0.3  # ghi "giả mạo" khi điểm TRUNG BÌNH 5 frame gần nhất thấp hơn ngưỡng này

attendance:
  window_frames: 5      # số quan sát gần nhất xét cho mỗi khuôn mặt
  confirm_frames: 3     # cần ít nhất chừng này quan sát khớp cùng một người
  cooldown_minutes: 5   # không ghi lại cùng người trong khoảng này

enrollment:
  samples_per_capture: 5
  capture_timeout_s: 10
  min_face_px: 80
  min_det_score: 0.9

storage:
  data_dir: data
  snapshot_retention_days: 30   # ảnh chụp cũ hơn số ngày này sẽ bị xoá

web:
  host: 127.0.0.1       # 0.0.0.0 để máy khác trong LAN truy cập (bắt buộc ADMIN_PASSWORD)
  port: 8000
```
- [ ] **Step 4:** chạy lại → PASS

## Task 2: Model store

**Files:** Create `cameraai/model_store.py`; Test `tests/test_model_store.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_model_store.py`**

```python
import hashlib

import pytest

from cameraai import model_store
from cameraai.model_store import ModelFile, ModelMissingError, download, model_path, verify


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_retrieve(content: bytes, calls: list):
    def retrieve(url, dest):
        calls.append(url)
        with open(dest, "wb") as f:
            f.write(content)

    return retrieve


def test_verify_reports_missing_and_bad_hash(tmp_path):
    (tmp_path / "good.onnx").write_bytes(b"abc")
    (tmp_path / "bad.onnx").write_bytes(b"xyz")
    models = [
        ModelFile("good.onnx", "u", sha(b"abc")),
        ModelFile("bad.onnx", "u", sha(b"abc")),
        ModelFile("missing.onnx", "u", sha(b"")),
    ]
    assert verify(tmp_path, models) == ["Sai SHA256: bad.onnx", "Thiếu missing.onnx"]


def test_model_path_missing(tmp_path):
    with pytest.raises(ModelMissingError, match="download"):
        model_path(tmp_path, model_store.DETECTOR)


def test_download_rejects_wrong_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store.urllib.request, "urlretrieve", fake_retrieve(b"evil", []))
    with pytest.raises(RuntimeError, match="SHA256"):
        download(tmp_path, [ModelFile("m.onnx", "https://x/m.onnx", sha(b"good"))])
    assert list(tmp_path.iterdir()) == []


def test_download_saves_then_skips_existing(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(model_store.urllib.request, "urlretrieve", fake_retrieve(b"good", calls))
    models = [ModelFile("m.onnx", "https://x/m.onnx", sha(b"good"))]
    download(tmp_path, models)
    download(tmp_path, models)
    assert (tmp_path / "m.onnx").read_bytes() == b"good"
    assert calls == ["https://x/m.onnx"]


def test_pinned_models_have_sha256_and_commit_urls():
    for model in model_store.ALL_MODELS:
        assert len(model.sha256) == 64
        assert "/raw/" in model.url and "/main/" not in model.url
```
- [ ] **Step 2: Cài đặt**
**`cameraai/model_store.py`**

```python
"""Danh sách model ONNX (ghim theo commit + SHA256): tải về, kiểm tra, lấy đường dẫn.

Chạy: python -m cameraai.model_store download   (hoặc: verify)
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str


_ZOO = "https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models"
_FAS = "https://github.com/facenox/face-antispoof-onnx/raw/2f8dd93dd0fd9d120f08e83d9020719f67ffdb68/models/best/98.20"

# YuNet — giấy phép MIT
DETECTOR = ModelFile(
    "face_detection_yunet_2023mar.onnx",
    f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
# SFace — giấy phép Apache-2.0
RECOGNIZER = ModelFile(
    "face_recognition_sface_2021dec.onnx",
    f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)
# MiniFASNetV2 128x128 — giấy phép Apache-2.0
LIVENESS = ModelFile(
    "liveness_minifasnetv2_128.onnx",
    f"{_FAS}/best_model.onnx",
    "af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6",
)
ALL_MODELS: tuple[ModelFile, ...] = (DETECTOR, RECOGNIZER, LIVENESS)

DOWNLOAD_HINT = "python -m cameraai.model_store download"


class ModelMissingError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(models_dir: Path, model: ModelFile) -> Path:
    path = Path(models_dir) / model.name
    if not path.is_file():
        raise ModelMissingError(f"Thiếu model {model.name} trong {models_dir}. Chạy: {DOWNLOAD_HINT}")
    return path


def verify(models_dir: Path, models: Iterable[ModelFile] = ALL_MODELS) -> list[str]:
    problems = []
    for model in models:
        path = Path(models_dir) / model.name
        if not path.is_file():
            problems.append(f"Thiếu {model.name}")
        elif sha256_of(path) != model.sha256:
            problems.append(f"Sai SHA256: {model.name}")
    return problems


def download(models_dir: Path, models: Iterable[ModelFile] = ALL_MODELS) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        target = models_dir / model.name
        if target.is_file() and sha256_of(target) == model.sha256:
            print(f"[có sẵn] {model.name}")
            continue
        part = target.with_name(target.name + ".part")
        print(f"[đang tải] {model.name}")
        urllib.request.urlretrieve(model.url, part)
        actual = sha256_of(part)
        if actual != model.sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 của {model.name} không khớp (nhận {actual})")
        part.replace(target)
        print(f"[xong] {model.name}")


def main(argv: list[str] | None = None) -> int:
    from .config import load_config
    from .logsetup import setup_console

    setup_console()
    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else "verify"
    if command not in ("download", "verify"):
        print("Cách dùng: python -m cameraai.model_store [download|verify]")
        return 2
    models_dir = load_config().models_path
    if command == "download":
        download(models_dir)
    problems = verify(models_dir)
    for problem in problems:
        print(problem)
    if problems:
        print(f"Chạy: {DOWNLOAD_HINT}")
        return 1
    print(f"Đủ {len(ALL_MODELS)} model trong {models_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- [ ] **Step 3:** `pytest tests/test_model_store.py` → PASS
- [ ] **Step 4:** Tải model: `.\.venv\Scripts\python.exe -m cameraai.model_store download` → "Đủ 3 model trong ...\models"

## Task 3: Gallery + FaceObservation

**Files:** Create `cameraai/gallery.py`, `cameraai/observation.py`, `tests/helpers.py`; Test `tests/test_gallery.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_gallery.py`**

```python
import numpy as np

from cameraai.gallery import Gallery, normalize


def vec(*head):
    v = np.zeros(128, np.float32)
    v[: len(head)] = head
    return v


def test_empty_gallery_matches_nobody():
    m = Gallery(0.363).match(vec(1))
    assert m.employee_id is None and m.best_employee_id is None and m.score == 0.0


def test_best_match_above_threshold():
    g = Gallery(0.363)
    g.load([(1, vec(1, 0)), (1, vec(0.9, 0.1)), (2, vec(0, 1))])
    m = g.match(vec(2, 0.1))
    assert m.employee_id == 1 and m.score > 0.99
    assert g.size == 3


def test_below_threshold_keeps_nearest():
    g = Gallery(0.9)
    g.load([(1, vec(1, 0)), (2, vec(0, 1))])
    m = g.match(vec(1, 1))
    assert m.employee_id is None
    assert m.best_employee_id in (1, 2)
    assert abs(m.score - 0.7071) < 1e-3


def test_reload_replaces_samples():
    g = Gallery(0.363)
    g.load([(1, vec(1))])
    g.load([(2, vec(1))])
    assert g.match(vec(1)).employee_id == 2


def test_normalize_zero_vector_safe():
    assert not np.isnan(normalize(np.zeros(4))).any()
```
**`tests/helpers.py`**

```python
"""Dữ liệu giả dùng chung cho test."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from cameraai.camera import CameraStatus
from cameraai.gallery import Match
from cameraai.observation import FaceObservation

TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def at(hour: int, minute: int = 0, second: int = 0, day: int = 26) -> datetime:
    return datetime(2026, 10, day, hour, minute, second, tzinfo=TZ)


def make_obs(
    employee_id: int | None = 1,
    *,
    score: float = 0.7,
    real: bool | None = True,
    box=(100, 100, 120, 120),
    det_score: float = 0.95,
    reject_reason: str | None = None,
    nearest: int | None = None,
    liveness: float | None = None,
) -> FaceObservation:
    if liveness is not None:
        real = liveness >= 0.5
    elif real is not None:
        liveness = 0.9 if real else 0.1
    return FaceObservation(
        box=box,
        det_score=det_score,
        sharpness=100.0,
        reject_reason=reject_reason,
        liveness=liveness,
        is_real=real,
        embedding=np.ones(128, dtype=np.float32),
        match=Match(employee_id, score, employee_id if employee_id is not None else nearest),
    )


class FakeEngine:
    def __init__(self, faces=None, liveness_enabled: bool = True):
        self.faces = list(faces or [])
        self.liveness_enabled = liveness_enabled

    def analyze(self, frame, min_face_px=None):
        return list(self.faces)


class FakeSource:
    """Mỗi lần gọi latest() trả về một frame mới (empty=True: chưa có hình)."""

    def __init__(self, empty: bool = False):
        self.frame = None if empty else np.zeros((480, 640, 3), np.uint8)
        self.frame_id = 0

    def latest(self):
        if self.frame is None:
            return self.frame_id, None
        self.frame_id += 1
        return self.frame_id, self.frame

    def status(self):
        return CameraStatus(True, False, "0", 640, 480, 30.0, None)
```
- [ ] **Step 2: Cài đặt**
**`cameraai/gallery.py`**

```python
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
```
**`cameraai/observation.py`**

```python
"""Kết quả phân tích một khuôn mặt trên một frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gallery import Match

Box = tuple[int, int, int, int]  # x, y, w, h trên frame gốc


@dataclass
class FaceObservation:
    box: Box
    det_score: float
    sharpness: float = 0.0
    reject_reason: str | None = None  # "small" | "edge" | "turned" | "blurry"
    liveness: float | None = None  # xác suất mặt thật; None nếu không kiểm
    is_real: bool | None = None
    embedding: np.ndarray | None = None
    match: Match | None = None

    @property
    def usable(self) -> bool:
        return self.reject_reason is None
```
- [ ] **Step 3:** `pytest tests/test_gallery.py` → PASS

## Task 4: Tiền xử lý chống giả mạo

**Files:** Create `cameraai/liveness.py`; Test `tests/test_liveness_preprocess.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_liveness_preprocess.py`**

```python
import numpy as np
import pytest

from cameraai.liveness import INPUT_SIZE, crop_square, real_probability, to_input


def gradient(h=100, w=100):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.arange(w, dtype=np.uint8)[None, :]
    return img


def test_crop_square_centered():
    img = gradient()
    crop = crop_square(img, (40, 40, 20, 20))
    assert crop.shape == (30, 30, 3)
    assert crop[15, 15, 0] == img[50, 50, 0]


def test_crop_square_pads_at_edge():
    assert crop_square(gradient(), (0, 0, 40, 40)).shape == (60, 60, 3)


def test_crop_square_outside_raises():
    with pytest.raises(ValueError):
        crop_square(gradient(), (500, 500, 20, 20))


def test_to_input_shape_and_range():
    out = to_input(np.full((30, 30, 3), 255, np.uint8))
    assert out.shape == (3, INPUT_SIZE, INPUT_SIZE) and out.dtype == np.float32
    assert out.max() == pytest.approx(1.0)


def test_to_input_letterboxes_non_square():
    assert to_input(np.zeros((60, 30, 3), np.uint8)).shape == (3, INPUT_SIZE, INPUT_SIZE)


def test_real_probability_is_stable():
    assert real_probability(0, 0) == pytest.approx(0.5)
    assert real_probability(10, 0) > 0.99
    assert real_probability(-1000, 0) == pytest.approx(0.0)
    assert real_probability(1000, 0) == pytest.approx(1.0)
```
- [ ] **Step 2: Cài đặt**
**`cameraai/liveness.py`**

```python
"""Chống giả mạo bằng MiniFASNetV2 (ONNX).

Tiền xử lý bám đúng mã gốc của repo model (facenox/face-antispoof-onnx): crop vuông ×1,5 quanh khung
mặt (thiếu biên thì pad phản chiếu), ảnh RGB, letterbox 128×128, chia 255, CHW float32.
Đầu ra là 2 logit [real, spoof]; xác suất mặt thật = sigmoid(real - spoof).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = 128
BBOX_EXPANSION = 1.5


def crop_square(image: np.ndarray, box, expansion: float = BBOX_EXPANSION) -> np.ndarray:
    x, y, w, h = box
    size = int(max(w, h) * expansion)
    if w <= 0 or h <= 0 or size <= 0:
        raise ValueError("Khung mặt không hợp lệ")
    x1 = int(x + w / 2 - size / 2)
    y1 = int(y + h / 2 - size / 2)
    img_h, img_w = image.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(img_w, x1 + size), min(img_h, y1 + size)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("Khung mặt nằm ngoài ảnh")
    crop = image[cy1:cy2, cx1:cx2]
    return cv2.copyMakeBorder(
        crop, cy1 - y1, y1 + size - cy2, cx1 - x1, x1 + size - cx2, cv2.BORDER_REFLECT_101
    )


def to_input(face_rgb: np.ndarray) -> np.ndarray:
    h, w = face_rgb.shape[:2]
    ratio = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(face_rgb, (new_w, new_h), interpolation=interpolation)
    dh, dw = INPUT_SIZE - new_h, INPUT_SIZE - new_w
    img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_REFLECT_101)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def real_probability(real_logit: float, spoof_logit: float) -> float:
    diff = real_logit - spoof_logit
    if diff >= 0:
        return 1.0 / (1.0 + math.exp(-diff))
    e = math.exp(diff)
    return e / (1.0 + e)


class LivenessChecker:
    def __init__(self, model_path: Path, threshold: float = 0.5):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.threshold = threshold

    def score(self, frame_bgr: np.ndarray, box) -> float:
        """Xác suất khuôn mặt trong box là mặt thật (0..1)."""
        face = cv2.cvtColor(crop_square(frame_bgr, box), cv2.COLOR_BGR2RGB)
        logits = self._session.run(None, {self._input_name: to_input(face)[np.newaxis]})[0][0]
        return real_probability(float(logits[0]), float(logits[1]))

    def is_real(self, probability: float) -> bool:
        return probability >= self.threshold
```
- [ ] **Step 3:** `pytest tests/test_liveness_preprocess.py` → PASS

## Task 5: Tracker + quyết định chấm công

**Files:** Create `cameraai/attendance.py`; Test `tests/test_attendance.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_attendance.py`**

```python
from datetime import timedelta

from helpers import at, make_obs

from cameraai.attendance import AttendanceDecider, FaceTracker, iou
from cameraai.config import AttendanceConfig

CFG = AttendanceConfig(window_frames=5, confirm_frames=3, cooldown_minutes=5)


def feed(decider, frames, start, tracker=None, step_s=0.125):
    """frames: list[list[obs]]. Trả về [(chỉ số frame, Decision)]."""
    tracker = tracker or FaceTracker(window=CFG.window_frames)
    out = []
    for index, observations in enumerate(frames):
        now = start + timedelta(seconds=index * step_s)
        for track, _ in tracker.update(observations, now.timestamp()):
            decision = decider.evaluate(track, now)
            if decision:
                out.append((index, decision))
    return out


def summary(results):
    return [(index, d.kind, d.employee_id) for index, d in results]


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_tracker_follows_moving_face_and_splits_far_faces():
    t = FaceTracker()
    ((a, _),) = t.update([make_obs(box=(100, 100, 100, 100))], 0.0)
    ((b, _),) = t.update([make_obs(box=(110, 105, 100, 100))], 0.1)
    assert a.id == b.id
    pairs = t.update([make_obs(box=(112, 105, 100, 100)), make_obs(box=(400, 100, 100, 100))], 0.2)
    assert pairs[0][0].id == a.id and pairs[1][0].id != a.id


def test_tracker_expires_old_tracks():
    t = FaceTracker(max_age_s=1.5)
    t.update([make_obs()], 0.0)
    t.update([], 2.0)
    assert t.tracks == []


def test_unusable_observation_not_in_history():
    ((track, _),) = FaceTracker().update([make_obs(reject_reason="blurry")], 0.0)
    assert len(track.history) == 0


def test_attendance_after_confirm_frames_once_per_track():
    d = AttendanceDecider(CFG, liveness_enabled=True)
    assert summary(feed(d, [[make_obs(7)]] * 6, at(8))) == [(2, "attendance", 7)]


def test_majority_vote_picks_most_seen_employee():
    d = AttendanceDecider(CFG, liveness_enabled=True)
    frames = [[make_obs(e)] for e in (7, 8, 7, 8, 7)]
    assert summary(feed(d, frames, at(8))) == [(4, "attendance", 7)]


def test_cooldown_blocks_new_track_then_allows_later():
    d = AttendanceDecider(CFG, liveness_enabled=True)
    first = feed(d, [[make_obs(7)]] * 3, at(8, 0))
    again = feed(d, [[make_obs(7)]] * 3, at(8, 3))
    later = feed(d, [[make_obs(7)]] * 3, at(8, 6))
    assert len(first) == 1 and again == [] and len(later) == 1


def test_requires_real_faces_when_liveness_on():
    d = AttendanceDecider(CFG, liveness_enabled=True)
    frames = [[make_obs(7, real=r)] for r in (True, False, True, False)]
    assert feed(d, frames, at(8)) == []


def test_spoof_recorded_once_after_full_window():
    d = AttendanceDecider(CFG, liveness_enabled=True)
    assert summary(feed(d, [[make_obs(7, real=False)]] * 8, at(8))) == [(4, "spoof", 7)]


def test_spoof_uses_window_average():
    d = AttendanceDecider(CFG, liveness_enabled=True, spoof_threshold=0.3)
    frames = [[make_obs(7, liveness=p)] for p in (0.2, 0.1, 0.4, 0.1, 0.2)]
    assert summary(feed(d, frames, at(8))) == [(4, "spoof", 7)]


def test_real_face_with_occasional_low_scores_not_spoof():
    d = AttendanceDecider(CFG, liveness_enabled=True, spoof_threshold=0.3)
    frames = [[make_obs(7, liveness=p)] for p in (0.9, 0.1, 0.95, 0.05, 0.98, 0.2, 0.97)]
    assert summary(feed(d, frames, at(8))) == [(4, "attendance", 7)]


def test_looks_spoof_follows_current_window():
    d = AttendanceDecider(CFG, liveness_enabled=True, spoof_threshold=0.3)
    tracker = FaceTracker(window=CFG.window_frames)
    frames = [[make_obs(7, liveness=0.05)]] * 5 + [[make_obs(7, liveness=0.95)]] * 5
    results = feed(d, frames[:5], at(8), tracker=tracker)
    (track,) = tracker.tracks
    assert d.looks_spoof(track) is True
    results += feed(d, frames[5:], at(8, 0, 1), tracker=tracker)
    assert summary(results) == [(4, "spoof", 7), (2, "attendance", 7)]
    assert d.looks_spoof(track) is False and track.employee_id == 7


def test_liveness_disabled_ignores_flag():
    d = AttendanceDecider(CFG, liveness_enabled=False)
    assert summary(feed(d, [[make_obs(7, real=None)]] * 3, at(8))) == [(2, "attendance", 7)]


def test_unknown_after_full_window():
    d = AttendanceDecider(CFG, liveness_enabled=True)
    frames = [[make_obs(None, score=0.2, nearest=3)]] * 7
    assert summary(feed(d, frames, at(8))) == [(4, "unknown", 3)]


def test_seed_restores_cooldown():
    d = AttendanceDecider(CFG, liveness_enabled=True)
    d.seed({7: at(7, 58)})
    assert feed(d, [[make_obs(7)]] * 3, at(8)) == []
    assert d.last_recorded(7) == at(7, 58)
```
- [ ] **Step 2: Cài đặt**
**`cameraai/attendance.py`**

```python
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
    def __init__(self, cfg: AttendanceConfig, liveness_enabled: bool, spoof_threshold: float = 0.3):
        self._cfg = cfg
        self._liveness_enabled = liveness_enabled
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
        """Cửa sổ HIỆN TẠI có đủ frame và điểm mặt thật trung bình dưới ngưỡng giả mạo."""
        if not self._liveness_enabled:
            return False
        scores = [o.liveness for o in track.history if o.liveness is not None]
        return len(scores) >= self._cfg.window_frames and sum(scores) / len(scores) < self._spoof_threshold

    def evaluate(self, track: Track, now: datetime) -> Decision | None:
        history = list(track.history)
        if not history:
            return None
        confirm = self._cfg.confirm_frames

        if "spoof" not in track.done and self.looks_spoof(track):
            track.done.add("spoof")
            rep = min(history, key=lambda o: o.liveness if o.liveness is not None else 1.0)
            return Decision("spoof", track.id, rep, _nearest(rep), _score(rep), rep.liveness)

        real = [o for o in history if not self._liveness_enabled or o.is_real]
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
                    best = max((o for o in real if o.match and o.match.employee_id == employee_id), key=_score)
                    return Decision("attendance", track.id, best, employee_id, _score(best), best.liveness)

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
```
- [ ] **Step 3:** `pytest tests/test_attendance.py` → PASS

## Task 6: Storage

**Files:** Create `cameraai/storage.py`; Test `tests/test_storage.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_storage.py`**

```python
import numpy as np
import pytest
from helpers import at

from cameraai.storage import DuplicateEmployeeError, Storage

JPEG = b"\xff\xd8fake-jpeg"


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "data")
    yield s
    s.close()


def test_add_and_list_employees(storage):
    a = storage.add_employee(" NV01 ", "Nguyễn Văn An", at(8))
    assert a["code"] == "NV01" and a["sample_count"] == 0
    with pytest.raises(DuplicateEmployeeError):
        storage.add_employee("NV01", "Khác", at(8))
    with pytest.raises(ValueError):
        storage.add_employee("", "Tên", at(8))
    storage.add_employee("NV00", "Bình", at(8))
    assert [e["code"] for e in storage.list_employees()] == ["NV00", "NV01"]


def test_samples_roundtrip_and_gallery(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    emb = np.arange(128, dtype=np.float32)
    sample_id = storage.add_sample(emp["id"], emb, JPEG, "camera", at(8))
    (sample,) = storage.list_samples(emp["id"])
    assert sample["id"] == sample_id and sample["source"] == "camera"
    assert storage.resolve_file(sample["image_path"]).read_bytes() == JPEG
    ((owner, vector),) = storage.gallery_samples()
    assert owner == emp["id"] and np.array_equal(vector, emb)
    assert storage.get_employee(emp["id"])["sample_count"] == 1


def test_deactivate_removes_biometrics_keeps_history(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    storage.add_sample(emp["id"], np.ones(128, np.float32), JPEG, "upload", at(8))
    image = storage.resolve_file(storage.list_samples(emp["id"])[0]["image_path"])
    storage.add_attendance(emp["id"], "cam", at(8, 5), 0.8, 0.9, JPEG)
    assert storage.deactivate_employee(emp["id"]) is True
    assert storage.gallery_samples() == [] and not image.exists()
    assert storage.list_employees() == []
    assert len(storage.attendance_on(at(8).date())) == 1
    assert storage.deactivate_employee(emp["id"]) is False
    again = storage.add_employee("NV01", "An mới", at(9))
    assert again["id"] == emp["id"] and again["name"] == "An mới"


def test_delete_sample(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    sample_id = storage.add_sample(emp["id"], np.ones(128, np.float32), JPEG, "camera", at(8))
    assert storage.delete_sample(sample_id) is True
    assert storage.list_samples(emp["id"]) == []
    assert storage.delete_sample(sample_id) is False


def test_daily_summary_and_day_boundaries(storage):
    a = storage.add_employee("NV01", "An", at(7))
    b = storage.add_employee("NV02", "Bình", at(7))
    for ts in (at(8, 1), at(12, 0), at(17, 30)):
        storage.add_attendance(a["id"], "cam", ts, 0.8, 0.9, None)
    storage.add_attendance(b["id"], "cam", at(9, 15), 0.7, None, JPEG)
    storage.add_attendance(a["id"], "cam", at(8, 0, day=27), 0.8, 0.9, None)
    summary = storage.daily_summary(at(0).date())
    assert [(s["code"], s["first_ts"][11:16], s["last_ts"][11:16], s["count"]) for s in summary] == [
        ("NV01", "08:01", "17:30", 3),
        ("NV02", "09:15", "09:15", 1),
    ]
    assert len(storage.attendance_on(at(0).date())) == 4
    assert len(storage.attendance_between(at(0).date(), at(0, day=27).date())) == 5


def test_rejected_and_last_times(storage):
    a = storage.add_employee("NV01", "An", at(7))
    storage.add_rejected("cam", at(8), "spoof", a["id"], 0.6, 0.1, JPEG)
    storage.add_rejected("cam", at(8, 1), "unknown", None, 0.2, 0.9, None)
    rows = storage.rejected_on(at(8).date())
    assert [(r["reason"], r["best_name"]) for r in rows] == [("unknown", None), ("spoof", "An")]
    storage.add_attendance(a["id"], "cam", at(8, 2), 0.8, 0.9, None)
    storage.add_attendance(a["id"], "other", at(8, 3), 0.8, 0.9, None)
    assert storage.last_attendance_times("cam", at(8)) == {a["id"]: at(8, 2)}


def test_pending_and_mark_synced(storage):
    a = storage.add_employee("NV01", "An", at(7))
    first = storage.add_attendance(a["id"], "cam", at(8), 0.8, 0.9, None)
    second = storage.add_attendance(a["id"], "cam", at(9), 0.8, 0.9, None)
    pending = storage.pending_events()
    assert [e["id"] for e in pending] == [first, second] and pending[0]["employee_code"] == "NV01"
    storage.mark_synced([first], at(10))
    assert [e["id"] for e in storage.pending_events()] == [second]


def test_cleanup_removes_old_snapshots(storage):
    a = storage.add_employee("NV01", "An", at(7))
    storage.add_attendance(a["id"], "cam", at(8, day=1), 0.8, 0.9, JPEG)
    storage.add_attendance(a["id"], "cam", at(8, day=26), 0.8, 0.9, JPEG)
    storage.add_rejected("cam", at(8, day=1), "unknown", None, 0.1, 0.9, JPEG)
    old_file = storage.resolve_file(storage.attendance_on(at(8, day=1).date())[0]["snapshot_path"])
    assert storage.cleanup(at(0, day=20)) == 2
    assert not old_file.exists()
    assert storage.attendance_on(at(8, day=1).date())[0]["snapshot_path"] is None
    assert storage.rejected_on(at(8, day=1).date()) == []
    assert storage.attendance_on(at(8).date())[0]["snapshot_path"] is not None


def test_resolve_file_blocks_escape(storage):
    assert storage.resolve_file("../secret.txt") is None
    assert storage.resolve_file("cameraai.db") is None
```
- [ ] **Step 2: Cài đặt**
**`cameraai/storage.py`**

```python
"""Lưu trữ SQLite + file ảnh trong thư mục data/.

Giờ lưu dạng ISO 8601 kèm offset của múi giờ cấu hình; vì cùng một offset nên so sánh chuỗi = so sánh thời gian.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Iterable
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
    embedding BLOB NOT NULL,
    image_path TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
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
        self._db.commit()

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
        self, employee_id: int, embedding: np.ndarray, image_jpeg: bytes | None, source: str, now: datetime
    ) -> int:
        image_path = self._write_file(f"faces/{employee_id}", image_jpeg) if image_jpeg else None
        blob = np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()
        with self._lock, self._db:
            return self._db.execute(
                "INSERT INTO face_samples(employee_id, embedding, image_path, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (employee_id, blob, image_path, source, _iso(now)),
            ).lastrowid

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

    def gallery_samples(self) -> list[tuple[int, np.ndarray]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT s.employee_id, s.embedding FROM face_samples s "
                "JOIN employees e ON e.id = s.employee_id WHERE e.active = 1"
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


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def _day_range(start: date, end: date) -> tuple[str, str]:
    return start.isoformat(), (end + timedelta(days=1)).isoformat()
```
- [ ] **Step 3:** `pytest tests/test_storage.py` → PASS

## Task 7: SyncWorker (chỗ cắm Fastdo)

**Files:** Create `cameraai/sync.py`; Test `tests/test_sync.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_sync.py`**

```python
from helpers import at

from cameraai.storage import Storage
from cameraai.sync import SyncWorker


class FakeSink:
    def __init__(self, fail=False, accept=None):
        self.fail = fail
        self.accept = accept
        self.batches = []

    def send(self, events):
        if self.fail:
            raise ConnectionError("mất mạng")
        self.batches.append(events)
        ids = [e["id"] for e in events]
        return ids if self.accept is None else ids[: self.accept]


def setup(tmp_path):
    storage = Storage(tmp_path)
    emp = storage.add_employee("NV01", "An", at(7))
    ids = [storage.add_attendance(emp["id"], "cam", at(8, i), 0.8, 0.9, None) for i in range(3)]
    return storage, ids


def test_run_once_marks_sent(tmp_path):
    storage, _ = setup(tmp_path)
    sink = FakeSink()
    assert SyncWorker(storage, sink, clock=lambda: at(9)).run_once() == 3
    assert storage.pending_events() == [] and sink.batches[0][0]["employee_code"] == "NV01"
    storage.close()


def test_failure_keeps_pending(tmp_path):
    storage, _ = setup(tmp_path)
    assert SyncWorker(storage, FakeSink(fail=True), clock=lambda: at(9)).run_once() == 0
    assert len(storage.pending_events()) == 3
    storage.close()


def test_partial_success(tmp_path):
    storage, ids = setup(tmp_path)
    assert SyncWorker(storage, FakeSink(accept=1), clock=lambda: at(9)).run_once() == 1
    assert [e["id"] for e in storage.pending_events()] == ids[1:]
    storage.close()


def test_nothing_pending_skips_sink(tmp_path):
    storage, _ = setup(tmp_path)
    sink = FakeSink()
    worker = SyncWorker(storage, sink, clock=lambda: at(9))
    worker.run_once()
    assert worker.run_once() == 0 and len(sink.batches) == 1
    storage.close()
```
- [ ] **Step 2: Cài đặt**
**`cameraai/sync.py`**

```python
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
```
- [ ] **Step 3:** `pytest tests/test_sync.py` → PASS

## Task 8: Nguồn video

**Files:** Create `cameraai/camera.py`; Test `tests/test_camera.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_camera.py`**

```python
import time

import numpy as np

from cameraai.camera import VideoSource
from cameraai.config import CameraConfig


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, prop):
        return 1000.0  # fps rất cao để test chạy nhanh

    def release(self):
        self.released = True


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_file_source_plays_all_frames_then_ends():
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    source = VideoSource(CameraConfig(source="clip.mp4"), opener=lambda spec, cfg: FakeCapture(frames))
    source.start()
    assert wait_until(lambda: source.status().ended)
    frame_id, frame = source.latest()
    assert frame_id == 3 and frame[0, 0, 0] == 2
    assert source.status().connected is False
    source.stop()


def test_stream_reconnects_and_masks_password():
    opens = []

    def opener(spec, cfg):
        opens.append(spec.value)
        return FakeCapture([np.zeros((4, 4, 3), np.uint8)])

    source = VideoSource(
        CameraConfig(source="rtsp://admin:secret@cam/x"), opener=opener, initial_backoff_s=0.01, max_backoff_s=0.02
    )
    source.start()
    assert wait_until(lambda: len(opens) >= 2)
    status = source.status()
    source.stop()
    assert "secret" not in status.source and status.width == 4
    assert source.status().ended is False


def test_unopenable_source_reports_error():
    source = VideoSource(
        CameraConfig(source="rtsp://cam/x"), opener=lambda spec, cfg: FakeCapture([], opened=False),
        initial_backoff_s=0.01, max_backoff_s=0.02,
    )
    source.start()
    assert wait_until(lambda: source.status().last_error is not None)
    source.stop()
    assert source.status().connected is False
```
- [ ] **Step 2: Cài đặt**
**`cameraai/camera.py`**

```python
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
```
- [ ] **Step 3:** `pytest tests/test_camera.py` → PASS

## Task 9: Ảnh + FaceEngine

**Files:** Create `cameraai/imaging.py`, `cameraai/face_engine.py`; Test `tests/test_imaging.py`, `tests/test_models_integration.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_imaging.py`**

```python
import numpy as np

from cameraai.imaging import GREEN, ascii_label, crop_around, draw_box, encode_jpeg, fit_within


def test_ascii_label_strips_vietnamese_marks():
    assert ascii_label("Nguyễn Văn Đức") == "Nguyen Van Duc"
    assert ascii_label("Lại gần hơn") == "Lai gan hon"


def test_crop_around_scales_and_clamps():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop_around(frame, (90, 40, 20, 20)).shape == (40, 40, 3)
    assert crop_around(frame, (0, 0, 20, 20)).shape == (30, 30, 3)


def test_fit_within():
    assert fit_within(np.zeros((800, 400, 3), np.uint8), 400).shape == (400, 200, 3)
    small = np.zeros((10, 10, 3), np.uint8)
    assert fit_within(small, 400) is small


def test_encode_jpeg_and_draw_box():
    frame = np.zeros((120, 160, 3), np.uint8)
    draw_box(frame, (40, 40, 50, 50), "Nguyễn An", GREEN)
    assert frame.any()
    assert encode_jpeg(frame).startswith(b"\xff\xd8")
```
**`tests/test_face_quality.py`**

```python
import numpy as np

from cameraai.face_engine import touches_edge, turn_offset

FRAME_SHAPE = (720, 1280, 3)


def test_face_inside_frame_is_kept():
    assert not touches_edge((100, 100, 200, 200), FRAME_SHAPE)


def test_face_cut_by_bottom_edge_is_rejected():
    # Ngồi quá gần webcam: khung mặt tràn mép dưới (cằm bị cắt).
    assert touches_edge((560, 400, 250, 360), FRAME_SHAPE)


def test_face_within_margin_of_edge_is_rejected():
    assert touches_edge((2, 100, 50, 50), FRAME_SHAPE)


def row(right_eye_x, left_eye_x, nose_x):
    r = np.zeros(15, np.float32)
    r[4], r[6], r[8] = right_eye_x, left_eye_x, nose_x
    return r


def test_frontal_face_has_small_offset():
    assert turn_offset(row(100, 160, 130)) == 0.0


def test_turned_face_has_large_offset():
    assert turn_offset(row(100, 160, 150)) > 0.25
    assert turn_offset(row(100, 160, 175)) > 0.5


def test_swapped_eyes_counts_as_turned():
    assert turn_offset(row(160, 100, 130)) == 1.0
```
**`tests/test_models_integration.py`**

```python
"""Kiểm tra 3 model thật nạp và chạy được (tự bỏ qua nếu chưa tải model)."""
import numpy as np
import pytest

from cameraai import model_store
from cameraai.config import AppConfig
from cameraai.face_engine import FaceEngine, sharpness
from cameraai.gallery import Gallery
from cameraai.liveness import LivenessChecker

CFG = AppConfig()
pytestmark = pytest.mark.skipif(
    bool(model_store.verify(CFG.models_path)), reason="chưa tải model (python -m cameraai.model_store download)"
)


def noise(h=480, w=640):
    return (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)


def test_no_face_on_blank_frame():
    assert FaceEngine(CFG, Gallery(0.363)).analyze(np.zeros((720, 1280, 3), np.uint8)) == []


def test_embedding_shape():
    row = np.array([200, 150, 160, 200, 240, 220, 320, 220, 280, 260, 250, 300, 310, 300, 0.99], np.float32)
    embedding = FaceEngine(CFG).embed(noise(), row)
    assert embedding.shape == (128,) and np.isfinite(embedding).all()


def test_liveness_probability_range():
    checker = LivenessChecker(model_store.model_path(CFG.models_path, model_store.LIVENESS))
    assert 0.0 <= checker.score(noise(), (200, 150, 160, 200)) <= 1.0


def test_sharpness_noise_higher_than_flat():
    flat = np.zeros((480, 640, 3), np.uint8)
    assert sharpness(noise(), (100, 100, 120, 120)) > sharpness(flat, (100, 100, 120, 120))
```
- [ ] **Step 2: Cài đặt**
**`cameraai/imaging.py`**

```python
"""Tiện ích ảnh: cắt quanh khung mặt, thu nhỏ, encode JPEG, vẽ chú thích.

Font Hershey của OpenCV chỉ có ASCII nên chữ trên hình được bỏ dấu tiếng Việt.
"""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .observation import Box

# Màu BGR
GREEN = (80, 200, 80)
YELLOW = (0, 200, 255)
RED = (60, 60, 230)
GRAY = (160, 160, 160)


def crop_around(frame: np.ndarray, box: Box, scale: float = 2.0) -> np.ndarray:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half_w, half_h = w * scale / 2, h * scale / 2
    img_h, img_w = frame.shape[:2]
    x1, y1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x2, y2 = min(img_w, int(cx + half_w)), min(img_h, int(cy + half_h))
    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    return frame[y1:y2, x1:x2].copy()


def fit_within(image: np.ndarray, max_side: int = 400) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Không encode được JPEG")
    return buf.tobytes()


def ascii_label(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return stripped.encode("ascii", "ignore").decode()


def draw_box(frame: np.ndarray, box: Box, label: str, color: tuple[int, int, int]) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = ascii_label(label)
    if not text:
        return
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    top = max(0, y - th - baseline - 6)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, top + th + 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
```
**`cameraai/face_engine.py`**

```python
"""Phát hiện (YuNet) + chống giả (MiniFASNet) + nhận diện (SFace) trên một frame.

Đối tượng OpenCV DNN không an toàn khi dùng chung giữa các thread: mỗi thread tạo FaceEngine riêng.
Đổi model nhận diện chỉ cần sửa file này (và đăng ký lại mẫu vì embedding khác nhau).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import model_store
from .config import AppConfig
from .gallery import Gallery
from .liveness import LivenessChecker
from .observation import Box, FaceObservation

SHARPNESS_SIZE = 112
EDGE_MARGIN_PX = 5  # như hàm detect() của repo model chống giả mạo


def touches_edge(box: Box, frame_shape, margin: int = EDGE_MARGIN_PX) -> bool:
    """Mặt chạm/tràn mép khung hình: điểm mốc lệch và crop chống giả bị lấp phần phản chiếu → bỏ qua."""
    x, y, w, h = box
    img_h, img_w = frame_shape[:2]
    return min(x, y, img_w - (x + w), img_h - (y + h)) < margin


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


def turn_offset(row: np.ndarray) -> float:
    """Độ quay ngang của mặt từ điểm mốc YuNet: 0 = nhìn thẳng, >= 0,5 = quay hẳn sang bên.

    Đo vị trí mũi giữa hai mắt (mắt phải của người nằm bên trái ảnh). Mặt quay nghiêng làm chống giả mạo
    và nhận diện kém tin cậy nên bị bỏ qua.
    """
    right_eye_x, left_eye_x, nose_x = float(row[4]), float(row[6]), float(row[8])
    span = left_eye_x - right_eye_x
    if span <= 0:
        return 1.0
    return abs((nose_x - right_eye_x) / span - 0.5)


class FaceEngine:
    def __init__(self, cfg: AppConfig, gallery: Gallery | None = None):
        models_dir = cfg.models_path
        self._cfg = cfg
        self._gallery = gallery
        self._detector = cv2.FaceDetectorYN.create(
            str(model_store.model_path(models_dir, model_store.DETECTOR)), "", (320, 320),
            cfg.recognition.detect_score, 0.3, 5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(model_store.model_path(models_dir, model_store.RECOGNIZER)), ""
        )
        self._liveness = (
            LivenessChecker(model_store.model_path(models_dir, model_store.LIVENESS), cfg.liveness.threshold)
            if cfg.liveness.enabled
            else None
        )

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
        aligned = self._recognizer.alignCrop(frame, row)
        return self._recognizer.feature(aligned).reshape(-1).copy()

    def analyze(self, frame: np.ndarray, min_face_px: int | None = None) -> list[FaceObservation]:
        min_px = self._cfg.recognition.min_face_px if min_face_px is None else min_face_px
        observations = []
        for row in self.detect(frame):
            box = tuple(int(round(v)) for v in row[:4])
            obs = FaceObservation(box=box, det_score=float(row[14]), sharpness=sharpness(frame, box))
            if min(box[2], box[3]) < min_px:
                obs.reject_reason = "small"
            elif touches_edge(box, frame.shape):
                obs.reject_reason = "edge"
            elif turn_offset(row) > self._cfg.recognition.max_turn:
                obs.reject_reason = "turned"
            elif obs.sharpness < self._cfg.recognition.min_sharpness:
                obs.reject_reason = "blurry"
            else:
                if self._liveness is not None:
                    obs.liveness = self._liveness.score(frame, box)
                    obs.is_real = self._liveness.is_real(obs.liveness)
                obs.embedding = self.embed(frame, row)
                if self._gallery is not None:
                    obs.match = self._gallery.match(obs.embedding)
            observations.append(obs)
        return observations
```
- [ ] **Step 3:** `pytest tests/test_imaging.py tests/test_models_integration.py` → PASS (test tích hợp chỉ chạy khi đã tải model)

## Task 10: Pipeline

**Files:** Create `cameraai/pipeline.py`; Test `tests/test_pipeline.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_pipeline.py`**

```python
from datetime import timedelta

import numpy as np
from helpers import FakeEngine, at, make_obs

from cameraai.config import AppConfig
from cameraai.gallery import Gallery
from cameraai.imaging import GREEN
from cameraai.pipeline import Pipeline
from cameraai.storage import Storage

FRAME = np.zeros((480, 640, 3), np.uint8)


def build(tmp_path):
    storage = Storage(tmp_path)
    emp = storage.add_employee("NV01", "Nguyễn Văn An", at(7))
    engine = FakeEngine()
    pipeline = Pipeline(AppConfig(), None, engine, storage, Gallery(0.363), clock=lambda: at(8))
    pipeline.prime()
    return storage, emp["id"], engine, pipeline


def run(pipeline, frames, start):
    return [pipeline.process(FRAME, start + timedelta(milliseconds=125 * i)) for i in range(frames)]


def test_records_attendance_with_snapshot(tmp_path):
    storage, emp_id, engine, pipeline = build(tmp_path)
    engine.faces = [make_obs(emp_id)]
    assert [len(r) for r in run(pipeline, 3, at(8))] == [0, 0, 1]
    (event,) = storage.attendance_on(at(8).date())
    assert event["employee_id"] == emp_id and event["camera_id"] == "laptop"
    assert storage.resolve_file(event["snapshot_path"]) is not None
    assert pipeline.recent_events()[0]["name"] == "Nguyễn Văn An"
    frame_id, jpeg = pipeline.latest_jpeg()
    assert frame_id == 3 and jpeg.startswith(b"\xff\xd8")
    storage.close()


def test_records_spoof_rejection(tmp_path):
    storage, emp_id, engine, pipeline = build(tmp_path)
    engine.faces = [make_obs(emp_id, real=False)]
    run(pipeline, 5, at(8))
    (rejected,) = storage.rejected_on(at(8).date())
    assert rejected["reason"] == "spoof" and rejected["best_employee_id"] == emp_id
    assert storage.attendance_on(at(8).date()) == []
    storage.close()


def test_label_turns_green_after_spoof_window_passes(tmp_path):
    # Lỗi thật: track bị đánh dấu giả mạo trước khi đăng ký vẫn hiện đỏ dù sau đó đã chấm công.
    storage, emp_id, engine, pipeline = build(tmp_path)
    engine.faces = [make_obs(emp_id, liveness=0.05)]
    run(pipeline, 5, at(8))
    (track,) = pipeline._tracker.tracks
    assert pipeline._label(track, engine.faces[0])[0] == "Nghi giả mạo"
    engine.faces = [make_obs(emp_id, liveness=0.95)]
    run(pipeline, 5, at(8, 0, 1))
    label, color = pipeline._label(track, engine.faces[0])
    assert label.startswith("Nguyễn Văn An - đã chấm") and color == GREEN
    assert len(storage.attendance_on(at(8).date())) == 1
    storage.close()


def test_prime_restores_cooldown_from_storage(tmp_path):
    storage, emp_id, engine, pipeline = build(tmp_path)
    storage.add_attendance(emp_id, "laptop", at(7, 58), 0.8, 0.9, None)
    pipeline.prime()
    engine.faces = [make_obs(emp_id)]
    assert sum(len(r) for r in run(pipeline, 3, at(8))) == 0
    storage.close()
```
- [ ] **Step 2: Cài đặt**
**`cameraai/pipeline.py`**

```python
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
        self._decider = AttendanceDecider(cfg.attendance, engine.liveness_enabled, cfg.liveness.spoof_threshold)
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
        self._gallery.load(self._storage.gallery_samples())
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
```
- [ ] **Step 3:** `pytest tests/test_pipeline.py` → PASS

## Task 11: Đăng ký khuôn mặt

**Files:** Create `cameraai/enrollment.py`; Test `tests/test_enrollment.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_enrollment.py`**

```python
import cv2
import numpy as np
import pytest
from helpers import FakeEngine, FakeSource, at, make_obs

from cameraai.config import AppConfig
from cameraai.enrollment import EnrollmentError, EnrollmentService
from cameraai.storage import Storage


def build(tmp_path, faces, source=None, timeout=0.3):
    cfg = AppConfig()
    cfg.enrollment.capture_timeout_s = timeout
    storage = Storage(tmp_path)
    emp = storage.add_employee("NV01", "An", at(7))
    changes = []
    service = EnrollmentService(
        cfg, lambda: FakeEngine(faces), source or FakeSource(), storage,
        lambda: changes.append(1), lambda: at(8), sleep=lambda s: None,
    )
    return service, storage, emp["id"], changes


def png(size):
    ok, data = cv2.imencode(".png", np.zeros((size, size, 3), np.uint8))
    return data.tobytes()


def test_capture_saves_samples(tmp_path):
    service, storage, emp_id, changes = build(tmp_path, [make_obs(None)])
    result = service.capture(emp_id)
    assert (result.saved, result.attempts, changes) == (5, 5, [1])
    assert [s["source"] for s in storage.list_samples(emp_id)] == ["camera"] * 5
    storage.close()


def test_capture_rejects_two_faces(tmp_path):
    service, storage, emp_id, changes = build(tmp_path, [make_obs(), make_obs(box=(300, 100, 120, 120))])
    result = service.capture(emp_id)
    assert result.saved == 0 and set(result.reasons) == {"many_faces"} and changes == []
    storage.close()


def test_capture_rejects_spoof(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, [make_obs(real=False)])
    assert set(service.capture(emp_id).reasons) == {"spoof"}
    storage.close()


def test_capture_without_frames(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, [make_obs()], source=FakeSource(empty=True), timeout=0.1)
    result = service.capture(emp_id)
    assert result.to_dict()["reasons"][0]["code"] == "no_camera" and result.saved == 0
    storage.close()


def test_capture_unknown_employee(tmp_path):
    service, storage, _, _ = build(tmp_path, [make_obs()])
    with pytest.raises(EnrollmentError):
        service.capture(999)
    storage.close()


def test_upload_valid_image(tmp_path):
    service, storage, emp_id, changes = build(tmp_path, [make_obs()])
    assert service.add_upload(emp_id, png(300)) is None
    assert storage.list_samples(emp_id)[0]["source"] == "upload" and changes == [1]
    storage.close()


def test_upload_rejects_garbage_and_low_score(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, [make_obs(det_score=0.5)])
    assert service.add_upload(emp_id, b"not an image") == "not_image"
    assert service.add_upload(emp_id, png(50)) == "low_score"
    storage.close()
```
- [ ] **Step 2: Cài đặt**
**`cameraai/enrollment.py`**

```python
"""Đăng ký mẫu khuôn mặt: chụp từ camera đang chạy hoặc từ ảnh tải lên.

Chụp phía server (lấy frame mới nhất của VideoSource) vì trên Windows webcam bị khoá cho một tiến trình,
trình duyệt không mở song song được.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from .config import AppConfig
from .imaging import crop_around, encode_jpeg, fit_within
from .observation import FaceObservation
from .storage import Storage

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

    def check_frame(self, frame: np.ndarray, require_real: bool) -> tuple[FaceObservation | None, str | None]:
        if self._engine is None:  # tạo lười: model chỉ nạp khi lần đầu đăng ký
            self._engine = self._engine_factory()
        faces = self._engine.analyze(frame, min_face_px=self._cfg.enrollment.min_face_px)
        if not faces:
            return None, "no_face"
        if len(faces) > 1:
            return None, "many_faces"
        face = faces[0]
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
            deadline = time.monotonic() + cfg.capture_timeout_s
            while saved < cfg.samples_per_capture and time.monotonic() < deadline:
                frame_id, frame = self._source.latest()
                if frame is None or frame_id == last_id:
                    self._sleep(0.1)
                    continue
                last_id = frame_id
                attempts += 1
                face, reason = self.check_frame(frame, require_real=True)
                if reason:
                    reasons[reason] += 1
                else:
                    self._save(employee_id, frame, face, "camera")
                    saved += 1
                self._sleep(0.4)  # giãn cách để các mẫu khác nhau đôi chút
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

    def _save(self, employee_id: int, frame: np.ndarray, face: FaceObservation, source: str) -> None:
        jpeg = encode_jpeg(fit_within(crop_around(frame, face.box, 1.6)))
        self._storage.add_sample(employee_id, face.embedding, jpeg, source, self._clock())

    def _require_employee(self, employee_id: int) -> None:
        if self._storage.get_employee(employee_id) is None:
            raise EnrollmentError("Không tìm thấy nhân viên")
```
- [ ] **Step 3:** `pytest tests/test_enrollment.py` → PASS

## Task 12: Web API + giao diện

**Files:** Create `cameraai/web/__init__.py`, `cameraai/web/app.py`, `cameraai/web/static/index.html`, `app.js`, `app.css`; Test `tests/test_web.py`

- [ ] **Step 1: Test (đỏ)**
**`tests/test_web.py`**

```python
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import FakeEngine, FakeSource, at, make_obs

from cameraai.config import AppConfig
from cameraai.enrollment import EnrollmentService
from cameraai.storage import Storage
from cameraai.web.app import Services, create_app


class FakePipeline:
    processed_fps = 7.5

    def __init__(self):
        self.reloads = 0

    def latest_jpeg(self):
        return 1, b"\xff\xd8jpeg"

    def recent_events(self):
        return [{"kind": "attendance", "name": "An", "ts": "2026-10-26T08:00:00+07:00", "score": 0.8, "liveness": 0.9}]

    def reload_gallery(self):
        self.reloads += 1


def make_ctx(tmp_path, password=""):
    cfg = AppConfig()
    cfg.web.admin_password = password
    storage = Storage(tmp_path / "data")
    pipeline = FakePipeline()
    source = FakeSource()
    enrollment = EnrollmentService(
        cfg, lambda: FakeEngine([make_obs()]), source, storage, pipeline.reload_gallery, lambda: at(8),
        sleep=lambda s: None,
    )
    client = TestClient(create_app(Services(cfg, storage, source, pipeline, enrollment, lambda: at(8))))
    return client, storage, pipeline


@pytest.fixture
def ctx(tmp_path):
    client, storage, pipeline = make_ctx(tmp_path)
    yield client, storage, pipeline
    storage.close()


def add_employee(client, code="NV01", name="An"):
    response = client.post("/api/employees", json={"code": code, "name": name})
    assert response.status_code == 201
    return response.json()["id"]


def test_index_and_status(ctx):
    client, _, _ = ctx
    assert "CameraAI" in client.get("/").text
    status = client.get("/api/status").json()
    assert status["camera"]["connected"] is True and status["processing_fps"] == 7.5
    assert status["recent"][0]["name"] == "An" and status["now"].startswith("2026-10-26T08:00")


def test_employee_crud(ctx):
    client, _, pipeline = ctx
    emp_id = add_employee(client)
    assert client.post("/api/employees", json={"code": "NV01", "name": "X"}).status_code == 409
    assert client.post("/api/employees", json={"code": " ", "name": "X"}).status_code == 400
    assert [e["code"] for e in client.get("/api/employees").json()] == ["NV01"]
    assert client.delete(f"/api/employees/{emp_id}").status_code == 200
    assert client.delete(f"/api/employees/{emp_id}").status_code == 404
    assert pipeline.reloads == 1


def test_upload_capture_and_samples(ctx):
    client, _, _ = ctx
    emp_id = add_employee(client)
    ok, png = cv2.imencode(".png", np.zeros((300, 300, 3), np.uint8))
    files = [("files", ("a.png", png.tobytes(), "image/png")), ("files", ("b.txt", b"hello", "text/plain"))]
    results = client.post(f"/api/employees/{emp_id}/upload", files=files).json()
    assert [r["ok"] for r in results] == [True, False]
    assert client.post(f"/api/employees/{emp_id}/capture").json()["saved"] == 5
    samples = client.get(f"/api/employees/{emp_id}/samples").json()
    assert len(samples) == 6
    assert client.get(f"/files/{samples[0]['image_path']}").status_code == 200
    assert client.delete(f"/api/samples/{samples[0]['id']}").status_code == 200
    assert client.post("/api/employees/999/capture").status_code == 404


def test_attendance_csv_and_rejected(ctx):
    client, storage, _ = ctx
    emp = storage.add_employee("NV01", "Nguyễn Văn An", at(7))
    storage.add_attendance(emp["id"], "laptop", at(8, 1), 0.81, 0.93, b"\xff\xd8x")
    storage.add_rejected("laptop", at(8, 2), "spoof", emp["id"], 0.6, 0.1, None)
    body = client.get("/api/attendance", params={"date": "2026-10-26"}).json()
    assert body["summary"][0]["count"] == 1 and body["events"][0]["code"] == "NV01"
    csv_response = client.get("/api/attendance.csv", params={"start": "2026-10-26", "end": "2026-10-26"})
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert csv_response.text.startswith("﻿Mã NV") and "Nguyễn Văn An" in csv_response.text
    assert client.get("/api/rejected", params={"date": "2026-10-26"}).json()["events"][0]["reason"] == "spoof"
    assert client.get("/api/attendance", params={"date": "26/10/2026"}).status_code == 400


def test_files_blocks_database(ctx):
    client, _, _ = ctx
    assert client.get("/files/cameraai.db").status_code == 404


def test_basic_auth(tmp_path):
    client, storage, _ = make_ctx(tmp_path, password="bi-mat")
    assert client.get("/api/status").status_code == 401
    good = base64.b64encode(b"admin:bi-mat").decode()
    bad = base64.b64encode(b"admin:sai").decode()
    assert client.get("/api/status", headers={"Authorization": f"Basic {good}"}).status_code == 200
    assert client.get("/api/status", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    storage.close()
```
- [ ] **Step 2: Cài đặt**
**`cameraai/web/__init__.py`**

```python
"""Giao diện web và API của CameraAI."""
```
**`cameraai/web/app.py`**

```python
"""Web: API JSON, luồng MJPEG, trang quản trị tĩnh."""
from __future__ import annotations

import asyncio
import base64
import csv
import io
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import AppConfig
from ..enrollment import REASON_TEXT, EnrollmentError, EnrollmentService
from ..storage import DuplicateEmployeeError, Storage

STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class Services:
    cfg: AppConfig
    storage: Storage
    source: Any  # VideoSource: status()
    pipeline: Any  # Pipeline: latest_jpeg(), recent_events(), processed_fps, reload_gallery()
    enrollment: EnrollmentService
    clock: Callable[[], datetime]


class EmployeeIn(BaseModel):
    code: str
    name: str


class BasicAuthMiddleware:
    """HTTP Basic cho mọi route khi có ADMIN_PASSWORD (tên đăng nhập tuỳ ý)."""

    def __init__(self, app, password: str):
        self.app = app
        self._password = password.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        response = PlainTextResponse(
            "Cần đăng nhập", status_code=401, headers={"WWW-Authenticate": 'Basic realm="CameraAI"'}
        )
        await response(scope, receive, send)

    def _authorized(self, scope) -> bool:
        header = dict(scope.get("headers") or []).get(b"authorization", b"")
        if not header.lower().startswith(b"basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:], validate=True)
        except ValueError:
            return False
        _, _, password = decoded.partition(b":")
        return secrets.compare_digest(password, self._password)


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="CameraAI", docs_url=None, redoc_url=None, openapi_url=None)
    if services.cfg.web.admin_password:
        app.add_middleware(BasicAuthMiddleware, password=services.cfg.web.admin_password)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    storage, enrollment, pipeline = services.storage, services.enrollment, services.pipeline

    def parse_day(value: str | None) -> date:
        if not value:
            return services.clock().date()
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise HTTPException(400, "Ngày không hợp lệ (định dạng YYYY-MM-DD)") from None

    def require_employee(employee_id: int) -> dict:
        employee = storage.get_employee(employee_id)
        if employee is None:
            raise HTTPException(404, "Không tìm thấy nhân viên")
        return employee

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return {
            "camera": asdict(services.source.status()),
            "camera_id": services.cfg.camera.id,
            "processing_fps": pipeline.processed_fps,
            "liveness_enabled": services.cfg.liveness.enabled,
            "employees": len(storage.list_employees()),
            "recent": pipeline.recent_events(),
            "now": services.clock().isoformat(),
        }

    @app.get("/stream.mjpg")
    async def stream(request: Request):
        async def frames():
            last_id = -1
            while not await request.is_disconnected():
                frame_id, jpeg = pipeline.latest_jpeg()
                if jpeg is None or frame_id == last_id:
                    await asyncio.sleep(0.03)
                    continue
                last_id = frame_id
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n" + jpeg + b"\r\n"
                )

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"}
        )

    # ---------- nhân viên + mẫu ----------
    @app.get("/api/employees")
    def list_employees():
        return storage.list_employees()

    @app.post("/api/employees", status_code=201)
    def create_employee(body: EmployeeIn):
        try:
            return storage.add_employee(body.code, body.name, services.clock())
        except DuplicateEmployeeError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.delete("/api/employees/{employee_id}")
    def delete_employee(employee_id: int):
        if not storage.deactivate_employee(employee_id):
            raise HTTPException(404, "Không tìm thấy nhân viên")
        pipeline.reload_gallery()
        return {"ok": True}

    @app.get("/api/employees/{employee_id}/samples")
    def list_samples(employee_id: int):
        require_employee(employee_id)
        return storage.list_samples(employee_id)

    @app.post("/api/employees/{employee_id}/capture")
    def capture(employee_id: int):
        require_employee(employee_id)
        try:
            return enrollment.capture(employee_id).to_dict()
        except EnrollmentError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/employees/{employee_id}/upload")
    def upload(employee_id: int, files: list[UploadFile] = File(...)):
        require_employee(employee_id)
        results = []
        for item in files:
            reason = enrollment.add_upload(employee_id, item.file.read())
            results.append({
                "filename": item.filename,
                "ok": reason is None,
                "message": "Đã lưu mẫu" if reason is None else REASON_TEXT.get(reason, reason),
            })
        return results

    @app.delete("/api/samples/{sample_id}")
    def delete_sample(sample_id: int):
        if not storage.delete_sample(sample_id):
            raise HTTPException(404, "Không tìm thấy mẫu")
        pipeline.reload_gallery()
        return {"ok": True}

    # ---------- lịch sử ----------
    @app.get("/api/attendance")
    def attendance(day: str | None = Query(None, alias="date")):
        d = parse_day(day)
        return {"date": d.isoformat(), "summary": storage.daily_summary(d), "events": storage.attendance_on(d)}

    @app.get("/api/attendance.csv")
    def attendance_csv(start: str | None = None, end: str | None = None):
        start_day = parse_day(start)
        end_day = parse_day(end) if end else start_day
        if end_day < start_day:
            raise HTTPException(400, "Ngày kết thúc phải sau ngày bắt đầu")
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Mã NV", "Tên", "Thời điểm", "Camera", "Điểm khớp", "Điểm mặt thật"])
        for e in storage.attendance_between(start_day, end_day):
            liveness = "" if e["liveness_score"] is None else f"{e['liveness_score']:.3f}"
            writer.writerow([e["code"], e["name"], e["ts"], e["camera_id"], f"{e['score']:.3f}", liveness])
        filename = f"cham-cong_{start_day}_{end_day}.csv"
        return Response(
            "﻿" + buffer.getvalue(),  # BOM để Excel đọc đúng tiếng Việt
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/rejected")
    def rejected(day: str | None = Query(None, alias="date")):
        d = parse_day(day)
        return {"date": d.isoformat(), "events": storage.rejected_on(d)}

    @app.get("/files/{rel_path:path}")
    def files(rel_path: str):
        path = storage.resolve_file(rel_path)
        if path is None:
            raise HTTPException(404, "Không tìm thấy file")
        return FileResponse(path)

    return app
```
**`cameraai/web/static/index.html`**

```html
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CameraAI — Chấm công</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
<header class="topbar">
  <div class="brand">CameraAI <span class="muted">chấm công bằng camera</span></div>
  <nav class="tabs">
    <button data-tab="live" class="active">Trực tiếp</button>
    <button data-tab="employees">Nhân viên</button>
    <button data-tab="history">Lịch sử</button>
    <button data-tab="rejected">Bị từ chối</button>
  </nav>
  <div id="cam-badge" class="badge">Đang kết nối…</div>
</header>

<main>
  <section id="tab-live" class="tab active">
    <div class="live-grid">
      <div class="video-wrap"><img id="live-img" alt="Hình camera"></div>
      <aside class="panel">
        <h3>Trạng thái</h3>
        <dl id="status-list" class="kv"></dl>
        <h3>Vừa ghi nhận</h3>
        <ul id="recent-list" class="recent"></ul>
      </aside>
    </div>
  </section>

  <section id="tab-employees" class="tab">
    <form id="add-form" class="panel row">
      <input name="code" placeholder="Mã nhân viên" required maxlength="32">
      <input name="name" placeholder="Họ tên" required maxlength="100">
      <button type="submit" class="primary">Thêm nhân viên</button>
    </form>
    <p class="hint">Cần sự đồng ý của nhân viên trước khi đăng ký khuôn mặt. Nên có ít nhất 3 mẫu mỗi người.</p>
    <table class="table">
      <thead><tr><th>Mã</th><th>Tên</th><th>Số mẫu</th><th></th></tr></thead>
      <tbody id="emp-body"></tbody>
    </table>
  </section>

  <section id="tab-history" class="tab">
    <div class="panel row">
      <label>Ngày <input type="date" id="hist-date"></label>
      <button id="hist-refresh">Tải lại</button>
      <span class="spacer"></span>
      <label>Từ <input type="date" id="csv-start"></label>
      <label>Đến <input type="date" id="csv-end"></label>
      <button id="csv-btn">Xuất CSV</button>
    </div>
    <h3>Tổng hợp trong ngày</h3>
    <table class="table">
      <thead><tr><th>Mã</th><th>Tên</th><th>Giờ vào</th><th>Giờ ra</th><th>Số lần</th></tr></thead>
      <tbody id="summary-body"></tbody>
    </table>
    <h3>Các lần chấm</h3>
    <div id="events-grid" class="cards"></div>
  </section>

  <section id="tab-rejected" class="tab">
    <div class="panel row">
      <label>Ngày <input type="date" id="rej-date"></label>
      <button id="rej-refresh">Tải lại</button>
    </div>
    <div id="rejected-grid" class="cards"></div>
  </section>
</main>

<dialog id="enroll-dialog">
  <form method="dialog" class="dialog-head">
    <h3 id="enroll-title">Đăng ký khuôn mặt</h3>
    <button class="icon" aria-label="Đóng">✕</button>
  </form>
  <div class="enroll-body">
    <div class="video-wrap small"><img id="enroll-img" alt="Hình camera"></div>
    <div class="enroll-actions">
      <button id="capture-btn" class="primary">Chụp từ camera (5 mẫu)</button>
      <label class="file-btn">Tải ảnh lên<input type="file" id="upload-input" accept="image/*" multiple hidden></label>
      <div id="enroll-msg" class="msg"></div>
    </div>
  </div>
  <h4>Mẫu đã có</h4>
  <div id="samples-grid" class="thumbs"></div>
</dialog>

<div id="toast" class="toast" hidden></div>
<script src="/static/app.js"></script>
</body>
</html>
```
**`cameraai/web/static/app.js`**

```javascript
"use strict";

const $ = (selector) => document.querySelector(selector);
const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
const KIND_TEXT = { attendance: "Chấm công", spoof: "Nghi giả mạo", unknown: "Người lạ" };

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));
const timeOf = (iso) => (iso ? iso.slice(11, 19) : "");
const fixed = (value) => (value == null ? "—" : Number(value).toFixed(2));

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error((body && body.detail) || body || response.statusText);
  return body;
}

function toast(text, kind = "info") {
  const el = $("#toast");
  el.textContent = text;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 4000);
}

// ---------- tab ----------
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${name}`));
  // Chỉ giữ 1 luồng MJPEG khi đang xem tab Trực tiếp.
  $("#live-img").src = name === "live" ? "/stream.mjpg" : BLANK;
  if (name === "employees") loadEmployees().catch((e) => toast(e.message, "bad"));
  if (name === "history") loadHistory().catch((e) => toast(e.message, "bad"));
  if (name === "rejected") loadRejected().catch((e) => toast(e.message, "bad"));
}

// ---------- trạng thái ----------
let datesReady = false;
async function refreshStatus() {
  const badge = $("#cam-badge");
  try {
    const s = await api("/api/status");
    const cam = s.camera;
    badge.textContent = cam.connected ? `Camera: đang chạy (${cam.fps} fps)` : cam.ended ? "Video đã phát hết" : "Mất kết nối camera";
    badge.className = `badge ${cam.connected ? "ok" : "bad"}`;
    $("#status-list").innerHTML = [
      ["Nguồn", cam.source],
      ["Camera ID", s.camera_id],
      ["Độ phân giải", cam.width ? `${cam.width}×${cam.height}` : "—"],
      ["FPS camera / xử lý", `${cam.fps} / ${s.processing_fps}`],
      ["Chống giả mạo", s.liveness_enabled ? "Bật" : "Tắt"],
      ["Nhân viên", s.employees],
      ["Lỗi gần nhất", cam.last_error || "—"],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    $("#recent-list").innerHTML = s.recent.length
      ? s.recent.map((r) => `<li class="${esc(r.kind)}"><b>${esc(r.kind === "attendance" ? r.name : KIND_TEXT[r.kind])}</b>
          <span>${esc(KIND_TEXT[r.kind])} · ${timeOf(r.ts)} · khớp ${fixed(r.score)}</span></li>`).join("")
      : `<li class="empty">Chưa có</li>`;
    if (!datesReady) {
      datesReady = true;
      const day = s.now.slice(0, 10);
      ["#hist-date", "#csv-start", "#csv-end", "#rej-date"].forEach((id) => { $(id).value = day; });
    }
  } catch (e) {
    badge.textContent = "Mất kết nối máy chủ";
    badge.className = "badge bad";
  }
}

// ---------- nhân viên ----------
async function loadEmployees() {
  const list = await api("/api/employees");
  $("#emp-body").innerHTML = list.length
    ? list.map((e) => `<tr><td>${esc(e.code)}</td><td>${esc(e.name)}</td>
        <td><span class="pill ${e.sample_count >= 3 ? "ok" : "warn"}">${e.sample_count}</span></td>
        <td class="actions">
          <button data-act="enroll" data-id="${e.id}" data-name="${esc(e.name)}">Đăng ký khuôn mặt</button>
          <button data-act="delete" data-id="${e.id}" data-name="${esc(e.name)}" class="danger">Xoá</button>
        </td></tr>`).join("")
    : `<tr><td colspan="4" class="empty">Chưa có nhân viên</td></tr>`;
}

$("#add-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    await api("/api/employees", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: data.get("code"), name: data.get("name") }),
    });
    form.reset();
    toast("Đã thêm nhân viên", "ok");
    await loadEmployees();
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    button.disabled = false;
  }
});

$("#emp-body").addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const { id, name, act } = button.dataset;
  if (act === "enroll") return openEnroll(id, name);
  if (!confirm(`Xoá ${name}? Lịch sử chấm công được giữ lại; ảnh và đặc trưng khuôn mặt bị xoá vĩnh viễn.`)) return;
  try {
    await api(`/api/employees/${id}`, { method: "DELETE" });
    toast("Đã xoá nhân viên", "ok");
    await loadEmployees();
  } catch (e) {
    toast(e.message, "bad");
  }
});

// ---------- đăng ký khuôn mặt ----------
let enrollId = null;

function openEnroll(id, name) {
  enrollId = id;
  $("#enroll-title").textContent = `Đăng ký khuôn mặt — ${name}`;
  $("#enroll-msg").textContent = "Ngồi lùi để thấy trọn khuôn mặt (không bị cắt mép), đủ sáng, nhìn vào camera; "
    + "trong lúc chụp hơi xoay đầu nhẹ trái/phải để mẫu đa dạng. Nên chụp 2 lượt.";
  $("#enroll-img").src = "/stream.mjpg";
  $("#enroll-dialog").showModal();
  loadSamples().catch((e) => toast(e.message, "bad"));
}

$("#enroll-dialog").addEventListener("close", () => {
  $("#enroll-img").src = BLANK;
  enrollId = null;
  loadEmployees().catch(() => {});
});

async function loadSamples() {
  const samples = await api(`/api/employees/${enrollId}/samples`);
  $("#samples-grid").innerHTML = samples.length
    ? samples.map((s) => `<figure><img src="/files/${esc(s.image_path)}" alt="">
        <figcaption>${s.source === "camera" ? "Camera" : "Tải lên"}
        <button data-sample="${s.id}" class="icon danger" title="Xoá mẫu">✕</button></figcaption></figure>`).join("")
    : `<p class="empty">Chưa có mẫu</p>`;
}

$("#capture-btn").addEventListener("click", async () => {
  const button = $("#capture-btn");
  button.disabled = true;
  $("#enroll-msg").textContent = "Đang chụp… giữ yên, nhìn thẳng camera.";
  try {
    const result = await api(`/api/employees/${enrollId}/capture`, { method: "POST" });
    const skipped = result.reasons.map((r) => `${r.text} (${r.count})`).join("; ");
    $("#enroll-msg").textContent = `Đã lưu ${result.saved} mẫu.` + (skipped ? ` Bỏ qua: ${skipped}` : "");
    await loadSamples();
  } catch (e) {
    $("#enroll-msg").textContent = e.message;
  } finally {
    button.disabled = false;
  }
});

$("#upload-input").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  $("#enroll-msg").textContent = "Đang xử lý ảnh…";
  try {
    const results = await api(`/api/employees/${enrollId}/upload`, { method: "POST", body: form });
    $("#enroll-msg").innerHTML = results
      .map((r) => `<div class="${r.ok ? "ok" : "bad"}">${esc(r.filename)}: ${esc(r.message)}</div>`).join("");
    await loadSamples();
  } catch (e) {
    $("#enroll-msg").textContent = e.message;
  } finally {
    event.target.value = "";
  }
});

$("#samples-grid").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-sample]");
  if (!button || !confirm("Xoá mẫu này?")) return;
  try {
    await api(`/api/samples/${button.dataset.sample}`, { method: "DELETE" });
    await loadSamples();
  } catch (e) {
    toast(e.message, "bad");
  }
});

// ---------- lịch sử ----------
function card(path, title, meta) {
  const image = path ? `<img loading="lazy" src="/files/${esc(path)}" alt="">` : `<div class="noimg">Không còn ảnh</div>`;
  return `<figure class="card">${image}<figcaption><b>${title}</b><span>${meta}</span></figcaption></figure>`;
}

async function loadHistory() {
  const day = $("#hist-date").value;
  if (!day) return;
  const r = await api(`/api/attendance?date=${day}`);
  $("#summary-body").innerHTML = r.summary.length
    ? r.summary.map((s) => `<tr><td>${esc(s.code)}</td><td>${esc(s.name)}</td><td>${timeOf(s.first_ts)}</td>
        <td>${s.count > 1 ? timeOf(s.last_ts) : "—"}</td><td>${s.count}</td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">Không có lần chấm nào</td></tr>`;
  $("#events-grid").innerHTML = r.events.slice().reverse().map((e) => card(
    e.snapshot_path,
    `${esc(e.name)} <span class="muted">${esc(e.code)}</span>`,
    `${timeOf(e.ts)} · khớp ${fixed(e.score)} · thật ${fixed(e.liveness_score)}`,
  )).join("") || `<p class="empty">—</p>`;
}

async function loadRejected() {
  const day = $("#rej-date").value;
  if (!day) return;
  const r = await api(`/api/rejected?date=${day}`);
  $("#rejected-grid").innerHTML = r.events.map((e) => card(
    e.snapshot_path,
    esc(KIND_TEXT[e.reason] || e.reason),
    `${timeOf(e.ts)}${e.best_name ? ` · gần nhất: ${esc(e.best_name)} (${fixed(e.score)})` : ""} · thật ${fixed(e.liveness_score)}`,
  )).join("") || `<p class="empty">Không có</p>`;
}

$("#hist-date").addEventListener("change", () => loadHistory().catch((e) => toast(e.message, "bad")));
$("#hist-refresh").addEventListener("click", () => loadHistory().catch((e) => toast(e.message, "bad")));
$("#rej-date").addEventListener("change", () => loadRejected().catch((e) => toast(e.message, "bad")));
$("#rej-refresh").addEventListener("click", () => loadRejected().catch((e) => toast(e.message, "bad")));
$("#csv-btn").addEventListener("click", () => {
  location.href = `/api/attendance.csv?start=${$("#csv-start").value}&end=${$("#csv-end").value}`;
});

// ---------- khởi động ----------
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
refreshStatus().then(() => showTab("live"));
setInterval(refreshStatus, 1500);
```
**`cameraai/web/static/app.css`**

```css
:root {
  --bg: #f5f6f8; --panel: #fff; --text: #1d2330; --muted: #6b7385; --line: #e3e6ec;
  --accent: #2563eb; --accent-bg: #eef2ff;
  --ok: #15803d; --ok-bg: #dcfce7; --bad: #b91c1c; --bad-bg: #fee2e2; --warn: #a16207; --warn-bg: #fef3c7;
  --radius: 10px;
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); }
h3 { font-size: 14px; margin: 18px 0 8px; }
h4 { font-size: 13px; margin: 16px 0 8px; }
.muted { color: var(--muted); font-weight: 400; }

.topbar { display: flex; align-items: center; gap: 16px; padding: 10px 20px; background: var(--panel); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 5; flex-wrap: wrap; }
.brand { font-weight: 700; font-size: 16px; }
.tabs { display: flex; gap: 4px; flex: 1; flex-wrap: wrap; }
.tabs button { border: 0; background: none; padding: 8px 12px; border-radius: 8px; cursor: pointer; color: var(--muted); font: inherit; }
.tabs button.active { background: var(--accent-bg); color: var(--accent); font-weight: 600; }
.badge { padding: 4px 10px; border-radius: 999px; font-size: 12px; background: var(--line); }
.badge.ok { background: var(--ok-bg); color: var(--ok); }
.badge.bad { background: var(--bad-bg); color: var(--bad); }

main { padding: 20px; max-width: 1280px; margin: 0 auto; }
.tab { display: none; }
.tab.active { display: block; }

.live-grid { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px; align-items: start; }
.video-wrap { background: #111; border-radius: var(--radius); overflow: hidden; aspect-ratio: 16 / 9; display: flex; align-items: center; justify-content: center; }
.video-wrap img { width: 100%; height: 100%; object-fit: contain; }
.video-wrap.small { width: 360px; max-width: 100%; }

.panel { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); padding: 14px 16px; }
.panel h3:first-child { margin-top: 0; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 0; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; word-break: break-all; }
.recent { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.recent li { padding: 8px 10px; border-radius: 8px; background: var(--bg); display: flex; flex-direction: column; }
.recent li.attendance { background: var(--ok-bg); }
.recent li.spoof, .recent li.unknown { background: var(--bad-bg); }
.recent li span { font-size: 12px; color: var(--muted); }

.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.spacer { flex: 1; }
input { font: inherit; padding: 7px 10px; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--text); }
button, .file-btn { font: inherit; padding: 7px 12px; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--text); cursor: pointer; }
button:hover, .file-btn:hover { border-color: #c7ccd6; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.danger { color: var(--bad); }
button:disabled { opacity: .6; cursor: wait; }
button.icon { border: 0; background: none; padding: 2px 6px; }
.file-btn { text-align: center; }
.hint { color: var(--muted); font-size: 13px; }

.table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.table th, .table td { padding: 9px 12px; border-bottom: 1px solid var(--line); text-align: left; }
.table th { background: #fafbfc; color: var(--muted); font-weight: 600; font-size: 12px; }
.table td.actions { text-align: right; white-space: nowrap; }
.table td.actions button { margin-left: 6px; }
.empty { color: var(--muted); text-align: center; padding: 16px; }
.pill { display: inline-block; min-width: 28px; text-align: center; padding: 1px 8px; border-radius: 999px; font-size: 12px; }
.pill.ok { background: var(--ok-bg); color: var(--ok); }
.pill.warn { background: var(--warn-bg); color: var(--warn); }

.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
.card { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.card img, .card .noimg { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #eee; }
.card .noimg { display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 12px; }
.card figcaption { padding: 8px 10px; display: flex; flex-direction: column; gap: 2px; }
.card figcaption span { font-size: 12px; color: var(--muted); }

dialog { border: 0; border-radius: 14px; padding: 18px 20px; width: min(720px, calc(100vw - 32px)); box-shadow: 0 20px 60px rgba(0, 0, 0, .25); }
dialog::backdrop { background: rgba(15, 20, 30, .45); }
.dialog-head { display: flex; justify-content: space-between; align-items: center; }
.dialog-head h3 { margin: 0; }
.enroll-body { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
.enroll-actions { display: flex; flex-direction: column; gap: 10px; flex: 1; min-width: 200px; }
.msg { font-size: 13px; }
.msg .ok { color: var(--ok); }
.msg .bad { color: var(--bad); }
.thumbs { display: flex; flex-wrap: wrap; gap: 10px; }
.thumbs figure { margin: 0; width: 96px; }
.thumbs img { width: 96px; height: 96px; object-fit: cover; border-radius: 8px; display: block; }
.thumbs figcaption { font-size: 11px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; }

.toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); padding: 10px 16px; border-radius: 10px; background: #1d2330; color: #fff; z-index: 20; }
.toast.ok { background: var(--ok); }
.toast.bad { background: var(--bad); }

@media (max-width: 900px) { .live-grid { grid-template-columns: 1fr; } }
@media (max-width: 640px) {
  main { padding: 16px; }
  .topbar { padding: 10px 16px; }
  .table td.actions { white-space: normal; }
  .table td.actions button { margin: 4px 0 0 6px; }
}
```
- [ ] **Step 3:** `pytest tests/test_web.py` → PASS

## Task 13: Điểm chạy + test tay trên laptop

**Files:** Create `cameraai/__main__.py`

**`cameraai/__main__.py`**

```python
"""Chạy CameraAI: python -m cameraai [--source ...] [--config ...] [--no-liveness]"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from . import model_store
from .camera import VideoSource
from .config import is_loopback, load_config, make_clock, mask_source
from .enrollment import EnrollmentService
from .face_engine import FaceEngine
from .gallery import Gallery
from .logsetup import setup_console
from .pipeline import Pipeline
from .storage import Storage
from .web.app import Services, create_app

log = logging.getLogger("cameraai")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cameraai", description="Chấm công bằng camera nhận diện khuôn mặt")
    parser.add_argument("--config", help="đường dẫn file cấu hình (mặc định config.yaml)")
    parser.add_argument("--source", help="nguồn video: 0 = webcam, rtsp://..., hoặc đường dẫn file .mp4")
    parser.add_argument("--no-liveness", action="store_true", help="tắt kiểm tra mặt thật / giả")
    args = parser.parse_args(argv)
    setup_console()

    cfg = load_config(args.config)
    if args.source:
        cfg.camera.source = args.source
    if args.no_liveness:
        cfg.liveness.enabled = False

    if not is_loopback(cfg.web.host) and not cfg.web.admin_password:
        log.error("Nghe trên %s (máy khác truy cập được) thì bắt buộc đặt ADMIN_PASSWORD", cfg.web.host)
        return 2
    problems = model_store.verify(cfg.models_path)
    if problems:
        for problem in problems:
            log.error(problem)
        log.error("Chạy: %s", model_store.DOWNLOAD_HINT)
        return 1

    clock = make_clock(cfg.timezone)
    storage = Storage(cfg.data_path)
    gallery = Gallery(cfg.recognition.threshold)
    source = VideoSource(cfg.camera)
    pipeline = Pipeline(cfg, source, FaceEngine(cfg, gallery), storage, gallery, clock)
    enrollment = EnrollmentService(cfg, lambda: FaceEngine(cfg), source, storage, pipeline.reload_gallery, clock)
    app = create_app(Services(cfg, storage, source, pipeline, enrollment, clock))

    log.info(
        "Nguồn video: %s | camera_id: %s | chống giả mạo: %s",
        mask_source(cfg.camera.source), cfg.camera.id, "bật" if cfg.liveness.enabled else "tắt",
    )
    source.start()
    pipeline.start()
    shown_host = "localhost" if is_loopback(cfg.web.host) else "<địa-chỉ-IP-máy-này>"
    log.info("Mở trình duyệt: http://%s:%d  (Ctrl+C để dừng)", shown_host, cfg.web.port)
    try:
        uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        pipeline.stop()
        source.stop()
        storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 1:** `.\.venv\Scripts\python.exe -m pytest` → toàn bộ PASS
- [ ] **Step 2:** `.\.venv\Scripts\python.exe -m cameraai` → log "Mở trình duyệt: http://localhost:8000", đèn webcam sáng
- [ ] **Step 3:** Mở http://localhost:8000 → tab Trực tiếp có hình, badge "Camera: đang chạy"
- [ ] **Step 4:** Tab Nhân viên → thêm mã + tên → "Đăng ký khuôn mặt" → "Chụp từ camera" → "Đã lưu 5 mẫu"
- [ ] **Step 5:** Ngồi trước camera → khung xanh có tên, tab Lịch sử có 1 lần chấm kèm ảnh
- [ ] **Step 6:** Giơ ảnh khuôn mặt trên điện thoại → khung đỏ "Nghi gia mao", tab Bị từ chối có mục "Nghi giả mạo"

## Task 14: Docker + hướng dẫn triển khai

**Files:** Create `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`, `README.md`

**`Dockerfile`**

```dockerfile
# Image chạy CameraAI trên mini PC / máy chủ cùng mạng LAN với IP camera (RTSP).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAMERAAI_HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cameraai ./cameraai
COPY config.yaml .
# Tải model lúc build (kiểm SHA256) để container chạy được khi không có Internet.
RUN python -m cameraai.model_store download

EXPOSE 8000
CMD ["python", "-m", "cameraai"]
```
**`docker-compose.yml`**

```yaml
services:
  cameraai:
    build: .
    image: cameraai:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data   # DB, ảnh mẫu, ảnh chụp — chép từ laptop sang để giữ danh sách đã đăng ký
```
**`.dockerignore`**

```text
.venv
data
models
tests
docs
.env
**/__pycache__
.pytest_cache
```
**`.env.example`**

```text
# Chép thành .env rồi sửa cho đúng camera thật.
# Hikvision (luồng phụ): rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/102
# Dahua (luồng phụ):     rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=1
CAMERA_SOURCE=rtsp://admin:doi-mat-khau@192.168.1.64:554/Streaming/Channels/102
CAMERA_ID=cong-chinh
# Bắt buộc khi truy cập từ máy khác trong LAN (đăng nhập: tên tuỳ ý + mật khẩu này)
ADMIN_PASSWORD=doi-mat-khau-nay
# LIVENESS_ENABLED=true
```
**`README.md`**

````markdown
# CameraAI — Chấm công bằng camera

Nhận diện khuôn mặt nhân viên qua camera để ghi lần chấm công. Chạy thử trên laptop bằng webcam; khi triển khai
thật dùng IP camera (RTSP) — **chỉ đổi cấu hình nguồn video, không đổi code**.

- Phát hiện mặt: YuNet (MIT) · Nhận diện: SFace (Apache-2.0) · Chống giả mạo: MiniFASNetV2 (Apache-2.0)
- Chạy CPU, không cần GPU. Dữ liệu lưu cục bộ trong `data/` (SQLite + ảnh).
- Thiết kế: `docs/superpowers/specs/2026-10-26-cham-cong-camera-design.md`

## 1. Chạy trên laptop (Windows, PowerShell)

Yêu cầu: Python 3.12, webcam. Tại thư mục `D:\CameraAI`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m cameraai.model_store download
.\.venv\Scripts\python.exe -m cameraai
```

Mở http://localhost:8000. Dừng bằng `Ctrl+C`. Không cần kích hoạt venv (gọi thẳng `.venv\Scripts\python.exe`).

Tham số hữu ích:

| Lệnh | Tác dụng |
|---|---|
| `python -m cameraai --source 1` | dùng webcam thứ 2 |
| `python -m cameraai --source D:\clip.mp4` | phát lại video đã quay (test lặp lại được) |
| `python -m cameraai --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"` | thử IP camera thật ngay trên laptop |
| `python -m cameraai --no-liveness` | tắt chống giả mạo |

## 2. Dùng thử

1. Tab **Nhân viên** → nhập mã + tên → **Thêm nhân viên** (cần sự đồng ý của nhân viên trước khi đăng ký khuôn mặt).
2. Bấm **Đăng ký khuôn mặt** → **Chụp từ camera** (nhìn thẳng, đủ sáng, một người trong khung) hoặc **Tải ảnh lên**.
   Nên có ≥ 3 mẫu mỗi người.
3. Tab **Trực tiếp**: ngồi trước camera → khung **xanh** có tên = đã chấm. Cùng người không bị ghi lại trong 5 phút.
4. Giơ ảnh khuôn mặt trên điện thoại (nhìn thẳng, đủ gần) khoảng 1 giây → khung **đỏ** "Nghi gia mao", ghi vào tab
   **Bị từ chối**. Tay che mặt hoặc cử động mạnh cũng có thể bị đánh giá là giả.
5. Tab **Lịch sử**: giờ vào (lần đầu) / giờ ra (lần cuối) trong ngày, ảnh từng lần chấm, **Xuất CSV** (mở được bằng Excel).

Ý nghĩa khung trên hình: xanh = đã xác nhận · vàng = đang nhận diện · đỏ = giả mạo / người lạ ·
xám = bỏ qua ("Lai gan hon" mặt nhỏ, "Lui ra / vao giua khung hinh" mặt bị cắt mép, "Nhin thang camera" mặt
nghiêng, "Giu yen" ảnh mờ).

Mẹo khi test bằng laptop: ngồi cách webcam khoảng một sải tay để thấy trọn khuôn mặt, và nhìn vào webcam (nếu dùng
màn hình ngoài, webcam laptop sẽ thấy mặt bạn quay nghiêng). Khi đăng ký nên chụp 2 lượt, hơi xoay đầu nhẹ.

## 3. Test tự động

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Test logic không cần camera; test tích hợp model tự bỏ qua nếu chưa tải model.

## 4. Tinh chỉnh (`config.yaml`)

| Khoá | Mặc định | Khi nào đổi |
|---|---|---|
| `recognition.threshold` | 0.363 | Nhận nhầm người → tăng (0.40–0.45); hay không nhận ra → giảm nhẹ và đăng ký thêm mẫu |
| `recognition.min_face_px` | 64 | Camera đặt xa → giảm (chống giả mạo kém đi dưới 64 px) |
| `recognition.max_turn` | 0.25 | Hay bị "Nhin thang camera" dù đứng thẳng → tăng lên 0.3 |
| `recognition.min_sharpness` | 25 | Hay bị "Giu yen" → giảm |
| `liveness.threshold` | 0.5 | Frame đạt ngưỡng này mới được tính chấm công. Ảnh giả lọt qua → tăng (0.7–0.8) |
| `liveness.spoof_threshold` | 0.3 | Ghi "giả mạo" khi điểm trung bình 5 frame thấp hơn. Người thật hay bị ghi giả mạo → giảm (0.2) |
| `attendance.cooldown_minutes` | 5 | Khoảng tối thiểu giữa 2 lần ghi của cùng người |
| `storage.snapshot_retention_days` | 30 | Số ngày giữ ảnh chụp |

Biến môi trường ghi đè: `CAMERA_SOURCE`, `CAMERA_ID`, `CAMERAAI_HOST`, `CAMERAAI_PORT`, `ADMIN_PASSWORD`,
`CAMERAAI_DATA_DIR`, `LIVENESS_ENABLED`.

## 5. Triển khai với IP camera

Camera thường chỉ phát luồng video, nên CameraAI chạy trên một máy (mini PC / máy chủ) **cùng mạng LAN** với camera.

1. Bật RTSP trên camera, lấy URL **luồng phụ** (độ phân giải thấp, nhẹ CPU):
   - Hikvision / Ezviz đời cũ: `rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/102`
   - Dahua / Imou: `rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=1`
   - Model khác: xem tài liệu của hãng. Kiểm tra trước bằng VLC (Media → Open Network Stream) hoặc chạy thử
     `python -m cameraai --source "rtsp://..."` ngay trên laptop.
2. Chép thư mục dự án sang máy triển khai (kèm `data/` để giữ danh sách đã đăng ký).
3. Tạo `.env` từ `.env.example`, sửa `CAMERA_SOURCE`, `CAMERA_ID`, `ADMIN_PASSWORD`.
4. Chạy:
   ```bash
   docker compose up -d --build
   docker compose logs -f
   ```
5. Mở `http://<ip-máy-triển-khai>:8000` (đăng nhập: tên tuỳ ý + `ADMIN_PASSWORD`).

Không dùng Docker cũng được: cài Python 3.12 rồi làm như mục 1, đặt `CAMERA_SOURCE` / `CAMERAAI_HOST=0.0.0.0` /
`ADMIN_PASSWORD` trong biến môi trường. Mỗi tiến trình phụ trách **một** camera; nhiều camera thì chạy nhiều
container với `CAMERA_ID` và cổng khác nhau.

Sau khi lắp thật nên chỉnh lại `min_face_px`, `max_turn`, `min_sharpness` và `liveness.threshold` theo góc đặt
camera (đặt ngang tầm mặt, đủ sáng, tránh ngược sáng cho kết quả tốt nhất).

Lưu ý: Docker Desktop trên Windows không truy cập được webcam laptop — trên laptop hãy chạy trực tiếp bằng Python.

## 6. Dữ liệu và bảo mật

- Ảnh / đặc trưng khuôn mặt là dữ liệu cá nhân nhạy cảm (Nghị định 13/2023/NĐ-CP): cần nhân viên đồng ý trước
  khi đăng ký. Dữ liệu chỉ nằm trong `data/`.
- Người lạ và lần giả mạo cũng được chụp ảnh lưu lại (tab Bị từ chối); ảnh tự xoá sau `snapshot_retention_days`.
- Xoá nhân viên: giữ lịch sử chấm công, **xoá vĩnh viễn** mẫu khuôn mặt.
- Mặc định web chỉ nghe `127.0.0.1`. Mở ra LAN (`CAMERAAI_HOST=0.0.0.0`) thì bắt buộc `ADMIN_PASSWORD`.

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| "Mất kết nối camera" với webcam | Đóng ứng dụng khác đang dùng webcam (Teams, Zoom, Camera) |
| Thiếu model / sai SHA256 | `python -m cameraai.model_store download` |
| RTSP giật, trễ | Dùng luồng phụ; kiểm tra mạng; CameraAI tự kết nối lại khi mất tín hiệu |
| Không nhận ra người đã đăng ký | Xem nhãn xám trên khung mặt (mặt bị cắt mép / nghiêng / mờ); đăng ký thêm mẫu tại đúng camera và tư thế thường dùng, đủ sáng |

## 8. Nối Fastdo sau này

Mỗi lần chấm có cột `synced_at` (chưa gửi = NULL). `cameraai/sync.py` có sẵn `AttendanceSink` + `SyncWorker` gửi bù
các lần chấm chưa đồng bộ; chỉ cần viết một lớp `send()` gọi API Timekeeping của Fastdo và khởi động `SyncWorker`
trong `cameraai/__main__.py`, không sửa phần nhận diện.
````

- [ ] **Step 1:** (tuỳ chọn, cần Docker) `docker compose build` → build xong, model được tải trong image

---

## Kế hoạch kiểm thử

| Mức | Nội dung | Lệnh / cách làm |
|---|---|---|
| Unit | config, model_store, gallery, liveness preprocess, tracker/decider, storage, sync, camera (capture giả), imaging, pipeline (engine giả), enrollment (engine giả), web (TestClient) | `.\.venv\Scripts\python.exe -m pytest` |
| Tích hợp model | 3 model thật nạp và chạy, đầu ra đúng hình dạng | nằm trong `pytest`, tự bỏ qua nếu chưa tải model |
| Thủ công laptop | đăng ký → chấm công → giả mạo bị từ chối → xuất CSV | Task 13 |
| Phát lại video | chạy lại clip đã quay để so kết quả | `python -m cameraai --source clip.mp4` |
| Triển khai | RTSP thật, chỉnh ngưỡng theo góc camera | README mục triển khai |

## Prompt gốc của người dùng (cho plan này)

1. Trong thư mục `D:\CameraAI`, phát triển chức năng dùng camera để chấm công. Cần môi trường chạy và cách code/test trên laptop, sau đó đẩy lên camera dùng mà không phải thay đổi quá nhiều cài đặt. Trước mắt muốn chạy test trên máy tính.
2. Trả lời làm rõ: camera triển khai là IP camera thường (RTSP); kết quả lưu cục bộ, nối Fastdo sau; đăng ký khuôn mặt bằng cả hai cách (chụp camera + tải ảnh); có chống giả mạo, bật/tắt được; chọn hướng A (Python + OpenCV, CPU).
3. Duyệt thiết kế: "Duyệt, làm tiếp".
