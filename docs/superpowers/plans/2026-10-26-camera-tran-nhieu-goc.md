# Camera trần nhiều góc (CameraAI) — Kế hoạch triển khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Mục tiêu:** Nhận diện được ở nhiều góc khi camera lắp trên trần ở cửa/hành lang: model nhận diện đổi được (SFace / ArcFace), chống giả mạo kiểu "kiểm khi đủ điều kiện", đăng ký nhiều góc, công cụ đo để chọn model.

**Kiến trúc:** `Recognizer` (Protocol) với 2 bản cài đặt; embedding lưu theo model trong `sample_embeddings` và được tính lại từ ảnh mẫu khi đổi model; `FaceEngine` tách cổng lọc nhận diện và cổng lọc chống giả; `AttendanceDecider` theo `liveness.mode`; `cameraai/evaluate.py` đo hai model trên cùng frame.

**Công nghệ:** như bản đầu + `w600k_r50.onnx` (InsightFace `buffalo_l`, chỉ đánh giá phi thương mại). Spec: `docs/superpowers/specs/2026-10-26-camera-tran-nhieu-goc-design.md`.

**Quy ước:** người dùng tự commit; lệnh chạy tại `D:\CameraAI` bằng `.\.venv\Scripts\python.exe`.

---

## Task 1: Cấu hình (model, liveness.mode, ngưỡng 2 mức)

**Files:** Modify `cameraai/config.py`, `config.yaml`; Test `tests/test_config.py`

- [ ] Test (đỏ) → cài đặt → `pytest tests/test_config.py` PASS
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
    path = write(tmp_path, "camera:\n  source: 0\n  id: cong-a\nprocessing:\n  fps: 5\nliveness:\n  mode: off\n")
    env = {
        "CAMERA_SOURCE": "rtsp://u:p@10.0.0.2/s",
        "LIVENESS_MODE": "required",
        "RECOGNITION_MODEL": "arcface",
        "CAMERAAI_PORT": "9000",
        "ADMIN_PASSWORD": "",
    }
    cfg = load_config(path, env=env)
    assert cfg.camera.source == "rtsp://u:p@10.0.0.2/s"
    assert cfg.camera.id == "cong-a"
    assert cfg.processing.fps == 5.0 and isinstance(cfg.processing.fps, float)
    assert cfg.liveness.mode == "required"
    assert cfg.recognition.model == "arcface"
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


def test_yaml_off_is_a_mode_not_false(tmp_path):
    # YAML 1.1 đọc `off` thành False; cấu hình phải hiểu là chế độ "off".
    assert load_config(write(tmp_path, "liveness:\n  mode: off\n"), env={}).liveness.mode == "off"


@pytest.mark.parametrize("text, key", [
    ("liveness:\n  mode: sometimes\n", "liveness.mode"),
    ("recognition:\n  model: facenet\n", "recognition.model"),
])
def test_invalid_choice_rejected(tmp_path, text, key):
    with pytest.raises(ValueError, match=key):
        load_config(write(tmp_path, text), env={})


def test_defaults_for_ceiling_camera():
    cfg = AppConfig()
    assert cfg.liveness.mode == "opportunistic" and cfg.recognition.model == "sface"
    assert cfg.recognition.max_turn > cfg.liveness.max_turn
    assert cfg.recognition.min_face_px < cfg.liveness.min_face_px


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
  # sface (mặc định, giấy phép thương mại được) | arcface (chính xác hơn ở góc nghiêng; CHỈ để đánh giá —
  # dùng thật cần mua license InsightFace). Tải arcface: python -m cameraai.model_store download --arcface
  model: sface
  sface_threshold: 0.363    # ngưỡng cosine; tăng để chặt hơn (ít nhận nhầm, dễ không nhận ra)
  arcface_threshold: 0.30   # tạm, chốt lại bằng: python -m cameraai.evaluate
  detect_score: 0.8     # độ tin cậy tối thiểu của bộ phát hiện mặt
  min_face_px: 40       # cạnh mặt tối thiểu để nhận diện (pixel trên frame gốc)
  min_sharpness: 12     # độ nét tối thiểu (phương sai Laplacian)
  max_turn: 0.45        # độ quay ngang tối đa để nhận diện (0 = nhìn thẳng); đo: 0,30–0,45 vẫn nhận tốt

liveness:
  # off | opportunistic (chỉ kiểm khi mặt đủ to và gần thẳng; không kiểm được vẫn chấm, ghi "chưa kiểm")
  # | required (chỉ chấm khi đã kiểm là mặt thật)
  mode: opportunistic
  threshold: 0.5        # frame có xác suất mặt thật >= ngưỡng này được coi là "thật"
  spoof_threshold: 0.3  # ghi "giả mạo" khi điểm TRUNG BÌNH 5 frame đã kiểm thấp hơn ngưỡng này
  min_face_px: 64       # model chống giả cần mặt >= 64 px
  max_turn: 0.25        # và gần thẳng (±30°)

attendance:
  window_frames: 5      # số quan sát gần nhất xét cho mỗi khuôn mặt
  confirm_frames: 3     # cần ít nhất chừng này quan sát khớp cùng một người
  cooldown_minutes: 5   # không ghi lại cùng người trong khoảng này

enrollment:
  samples_per_capture: 8   # mỗi lượt "Chụp từ camera"
  capture_timeout_s: 15
  min_face_px: 80          # mặt nhỏ hơn (hậu cảnh) không tính; camera trần ở xa nên giảm còn ~50
  min_det_score: 0.9
  max_similarity: 0.92     # chỉ nhận mẫu khác đủ so với mẫu đã có -> xoay đầu chậm khi chụp

storage:
  data_dir: data
  snapshot_retention_days: 30   # ảnh chụp cũ hơn số ngày này sẽ bị xoá

web:
  host: 127.0.0.1       # 0.0.0.0 để máy khác trong LAN truy cập (bắt buộc ADMIN_PASSWORD)
  port: 8000
```

## Task 2: Model ArcFace tuỳ chọn

**Files:** Modify `cameraai/model_store.py`; Test `tests/test_model_store.py`

- [ ] Test (đỏ) → cài đặt → PASS → `python -m cameraai.model_store download --arcface`
**`tests/test_model_store.py`**

```python
import hashlib
import zipfile

import pytest

from cameraai import model_store
from cameraai.config import AppConfig
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


def test_download_extracts_archive_member(tmp_path, monkeypatch):
    def retrieve(url, dest):
        with zipfile.ZipFile(dest, "w") as archive:
            archive.writestr("other.onnx", b"skip")
            archive.writestr("w600k.onnx", b"good")

    monkeypatch.setattr(model_store.urllib.request, "urlretrieve", retrieve)
    download(tmp_path, [ModelFile("m.onnx", "https://x/pack.zip", sha(b"good"), archive_member="w600k.onnx")])
    assert (tmp_path / "m.onnx").read_bytes() == b"good"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx"]


def test_required_models_include_arcface_only_when_selected():
    cfg = AppConfig()
    assert model_store.ARCFACE not in model_store.required_models(cfg)
    cfg.recognition.model = "arcface"
    assert model_store.ARCFACE in model_store.required_models(cfg)


def test_pinned_models_have_sha256_and_fixed_urls():
    for model in model_store.ALL_MODELS + model_store.OPTIONAL_MODELS:
        assert len(model.sha256) == 64
        assert ("/raw/" in model.url and "/main/" not in model.url) or "/releases/download/v" in model.url
```
**`cameraai/model_store.py`**

```python
"""Danh sách model ONNX (ghim theo commit/phiên bản + SHA256): tải về, kiểm tra, lấy đường dẫn.

Chạy: python -m cameraai.model_store download [--arcface]   (hoặc: verify)
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str  # SHA256 của file .onnx cuối cùng (đã giải nén nếu tải từ zip)
    archive_member: str | None = None  # tên file cần lấy trong zip


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

# ArcFace ResNet50 @ WebFace600K trong gói buffalo_l của InsightFace — CHỈ phi thương mại (đánh giá nội bộ).
ARCFACE = ModelFile(
    "w600k_r50.onnx",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    archive_member="w600k_r50.onnx",
)
OPTIONAL_MODELS: tuple[ModelFile, ...] = (ARCFACE,)
ARCFACE_LICENSE_NOTE = (
    "LƯU Ý GIẤY PHÉP: model ArcFace của InsightFace chỉ dùng cho nghiên cứu phi thương mại "
    "(\"ALL models are available for non-commercial research purposes only\"). Dùng để ĐÁNH GIÁ nội bộ; "
    "dùng thật cho sản phẩm thương mại phải mua license InsightFace hoặc giữ SFace."
)

DOWNLOAD_HINT = "python -m cameraai.model_store download"


def required_models(cfg) -> tuple[ModelFile, ...]:
    return ALL_MODELS + ((ARCFACE,) if cfg.recognition.model == "arcface" else ())


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
        if model.archive_member:
            archive = target.with_name(target.name + ".zip.part")
            try:
                urllib.request.urlretrieve(model.url, archive)
                with zipfile.ZipFile(archive) as zf, zf.open(model.archive_member) as src, part.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            finally:
                archive.unlink(missing_ok=True)
        else:
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
    with_arcface = "--arcface" in args
    if command not in ("download", "verify"):
        print("Cách dùng: python -m cameraai.model_store [download|verify] [--arcface]")
        return 2
    cfg = load_config()
    models = required_models(cfg) + ((ARCFACE,) if with_arcface and ARCFACE not in required_models(cfg) else ())
    if ARCFACE in models:
        print(ARCFACE_LICENSE_NOTE)
    if command == "download":
        download(cfg.models_path, models)
    problems = verify(cfg.models_path, models)
    for problem in problems:
        print(problem)
    if problems:
        print(f"Chạy: {DOWNLOAD_HINT}" + (" --arcface" if ARCFACE in models else ""))
        return 1
    print(f"Đủ {len(models)} model trong {cfg.models_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Task 3: Recognizer đổi được

**Files:** Create `cameraai/recognizers.py`; Test `tests/test_recognizers.py`

- [ ] Test (đỏ) → cài đặt → PASS (test ArcFace thật tự bỏ qua nếu chưa tải)
**`tests/test_recognizers.py`**

```python
import cv2
import numpy as np
import pytest

from cameraai import model_store
from cameraai.config import AppConfig
from cameraai.recognizers import ARCFACE_TEMPLATE, align_112, make_recognizer, similarity_transform

CFG = AppConfig()
ROW = np.array([200, 150, 160, 200, 240, 220, 320, 220, 280, 260, 250, 300, 310, 300, 0.99], np.float32)


def gradient_frame():
    y, x = np.mgrid[0:480, 0:640]
    return np.dstack([(x * 255 / 639), (y * 255 / 479), ((x + y) * 255 / 1118)]).astype(np.uint8)


def test_similarity_transform_recovers_rotation_scale_shift():
    angle = np.deg2rad(20)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = (ARCFACE_TEMPLATE @ rot.T) * 2.5 + np.array([50.0, 30.0])
    m = similarity_transform(src, ARCFACE_TEMPLATE)
    mapped = np.hstack([src, np.ones((5, 1))]) @ m.T
    assert np.allclose(mapped, ARCFACE_TEMPLATE, atol=1e-3)


def test_align_112_shape():
    assert align_112(gradient_frame(), ROW).shape == (112, 112, 3)


def test_unknown_recognizer_rejected():
    with pytest.raises(ValueError):
        make_recognizer(CFG, "facenet")


have_sface = not model_store.verify(CFG.models_path, [model_store.RECOGNIZER])
have_arcface = not model_store.verify(CFG.models_path, [model_store.ARCFACE])


@pytest.mark.skipif(not have_sface, reason="chưa tải SFace")
def test_our_alignment_matches_opencv_aligncrop():
    # Cùng template 5 điểm: căn chỉnh tự viết (cho ArcFace) phải gần như trùng alignCrop của OpenCV (cho SFace).
    opencv = cv2.FaceRecognizerSF.create(str(model_store.model_path(CFG.models_path, model_store.RECOGNIZER)), "")
    frame = gradient_frame()
    diff = np.abs(align_112(frame, ROW).astype(int) - opencv.alignCrop(frame, ROW).astype(int))
    assert diff.mean() < 2.0


@pytest.mark.skipif(not have_sface, reason="chưa tải SFace")
def test_sface_embedding_dim():
    assert make_recognizer(CFG, "sface").embed(gradient_frame(), ROW).shape == (128,)


@pytest.mark.skipif(not have_arcface, reason="chưa tải ArcFace (download --arcface)")
def test_arcface_embedding_dim():
    recognizer = make_recognizer(CFG, "arcface")
    embedding = recognizer.embed(gradient_frame(), ROW)
    assert recognizer.dim == 512 and embedding.shape == (512,) and np.isfinite(embedding).all()
```
**`cameraai/recognizers.py`**

```python
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
```

## Task 4: Embedding theo model + migration DB

**Files:** Modify `cameraai/storage.py`; Test `tests/test_storage.py`

- [ ] Test (đỏ, gồm migration từ DB bản đầu) → cài đặt → PASS
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
    sample_id = storage.add_sample(emp["id"], {"sface": emb}, JPEG, "camera", at(8))
    (sample,) = storage.list_samples(emp["id"])
    assert sample["id"] == sample_id and sample["source"] == "camera"
    assert storage.resolve_file(sample["image_path"]).read_bytes() == JPEG
    ((owner, vector),) = storage.gallery_samples("sface")
    assert owner == emp["id"] and np.array_equal(vector, emb)
    assert storage.get_employee(emp["id"])["sample_count"] == 1


def test_deactivate_removes_biometrics_keeps_history(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, JPEG, "upload", at(8))
    image = storage.resolve_file(storage.list_samples(emp["id"])[0]["image_path"])
    storage.add_attendance(emp["id"], "cam", at(8, 5), 0.8, 0.9, JPEG)
    assert storage.deactivate_employee(emp["id"]) is True
    assert storage.gallery_samples("sface") == [] and not image.exists()
    assert storage.list_employees() == []
    assert len(storage.attendance_on(at(8).date())) == 1
    assert storage.deactivate_employee(emp["id"]) is False
    again = storage.add_employee("NV01", "An mới", at(9))
    assert again["id"] == emp["id"] and again["name"] == "An mới"


def test_delete_sample(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    sample_id = storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, JPEG, "camera", at(8))
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


def test_embeddings_per_model_and_missing_list(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    first = storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, JPEG, "camera", at(8))
    second = storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, None, "camera", at(8))
    assert storage.gallery_samples("arcface") == []
    missing = storage.samples_missing_embedding("arcface")
    assert [m["id"] for m in missing] == [first, second] and missing[0]["image_path"]
    storage.add_embedding(first, "arcface", np.full(512, 0.5, np.float32))
    assert [m["id"] for m in storage.samples_missing_embedding("arcface")] == [second]
    ((owner, vector),) = storage.gallery_samples("arcface")
    assert owner == emp["id"] and vector.shape == (512,)
    assert len(storage.employee_embeddings(emp["id"], "sface")) == 2
    storage.delete_sample(first)
    assert storage.gallery_samples("arcface") == []


def test_migrates_first_version_database(tmp_path):
    import sqlite3

    data = tmp_path / "old"
    data.mkdir()
    db = sqlite3.connect(data / "cameraai.db")
    db.executescript(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,"
        " active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);"
        "CREATE TABLE face_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL,"
        " embedding BLOB NOT NULL, image_path TEXT, source TEXT NOT NULL, created_at TEXT NOT NULL);"
    )
    db.execute("INSERT INTO employees(code, name, active, created_at) VALUES ('NV01', 'An', 1, 'x')")
    db.execute(
        "INSERT INTO face_samples(employee_id, embedding, image_path, source, created_at) VALUES (1, ?, NULL, 'camera', 'x')",
        (np.arange(128, dtype=np.float32).tobytes(),),
    )
    db.commit()
    db.close()

    storage = Storage(data)
    ((owner, vector),) = storage.gallery_samples("sface")
    assert owner == 1 and np.array_equal(vector, np.arange(128, dtype=np.float32))
    assert storage.list_samples(1)[0]["source"] == "camera"
    storage.close()
    assert Storage(data).gallery_samples("sface")  # mở lại không migrate lần nữa


def test_resolve_file_blocks_escape(storage):
    assert storage.resolve_file("../secret.txt") is None
    assert storage.resolve_file("cameraai.db") is None
```
**`cameraai/storage.py`**

```python
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
```

## Task 5: FaceEngine — cổng lọc 2 mức

**Files:** Modify `cameraai/face_engine.py`, `cameraai/observation.py`; Test `tests/test_face_quality.py`, `tests/test_models_integration.py`

- [ ] Test (đỏ) → cài đặt → PASS
**`tests/test_face_quality.py`**

```python
import numpy as np

from cameraai.config import AppConfig
from cameraai.face_engine import liveness_checkable, reject_reason, touches_edge, turn_offset

FRAME_SHAPE = (720, 1280, 3)
CFG = AppConfig()


def test_reject_reason_order_and_ceiling_defaults():
    rec = CFG.recognition
    inside = (300, 200, 50, 60)  # mặt 50 px: nhận diện được (>= 40) dù chống giả cần 64
    assert reject_reason(inside, 0.3, 20.0, FRAME_SHAPE, rec.min_face_px, rec) is None
    assert reject_reason((300, 200, 30, 40), 0.0, 99.0, FRAME_SHAPE, rec.min_face_px, rec) == "small"
    assert reject_reason((1250, 200, 50, 60), 0.0, 99.0, FRAME_SHAPE, rec.min_face_px, rec) == "edge"
    assert reject_reason(inside, 0.5, 99.0, FRAME_SHAPE, rec.min_face_px, rec) == "turned"
    assert reject_reason(inside, 0.0, 5.0, FRAME_SHAPE, rec.min_face_px, rec) == "blurry"


def test_liveness_only_checked_on_large_frontal_faces():
    live = CFG.liveness
    assert liveness_checkable((300, 200, 120, 150), 0.1, live)
    assert not liveness_checkable((300, 200, 50, 60), 0.1, live)  # nhỏ
    assert not liveness_checkable((300, 200, 120, 150), 0.3, live)  # nghiêng quá ±30°


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
**`cameraai/face_engine.py`**

```python
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
    turn: float = 0.0  # độ quay ngang (0 = nhìn thẳng)
    reject_reason: str | None = None  # "small" | "edge" | "turned" | "blurry"
    liveness: float | None = None  # xác suất mặt thật; None = chưa kiểm (tắt, hoặc mặt nhỏ/nghiêng)
    is_real: bool | None = None
    embedding: np.ndarray | None = None
    match: Match | None = None

    @property
    def usable(self) -> bool:
        return self.reject_reason is None
```

## Task 6: Quyết định chấm công theo liveness.mode

**Files:** Modify `cameraai/attendance.py`; Test `tests/test_attendance.py`

- [ ] Test (đỏ) → cài đặt → PASS
**`tests/test_attendance.py`**

```python
from datetime import timedelta

import pytest
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
    d = AttendanceDecider(CFG, liveness_mode="required")
    assert summary(feed(d, [[make_obs(7)]] * 6, at(8))) == [(2, "attendance", 7)]


def test_majority_vote_picks_most_seen_employee():
    d = AttendanceDecider(CFG, liveness_mode="required")
    frames = [[make_obs(e)] for e in (7, 8, 7, 8, 7)]
    assert summary(feed(d, frames, at(8))) == [(4, "attendance", 7)]


def test_cooldown_blocks_new_track_then_allows_later():
    d = AttendanceDecider(CFG, liveness_mode="required")
    first = feed(d, [[make_obs(7)]] * 3, at(8, 0))
    again = feed(d, [[make_obs(7)]] * 3, at(8, 3))
    later = feed(d, [[make_obs(7)]] * 3, at(8, 6))
    assert len(first) == 1 and again == [] and len(later) == 1


def test_requires_real_faces_when_liveness_on():
    d = AttendanceDecider(CFG, liveness_mode="required")
    frames = [[make_obs(7, real=r)] for r in (True, False, True, False)]
    assert feed(d, frames, at(8)) == []


def test_spoof_recorded_once_after_full_window():
    d = AttendanceDecider(CFG, liveness_mode="required")
    assert summary(feed(d, [[make_obs(7, real=False)]] * 8, at(8))) == [(4, "spoof", 7)]


def test_spoof_uses_window_average():
    d = AttendanceDecider(CFG, liveness_mode="required", spoof_threshold=0.3)
    frames = [[make_obs(7, liveness=p)] for p in (0.2, 0.1, 0.4, 0.1, 0.2)]
    assert summary(feed(d, frames, at(8))) == [(4, "spoof", 7)]


def test_real_face_with_occasional_low_scores_not_spoof():
    d = AttendanceDecider(CFG, liveness_mode="required", spoof_threshold=0.3)
    frames = [[make_obs(7, liveness=p)] for p in (0.9, 0.1, 0.95, 0.05, 0.98, 0.2, 0.97)]
    assert summary(feed(d, frames, at(8))) == [(4, "attendance", 7)]


def test_looks_spoof_follows_current_window():
    d = AttendanceDecider(CFG, liveness_mode="required", spoof_threshold=0.3)
    tracker = FaceTracker(window=CFG.window_frames)
    frames = [[make_obs(7, liveness=0.05)]] * 5 + [[make_obs(7, liveness=0.95)]] * 5
    results = feed(d, frames[:5], at(8), tracker=tracker)
    (track,) = tracker.tracks
    assert d.looks_spoof(track) is True
    results += feed(d, frames[5:], at(8, 0, 1), tracker=tracker)
    assert summary(results) == [(4, "spoof", 7), (2, "attendance", 7)]
    assert d.looks_spoof(track) is False and track.employee_id == 7


def test_liveness_disabled_ignores_flag():
    d = AttendanceDecider(CFG, liveness_mode="off")
    assert summary(feed(d, [[make_obs(7, real=None)]] * 3, at(8))) == [(2, "attendance", 7)]


def test_unknown_after_full_window():
    d = AttendanceDecider(CFG, liveness_mode="required")
    frames = [[make_obs(None, score=0.2, nearest=3)]] * 7
    assert summary(feed(d, frames, at(8))) == [(4, "unknown", 3)]


def test_opportunistic_counts_unchecked_frames():
    # Camera trần: mặt nhỏ/xiên không kiểm được giả mạo (liveness None) nhưng vẫn được chấm.
    d = AttendanceDecider(CFG, liveness_mode="opportunistic")
    results = feed(d, [[make_obs(7, real=None)]] * 3, at(8))
    assert summary(results) == [(2, "attendance", 7)] and results[0][1].liveness is None


def test_required_ignores_unchecked_frames():
    d = AttendanceDecider(CFG, liveness_mode="required")
    assert feed(d, [[make_obs(7, real=None)]] * 5, at(8)) == []


def test_opportunistic_rejects_checked_spoof_frames():
    d = AttendanceDecider(CFG, liveness_mode="opportunistic")
    frames = [[make_obs(7, real=r)] for r in (None, False, None, False)]
    assert feed(d, frames, at(8)) == []  # chỉ 2 frame không bị kiểm là giả


def test_spoof_needs_full_window_of_checked_frames():
    d = AttendanceDecider(CFG, liveness_mode="opportunistic", spoof_threshold=0.3)
    frames = [[make_obs(None, score=0.2, real=r)] for r in (False, None, False, None, False, None, False)]
    assert all(kind != "spoof" for _, kind, _ in summary(feed(d, frames, at(8))))


def test_attendance_reports_mean_of_checked_liveness():
    d = AttendanceDecider(CFG, liveness_mode="opportunistic")
    frames = [[make_obs(7, liveness=0.8)], [make_obs(7, real=None)], [make_obs(7, liveness=1.0)]]
    ((_, decision),) = feed(d, frames, at(8))
    assert decision.liveness == pytest.approx(0.9)


def test_seed_restores_cooldown():
    d = AttendanceDecider(CFG, liveness_mode="required")
    d.seed({7: at(7, 58)})
    assert feed(d, [[make_obs(7)]] * 3, at(8)) == []
    assert d.last_recorded(7) == at(7, 58)
```
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
```

## Task 7: Đăng ký nhiều góc + tính lại embedding

**Files:** Modify `cameraai/enrollment.py`; Test `tests/test_enrollment.py`, `tests/helpers.py`

- [ ] Test (đỏ) → cài đặt → PASS
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
    def __init__(self, faces=None, liveness_enabled: bool = True, model_name: str = "sface", rows=None):
        self.faces = list(faces or [])
        self.liveness_enabled = liveness_enabled
        self.model_name = model_name
        self.rows = np.array([[10, 10, 50, 50] + [0] * 10 + [0.99]], np.float32) if rows is None else rows

    def analyze(self, frame, min_face_px=None):
        return list(self.faces)

    def detect(self, frame):
        return self.rows

    def embed(self, frame, row):
        return np.full(128, 0.5, np.float32)


class VaryingEngine(FakeEngine):
    """Mỗi frame trả 1 mặt với embedding ngẫu nhiên (các mẫu khác nhau, như khi xoay đầu)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(0)

    def analyze(self, frame, min_face_px=None):
        face = make_obs(None)
        face.embedding = self._rng.standard_normal(128).astype(np.float32)
        return [face]


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
**`tests/test_enrollment.py`**

```python
import cv2
import numpy as np
import pytest
from helpers import FakeEngine, FakeSource, VaryingEngine, at, make_obs

from cameraai.config import AppConfig
from cameraai.enrollment import EnrollmentError, EnrollmentService
from cameraai.storage import Storage


def build(tmp_path, engine, source=None, timeout=0.3):
    cfg = AppConfig()
    cfg.enrollment.capture_timeout_s = timeout
    storage = Storage(tmp_path)
    emp = storage.add_employee("NV01", "An", at(7))
    changes = []
    service = EnrollmentService(
        cfg, lambda: engine, source or FakeSource(), storage,
        lambda: changes.append(1), lambda: at(8), sleep=lambda s: None,
    )
    return service, storage, emp["id"], changes


def image_bytes(size, ext=".png"):
    ok, data = cv2.imencode(ext, np.zeros((size, size, 3), np.uint8))
    return data.tobytes()


def test_capture_saves_diverse_samples(tmp_path):
    service, storage, emp_id, changes = build(tmp_path, VaryingEngine())
    result = service.capture(emp_id)
    assert (result.saved, result.attempts, changes) == (8, 8, [1])
    assert [s["source"] for s in storage.list_samples(emp_id)] == ["camera"] * 8
    assert len(storage.employee_embeddings(emp_id, "sface")) == 8
    storage.close()


def test_capture_skips_samples_too_similar_to_existing(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(None)]))
    result = service.capture(emp_id)
    assert result.saved == 1 and set(result.reasons) == {"similar"}
    storage.close()


def test_capture_ignores_small_background_faces(tmp_path):
    faces = [make_obs(None), make_obs(None, box=(400, 50, 40, 40), reject_reason="small")]
    service, storage, emp_id, _ = build(tmp_path, FakeEngine(faces))
    result = service.capture(emp_id)
    assert result.saved == 1 and "many_faces" not in result.reasons
    storage.close()


def test_capture_only_small_faces_reports_small(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(None, reject_reason="small")]))
    assert set(service.capture(emp_id).reasons) == {"small"}
    storage.close()


def test_capture_rejects_two_large_faces(tmp_path):
    engine = FakeEngine([make_obs(), make_obs(box=(300, 100, 120, 120))])
    service, storage, emp_id, changes = build(tmp_path, engine)
    result = service.capture(emp_id)
    assert result.saved == 0 and set(result.reasons) == {"many_faces"} and changes == []
    storage.close()


def test_capture_rejects_spoof(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(real=False)]))
    assert set(service.capture(emp_id).reasons) == {"spoof"}
    storage.close()


def test_capture_without_frames(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs()]), source=FakeSource(empty=True), timeout=0.1)
    result = service.capture(emp_id)
    assert result.to_dict()["reasons"][0]["code"] == "no_camera" and result.saved == 0
    storage.close()


def test_capture_unknown_employee(tmp_path):
    service, storage, _, _ = build(tmp_path, FakeEngine([make_obs()]))
    with pytest.raises(EnrollmentError):
        service.capture(999)
    storage.close()


def test_upload_valid_image(tmp_path):
    service, storage, emp_id, changes = build(tmp_path, FakeEngine([make_obs()]))
    assert service.add_upload(emp_id, image_bytes(300)) is None
    assert storage.list_samples(emp_id)[0]["source"] == "upload" and changes == [1]
    storage.close()


def test_upload_rejects_garbage_and_low_score(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(det_score=0.5)]))
    assert service.add_upload(emp_id, b"not an image") == "not_image"
    assert service.add_upload(emp_id, image_bytes(50)) == "low_score"
    storage.close()


def test_reindex_builds_embeddings_for_new_model_from_sample_images(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine(model_name="arcface"))
    storage.add_sample(emp_id, {"sface": np.ones(128, np.float32)}, image_bytes(200, ".jpg"), "camera", at(8))
    storage.add_sample(emp_id, {"sface": np.ones(128, np.float32)}, None, "upload", at(8))  # không có ảnh
    assert service.reindex() == 1
    assert len(storage.gallery_samples("arcface")) == 1
    assert len(storage.samples_missing_embedding("arcface")) == 1
    storage.close()


def test_reindex_skips_image_without_face(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine(model_name="arcface", rows=np.zeros((0, 15), np.float32)))
    storage.add_sample(emp_id, {"sface": np.ones(128, np.float32)}, image_bytes(200, ".jpg"), "camera", at(8))
    assert service.reindex() == 0
    storage.close()
```
**`cameraai/enrollment.py`**

```python
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
```

## Task 8: Nối pipeline / web / điểm chạy

**Files:** Modify `cameraai/pipeline.py`, `cameraai/web/app.py`, `cameraai/web/static/app.js`, `cameraai/__main__.py`; Test `tests/test_pipeline.py`, `tests/test_web.py`

- [ ] Test (đỏ) → cài đặt → `pytest` toàn bộ PASS
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
**`tests/test_web.py`**

```python
import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import FakeSource, VaryingEngine, at

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
        cfg, VaryingEngine, source, storage, pipeline.reload_gallery, lambda: at(8),
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
    assert status["recognition_model"] == "sface" and status["liveness_mode"] == "opportunistic"
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
    assert client.post(f"/api/employees/{emp_id}/capture").json()["saved"] == 8
    samples = client.get(f"/api/employees/{emp_id}/samples").json()
    assert len(samples) == 9
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
            "recognition_model": services.cfg.recognition.model,
            "liveness_mode": services.cfg.liveness.mode,
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
**`cameraai/web/static/app.js`**

```javascript
"use strict";

const $ = (selector) => document.querySelector(selector);
const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
const KIND_TEXT = { attendance: "Chấm công", spoof: "Nghi giả mạo", unknown: "Người lạ" };
const LIVENESS_TEXT = { off: "Tắt", opportunistic: "Kiểm khi đủ điều kiện", required: "Bắt buộc" };
const livenessText = (value) => (value == null ? "chưa kiểm giả mạo" : `thật ${fixed(value)}`);

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
      ["Model nhận diện", s.recognition_model === "arcface" ? "ArcFace (chỉ đánh giá)" : "SFace"],
      ["Chống giả mạo", LIVENESS_TEXT[s.liveness_mode] || s.liveness_mode],
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
  $("#enroll-msg").textContent = "Đứng/ngồi để thấy trọn khuôn mặt (không bị cắt mép), đủ sáng. Bấm chụp rồi "
    + "XOAY ĐẦU CHẬM: trái, phải, ngẩng, cúi — chỉ mẫu khác góc mới được lưu. Nên chụp bằng chính camera lắp thật.";
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
    `${timeOf(e.ts)} · khớp ${fixed(e.score)} · ${livenessText(e.liveness_score)}`,
  )).join("") || `<p class="empty">—</p>`;
}

async function loadRejected() {
  const day = $("#rej-date").value;
  if (!day) return;
  const r = await api(`/api/rejected?date=${day}`);
  $("#rejected-grid").innerHTML = r.events.map((e) => card(
    e.snapshot_path,
    esc(KIND_TEXT[e.reason] || e.reason),
    `${timeOf(e.ts)}${e.best_name ? ` · gần nhất: ${esc(e.best_name)} (${fixed(e.score)})` : ""} · ${livenessText(e.liveness_score)}`,
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
from .recognizers import threshold_for
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
        cfg.liveness.mode = "off"

    if not is_loopback(cfg.web.host) and not cfg.web.admin_password:
        log.error("Nghe trên %s (máy khác truy cập được) thì bắt buộc đặt ADMIN_PASSWORD", cfg.web.host)
        return 2
    problems = model_store.verify(cfg.models_path, model_store.required_models(cfg))
    if problems:
        for problem in problems:
            log.error(problem)
        log.error("Chạy: %s%s", model_store.DOWNLOAD_HINT, " --arcface" if cfg.recognition.model == "arcface" else "")
        return 1
    if cfg.recognition.model == "arcface":
        log.warning(model_store.ARCFACE_LICENSE_NOTE)

    clock = make_clock(cfg.timezone)
    storage = Storage(cfg.data_path)
    gallery = Gallery(threshold_for(cfg, cfg.recognition.model))
    source = VideoSource(cfg.camera)
    pipeline = Pipeline(cfg, source, FaceEngine(cfg, gallery), storage, gallery, clock)
    enrollment = EnrollmentService(cfg, lambda: FaceEngine(cfg), source, storage, pipeline.reload_gallery, clock)
    enrollment.reindex()  # đổi model → tính embedding mới từ ảnh mẫu trước khi nạp gallery
    app = create_app(Services(cfg, storage, source, pipeline, enrollment, clock))

    log.info(
        "Nguồn video: %s | camera_id: %s | model: %s (ngưỡng %.3f) | chống giả mạo: %s",
        mask_source(cfg.camera.source), cfg.camera.id, cfg.recognition.model, gallery.threshold, cfg.liveness.mode,
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

## Task 9: Công cụ đo SFace vs ArcFace

**Files:** Create `cameraai/evaluate.py`; Test `tests/test_evaluate.py`

- [ ] Test (đỏ) → cài đặt → PASS
- [ ] Dừng server, `python -m cameraai.evaluate --seconds 30` trước webcam, xoay đầu nhiều góc → bảng so sánh
**`tests/test_evaluate.py`**

```python
import numpy as np
import pytest

from cameraai.evaluate import choose_target, size_bucket, summarize, turn_bucket, vertical_ratio


def landmarks(eye_y=100.0, nose_y=130.0, mouth_y=160.0):
    row = np.zeros(15, np.float32)
    row[5] = row[7] = eye_y
    row[9] = nose_y
    row[11] = row[13] = mouth_y
    return row


def test_vertical_ratio():
    assert vertical_ratio(landmarks()) == pytest.approx(0.5)
    assert vertical_ratio(landmarks(nose_y=150)) == pytest.approx(5 / 6)
    assert vertical_ratio(landmarks(mouth_y=100)) is None


def test_buckets():
    assert [turn_bucket(t) for t in (0.0, 0.2, 0.4, 0.6)] == ["0.00–0.15", "0.15–0.30", "0.30–0.45", "≥ 0.45"]
    assert [size_bucket(s) for s in (50, 100, 150, 250)] == ["< 64", "64–120", "120–200", "≥ 200"]


def test_summarize_reports_median_and_rate_per_model():
    records = [
        {"turn": 0.1, "scores": {"sface": 0.5, "arcface": 0.6}},
        {"turn": 0.1, "scores": {"sface": 0.3, "arcface": 0.5}},
        {"turn": 0.5, "scores": {"sface": 0.1, "arcface": 0.4}},
    ]
    rows = summarize(records, "turn", turn_bucket, {"sface": 0.363, "arcface": 0.3})
    assert rows[0] == ("0.00–0.15", 2, {"sface": (0.4, 0.5), "arcface": (0.55, 1.0)})
    assert rows[1] == ("≥ 0.45", 1, {"sface": (0.1, 0.0), "arcface": (0.4, 1.0)})


def test_summarize_orders_groups_by_value_not_text():
    records = [{"size": s, "scores": {"sface": 0.5}} for s in (250, 150, 100, 50)]
    rows = summarize(records, "size", size_bucket, {"sface": 0.363})
    assert [name for name, _, _ in rows] == ["< 64", "64–120", "120–200", "≥ 200"]


def test_choose_target():
    employees = [{"code": "MA01", "sample_count": 0}, {"code": "DEV01", "sample_count": 7}]
    assert choose_target(employees, None)["code"] == "DEV01"
    assert choose_target(employees, "MA01")["code"] == "MA01"
    with pytest.raises(SystemExit):
        choose_target(employees + [{"code": "X", "sample_count": 2}], None)
```
**`cameraai/evaluate.py`**

```python
"""Đo SFace và ArcFace trên cùng các frame để chọn model + ngưỡng (không ghi gì vào DB).

    python -m cameraai.evaluate [--source 0|rtsp://...|clip.mp4] [--seconds 30] [--employee MÃ] [--record clip.mp4]

Mặt lớn nhất trong frame được coi là người cần đo (nhân viên --employee); các mặt khác là hậu cảnh (người lạ),
dùng để xem điểm cao nhất của người lạ khi chọn ngưỡng. Dừng server trước nếu dùng webcam.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable

import cv2
import numpy as np

from . import model_store
from .camera import open_capture
from .config import load_config, mask_source, parse_source
from .face_engine import FaceEngine, reject_reason, sharpness, turn_offset
from .gallery import normalize
from .logsetup import setup_console
from .recognizers import Recognizer, make_recognizer, threshold_for
from .storage import Storage

REASON_NAMES = {"small": "mặt nhỏ", "edge": "chạm mép khung hình", "blurry": "mờ"}
TURN_EDGES = (0.15, 0.30, 0.45)
SIZE_EDGES = (64, 120, 200)
VERTICAL_EDGES = (0.45, 0.60, 0.75)


def vertical_ratio(row: np.ndarray) -> float | None:
    """(mũi − mắt) / (miệng − mắt) theo trục dọc. Khoảng 0,5 khi nhìn ngang; lớn dần khi camera nhìn từ trên xuống."""
    eye_y = (float(row[5]) + float(row[7])) / 2
    mouth_y = (float(row[11]) + float(row[13])) / 2
    span = mouth_y - eye_y
    return (float(row[9]) - eye_y) / span if span > 0 else None


def _bucket(value: float, edges: tuple, fmt: Callable[[float], str], open_start: bool) -> str:
    if value < edges[0]:
        return f"< {fmt(edges[0])}" if open_start else f"{fmt(0)}–{fmt(edges[0])}"
    for low, high in zip(edges, edges[1:]):
        if value < high:
            return f"{fmt(low)}–{fmt(high)}"
    return f"≥ {fmt(edges[-1])}"


def turn_bucket(turn: float) -> str:
    return _bucket(turn, TURN_EDGES, lambda v: f"{v:.2f}", open_start=False)


def size_bucket(size: float) -> str:
    return _bucket(size, SIZE_EDGES, lambda v: f"{v:.0f}", open_start=True)


def vertical_bucket(ratio: float) -> str:
    return _bucket(ratio, VERTICAL_EDGES, lambda v: f"{v:.2f}", open_start=True)


def summarize(records: list[dict], key: str, bucket: Callable[[float], str], thresholds: dict[str, float]):
    """[(nhóm, số frame, {model: (điểm trung vị, tỷ lệ >= ngưỡng)})] theo thứ tự nhóm."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        if record.get(key) is not None:
            groups.setdefault(bucket(record[key]), []).append(record)
    rows = []
    for name in sorted(groups, key=lambda g: min(r[key] for r in groups[g])):  # theo giá trị, không theo chuỗi
        items = groups[name]
        stats = {}
        for model, threshold in thresholds.items():
            scores = [r["scores"][model] for r in items if model in r["scores"]]
            if scores:
                stats[model] = (
                    round(statistics.median(scores), 3),
                    round(sum(s >= threshold for s in scores) / len(scores), 3),
                )
        rows.append((name, len(items), stats))
    return rows


def choose_target(employees: list[dict], code: str | None) -> dict:
    if code:
        for employee in employees:
            if employee["code"] == code:
                return employee
        raise SystemExit(f"Không có nhân viên mã {code}")
    enrolled = [e for e in employees if e["sample_count"] > 0]
    if len(enrolled) == 1:
        return enrolled[0]
    raise SystemExit("Cần --employee MÃ (người đứng trước camera). Đã đăng ký: "
                     + ", ".join(e["code"] for e in enrolled))


def _target_gallery(storage: Storage, engine: FaceEngine, recognizer: Recognizer, employee_id: int) -> np.ndarray:
    """Embedding các mẫu của nhân viên cho model này: lấy từ DB, thiếu thì tính từ ảnh mẫu (chỉ trong bộ nhớ)."""
    vectors = [normalize(e) for e in storage.employee_embeddings(employee_id, recognizer.name)]
    for sample in storage.samples_missing_embedding(recognizer.name):
        if sample["employee_id"] != employee_id or not sample["image_path"]:
            continue
        path = storage.resolve_file(sample["image_path"])
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if path else None
        rows = engine.detect(image) if image is not None else []
        if len(rows):
            vectors.append(normalize(recognizer.embed(image, max(rows, key=lambda r: r[2]))))
    return np.stack(vectors) if vectors else np.zeros((0, recognizer.dim), np.float32)


def _print_table(title: str, rows, models: list[str]) -> None:
    print(f"\n{title}")
    header = f"  {'nhóm':<12}{'frame':>6}" + "".join(f"{m + ' trung vị':>18}{m + ' ≥ngưỡng':>16}" for m in models)
    print(header)
    for name, count, stats in rows:
        cells = "".join(
            f"{stats[m][0]:>18.3f}{stats[m][1]:>15.0%} " if m in stats else f"{'-':>18}{'-':>16}" for m in models
        )
        print(f"  {name:<12}{count:>6}{cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cameraai.evaluate", description="So sánh SFace và ArcFace")
    parser.add_argument("--source", help="mặc định: camera.source trong config")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--employee", help="mã nhân viên đứng trước camera")
    parser.add_argument("--record", help="lưu frame gốc ra file .mp4 để chạy lại")
    args = parser.parse_args(argv)
    setup_console()

    cfg = load_config()
    source = args.source or cfg.camera.source
    names = ["sface"] + (["arcface"] if not model_store.verify(cfg.models_path, [model_store.ARCFACE]) else [])
    if "arcface" in names:
        print(model_store.ARCFACE_LICENSE_NOTE)
    else:
        print("Chưa có ArcFace (python -m cameraai.model_store download --arcface) — chỉ đo SFace.")

    storage = Storage(cfg.data_path)
    target = choose_target(storage.list_employees(), args.employee)
    engine = FaceEngine(cfg)  # chỉ dùng bộ phát hiện + các hàm chất lượng
    recognizers = {name: make_recognizer(cfg, name) for name in names}
    galleries = {name: _target_gallery(storage, engine, rec, target["id"]) for name, rec in recognizers.items()}
    storage.close()
    thresholds = {name: threshold_for(cfg, name) for name in names}
    print(f"Nhân viên: {target['code']} {target['name']} | mẫu: "
          + ", ".join(f"{n}={len(g)}" for n, g in galleries.items()) + f" | nguồn: {mask_source(source)}")
    if any(len(g) == 0 for g in galleries.values()):
        print("Nhân viên chưa có mẫu dùng được.")
        return 1

    cap = open_capture(parse_source(source), cfg.camera)
    if not cap.isOpened():
        print("Không mở được nguồn video (webcam đang bị server dùng?)")
        return 1
    writer = None
    records, background, skipped, frames = [], {n: [] for n in names}, Counter(), 0
    deadline = time.monotonic() + args.seconds
    print(f"Đang đo {args.seconds:.0f} giây… xoay đầu, ngẩng, cúi, đi lại gần/xa camera.")
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            if parse_source(source).kind == "file":
                break
            continue
        frames += 1
        if args.record:
            if writer is None:
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10, frame.shape[1::-1])
            writer.write(frame)
        rows = sorted(engine.detect(frame), key=lambda r: -r[2])
        for index, row in enumerate(rows):
            box = tuple(int(round(v)) for v in row[:4])
            turn = turn_offset(row)
            # Bỏ mặt nhỏ/chạm mép/mờ như ứng dụng; GIỮ mặt nghiêng để xem model chịu được tới đâu.
            reason = reject_reason(box, 0.0, sharpness(frame, box), frame.shape, cfg.recognition.min_face_px,
                                   cfg.recognition)
            if reason:
                if index == 0:
                    skipped[reason] += 1
                continue
            scores = {n: float((galleries[n] @ normalize(r.embed(frame, row))).max()) for n, r in recognizers.items()}
            if index == 0:
                records.append({"turn": turn, "size": min(box[2], box[3]), "vertical": vertical_ratio(row),
                                "scores": scores})
            else:
                for n, s in scores.items():
                    background[n].append(s)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Đã lưu clip: {args.record}")

    reasons = ", ".join(f"{REASON_NAMES.get(r, r)} {n}" for r, n in skipped.most_common()) or "không"
    print(f"\n{frames} frame; {len(records)} frame có mặt dùng được; mặt chính bị loại: {reasons}")
    print("Ngưỡng: " + ", ".join(f"{n}={t}" for n, t in thresholds.items()))
    if not records:
        return 1
    _print_table("Theo độ quay ngang (0 = nhìn thẳng)", summarize(records, "turn", turn_bucket, thresholds), names)
    _print_table("Theo cỡ mặt (px)", summarize(records, "size", size_bucket, thresholds), names)
    _print_table("Theo tỷ lệ dọc (≈0,5 ngang tầm; lớn hơn = nhìn từ trên xuống)",
                 summarize(records, "vertical", vertical_bucket, thresholds), names)
    _print_table("Tất cả", summarize(records, "turn", lambda _: "tất cả", thresholds), names)
    print("\nMặt hậu cảnh (người khác) — điểm khớp với nhân viên này, càng thấp càng tốt:")
    for name, scores in background.items():
        if scores:
            print(f"  {name}: {len(scores)} mặt, cao nhất {max(scores):.3f}, p95 {np.percentile(scores, 95):.3f}")
        else:
            print(f"  {name}: không có mặt hậu cảnh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Task 10: Tài liệu

**Files:** Modify `README.md`

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
| `recognition.model` | sface | `arcface` để đánh giá (xem mục 5) |
| `recognition.sface_threshold` / `arcface_threshold` | 0.363 / 0.30 | Nhận nhầm người → tăng; hay không nhận ra → đăng ký thêm mẫu nhiều góc trước, rồi mới giảm nhẹ. Chọn bằng `python -m cameraai.evaluate` |
| `recognition.min_face_px` | 40 | Cạnh mặt tối thiểu để nhận diện |
| `recognition.max_turn` | 0.45 | Hay bị "Nhin thang camera" → tăng (0.5); đo trên webcam: quay 0,30–0,45 vẫn nhận tốt |
| `recognition.min_sharpness` | 12 | Hay bị "Giu yen" → giảm |
| `liveness.mode` | opportunistic | `required`: chỉ chấm khi đã kiểm là mặt thật · `off`: tắt |
| `liveness.min_face_px` / `max_turn` | 64 / 0.25 | Vùng model chống giả còn tin cậy; ngoài vùng này frame là "chưa kiểm" |
| `liveness.threshold` | 0.5 | Frame đạt ngưỡng này được coi là thật. Ảnh giả lọt qua → tăng (0.7–0.8) |
| `liveness.spoof_threshold` | 0.3 | Ghi "giả mạo" khi điểm trung bình 5 frame đã kiểm thấp hơn. Người thật hay bị ghi giả mạo → giảm (0.2) |
| `enrollment.max_similarity` | 0.92 | Mẫu mới phải khác mẫu cũ; báo "Giống mẫu đã có" quá nhiều → tăng (0.95) |
| `processing.detect_width` | 640 | Camera trần, mặt nhỏ → 960–1280 (tốn CPU hơn) |
| `attendance.cooldown_minutes` | 5 | Khoảng tối thiểu giữa 2 lần ghi của cùng người |
| `storage.snapshot_retention_days` | 30 | Số ngày giữ ảnh chụp |

Biến môi trường ghi đè: `CAMERA_SOURCE`, `CAMERA_ID`, `CAMERAAI_HOST`, `CAMERAAI_PORT`, `ADMIN_PASSWORD`,
`CAMERAAI_DATA_DIR`, `LIVENESS_MODE`, `RECOGNITION_MODEL`.

## 5. Camera trần ở cửa / hành lang (nhiều góc)

**Vị trí lắp quyết định nhiều hơn model.** Theo hướng dẫn camera nhận diện của Hikvision: góc chúc xuống
15° ± 5°, lệch ngang < 25°, khoảng cách hai đồng tử trên ảnh > 40 px, camera 2MP phủ ≤ 3 m bề ngang. Quá ~35°
(nhìn từ trên xuống dốc) thì model nào cũng kém. Đặt camera ở chỗ người đi thẳng về phía nó (cửa, hành lang hẹp),
lùi điểm nhìn ra xa cửa để góc chúc nhỏ, đủ sáng, tránh ngược sáng.

- **Đăng ký bằng chính camera đó**: người cần đăng ký đứng ở cửa, bấm "Chụp từ camera" rồi xoay đầu chậm
  (trái, phải, ngẩng, cúi). Chỉ mẫu khác góc mới được lưu → gallery phủ nhiều góc.
- **Chống giả mạo** mặc định `opportunistic`: chỉ kiểm khi mặt ≥ 64 px và gần thẳng; mặt xa/xiên vẫn được chấm,
  ghi "chưa kiểm giả mạo" kèm ảnh để hậu kiểm (tab Lịch sử).
- **Mặt nhỏ**: tăng `processing.detect_width` lên 960–1280; dùng luồng chính (độ phân giải cao) nếu CPU đủ.

## 6. So sánh SFace và ArcFace

SFace (mặc định) có giấy phép dùng thương mại. ArcFace (InsightFace `buffalo_l`) thường chịu góc nghiêng và mặt
nhỏ tốt hơn nhưng **chỉ dùng cho nghiên cứu phi thương mại** — dùng thật cho sản phẩm phải mua license InsightFace.

```powershell
.\.venv\Scripts\python.exe -m cameraai.model_store download --arcface
```

Đo hai model trên cùng frame (dừng server trước nếu dùng webcam; đứng trước camera, xoay đầu, đi lại gần/xa):

```powershell
.\.venv\Scripts\python.exe -m cameraai.evaluate --seconds 40 --employee DEV01
```

Thêm `--record clip.mp4` để lưu đoạn quay, sau đó chạy lại `--source clip.mp4` cho các lần so sánh sau
(nên quay một đoạn từ camera trần thật). Bảng kết quả cho mỗi nhóm góc / cỡ mặt: điểm trung vị và tỷ lệ vượt
ngưỡng của từng model, kèm điểm cao nhất của người khác (hậu cảnh) để chọn ngưỡng an toàn.

Đổi model: sửa `recognition.model: arcface` trong `config.yaml` (hoặc `RECOGNITION_MODEL=arcface`) rồi khởi động
lại — embedding của model mới được tự tính từ ảnh mẫu đã lưu, không phải đăng ký lại.

## 7. Triển khai với IP camera

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

## 8. Dữ liệu và bảo mật

- Ảnh / đặc trưng khuôn mặt là dữ liệu cá nhân nhạy cảm (Nghị định 13/2023/NĐ-CP): cần nhân viên đồng ý trước
  khi đăng ký. Dữ liệu chỉ nằm trong `data/`.
- Người lạ và lần giả mạo cũng được chụp ảnh lưu lại (tab Bị từ chối); ảnh tự xoá sau `snapshot_retention_days`.
- Xoá nhân viên: giữ lịch sử chấm công, **xoá vĩnh viễn** mẫu khuôn mặt.
- Mặc định web chỉ nghe `127.0.0.1`. Mở ra LAN (`CAMERAAI_HOST=0.0.0.0`) thì bắt buộc `ADMIN_PASSWORD`.

## 9. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| "Mất kết nối camera" với webcam | Đóng ứng dụng khác đang dùng webcam (Teams, Zoom, Camera) |
| Thiếu model / sai SHA256 | `python -m cameraai.model_store download` |
| RTSP giật, trễ | Dùng luồng phụ; kiểm tra mạng; CameraAI tự kết nối lại khi mất tín hiệu |
| Không nhận ra người đã đăng ký | Xem nhãn xám trên khung mặt (mặt bị cắt mép / nghiêng / mờ); đăng ký thêm mẫu tại đúng camera và tư thế thường dùng, đủ sáng |

## 10. Nối Fastdo sau này

Mỗi lần chấm có cột `synced_at` (chưa gửi = NULL). `cameraai/sync.py` có sẵn `AttendanceSink` + `SyncWorker` gửi bù
các lần chấm chưa đồng bộ; chỉ cần viết một lớp `send()` gọi API Timekeeping của Fastdo và khởi động `SyncWorker`
trong `cameraai/__main__.py`, không sửa phần nhận diện.
````

---

## Kế hoạch kiểm thử

| Mức | Nội dung | Cách chạy |
|---|---|---|
| Unit | config, model_store (zip member), Umeyama, migration + embedding theo model, gating, decider 3 chế độ, enrollment đa dạng/hậu cảnh/tính lại, bảng evaluate | `.\.venv\Scripts\python.exe -m pytest` |
| Tích hợp model | SFace + ArcFace thật nạp và cho embedding đúng chiều | trong `pytest`, tự bỏ qua nếu thiếu model |
| Đo thật | webcam ở nhiều góc: SFace vs ArcFace | `python -m cameraai.evaluate --seconds 30` |
| Triển khai | quay clip từ camera trần rồi `--source clip.mp4` | README |

## Prompt gốc của người dùng (cho plan này)

1. Bản hiện tại phải nhìn trực diện mới nhận diện được; muốn nhận diện ở nhiều góc độ vì sau này camera lắp phía trên trần nhà và ở xa.
2. Trả lời làm rõ: camera đặt ở cửa ra vào / hành lang, người đi về phía camera; cho phép đổi model, đo ArcFace so với SFace rồi chọn (ArcFace chỉ đánh giá nội bộ); chống giả mạo kiểm khi đủ điều kiện.
3. Duyệt thiết kế: "Duyệt, làm tiếp" (tải ArcFace để đo).
