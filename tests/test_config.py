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
