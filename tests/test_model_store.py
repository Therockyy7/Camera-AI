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
