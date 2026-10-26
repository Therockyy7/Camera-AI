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
