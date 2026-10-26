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
