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
