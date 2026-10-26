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
