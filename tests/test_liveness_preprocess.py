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
