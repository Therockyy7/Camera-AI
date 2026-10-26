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
