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
