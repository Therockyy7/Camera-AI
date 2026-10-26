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
