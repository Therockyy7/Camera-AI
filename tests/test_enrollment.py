import cv2
import numpy as np
import pytest
from helpers import FakeEngine, FakeSource, VaryingEngine, at, make_obs

from cameraai.config import AppConfig
from cameraai.enrollment import EnrollmentError, EnrollmentService
from cameraai.storage import Storage


def build(tmp_path, engine, source=None, timeout=0.3):
    cfg = AppConfig()
    cfg.enrollment.capture_timeout_s = timeout
    storage = Storage(tmp_path)
    emp = storage.add_employee("NV01", "An", at(7))
    changes = []
    service = EnrollmentService(
        cfg, lambda: engine, source or FakeSource(), storage,
        lambda: changes.append(1), lambda: at(8), sleep=lambda s: None,
    )
    return service, storage, emp["id"], changes


def image_bytes(size, ext=".png"):
    ok, data = cv2.imencode(ext, np.zeros((size, size, 3), np.uint8))
    return data.tobytes()


def test_capture_saves_diverse_samples(tmp_path):
    service, storage, emp_id, changes = build(tmp_path, VaryingEngine())
    result = service.capture(emp_id)
    assert (result.saved, result.attempts, changes) == (8, 8, [1])
    assert [s["source"] for s in storage.list_samples(emp_id)] == ["camera"] * 8
    assert len(storage.employee_embeddings(emp_id, "sface")) == 8
    storage.close()


def test_capture_skips_samples_too_similar_to_existing(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(None)]))
    result = service.capture(emp_id)
    assert result.saved == 1 and set(result.reasons) == {"similar"}
    storage.close()


def test_capture_ignores_small_background_faces(tmp_path):
    faces = [make_obs(None), make_obs(None, box=(400, 50, 40, 40), reject_reason="small")]
    service, storage, emp_id, _ = build(tmp_path, FakeEngine(faces))
    result = service.capture(emp_id)
    assert result.saved == 1 and "many_faces" not in result.reasons
    storage.close()


def test_capture_only_small_faces_reports_small(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(None, reject_reason="small")]))
    assert set(service.capture(emp_id).reasons) == {"small"}
    storage.close()


def test_capture_rejects_two_large_faces(tmp_path):
    engine = FakeEngine([make_obs(), make_obs(box=(300, 100, 120, 120))])
    service, storage, emp_id, changes = build(tmp_path, engine)
    result = service.capture(emp_id)
    assert result.saved == 0 and set(result.reasons) == {"many_faces"} and changes == []
    storage.close()


def test_capture_rejects_spoof(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(real=False)]))
    assert set(service.capture(emp_id).reasons) == {"spoof"}
    storage.close()


def test_capture_without_frames(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs()]), source=FakeSource(empty=True), timeout=0.1)
    result = service.capture(emp_id)
    assert result.to_dict()["reasons"][0]["code"] == "no_camera" and result.saved == 0
    storage.close()


def test_capture_unknown_employee(tmp_path):
    service, storage, _, _ = build(tmp_path, FakeEngine([make_obs()]))
    with pytest.raises(EnrollmentError):
        service.capture(999)
    storage.close()


def test_upload_valid_image(tmp_path):
    service, storage, emp_id, changes = build(tmp_path, FakeEngine([make_obs()]))
    assert service.add_upload(emp_id, image_bytes(300)) is None
    assert storage.list_samples(emp_id)[0]["source"] == "upload" and changes == [1]
    storage.close()


def test_upload_rejects_garbage_and_low_score(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine([make_obs(det_score=0.5)]))
    assert service.add_upload(emp_id, b"not an image") == "not_image"
    assert service.add_upload(emp_id, image_bytes(50)) == "low_score"
    storage.close()


def test_reindex_builds_embeddings_for_new_model_from_sample_images(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine(model_name="arcface"))
    storage.add_sample(emp_id, {"sface": np.ones(128, np.float32)}, image_bytes(200, ".jpg"), "camera", at(8))
    storage.add_sample(emp_id, {"sface": np.ones(128, np.float32)}, None, "upload", at(8))  # không có ảnh
    assert service.reindex() == 1
    assert len(storage.gallery_samples("arcface")) == 1
    assert len(storage.samples_missing_embedding("arcface")) == 1
    storage.close()


def test_reindex_skips_image_without_face(tmp_path):
    service, storage, emp_id, _ = build(tmp_path, FakeEngine(model_name="arcface", rows=np.zeros((0, 15), np.float32)))
    storage.add_sample(emp_id, {"sface": np.ones(128, np.float32)}, image_bytes(200, ".jpg"), "camera", at(8))
    assert service.reindex() == 0
    storage.close()
