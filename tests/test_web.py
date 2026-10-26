import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import FakeSource, VaryingEngine, at

from cameraai.config import AppConfig
from cameraai.enrollment import EnrollmentService
from cameraai.storage import Storage
from cameraai.web.app import Services, create_app


class FakePipeline:
    processed_fps = 7.5

    def __init__(self):
        self.reloads = 0

    def latest_jpeg(self):
        return 1, b"\xff\xd8jpeg"

    def recent_events(self):
        return [{"kind": "attendance", "name": "An", "ts": "2026-10-26T08:00:00+07:00", "score": 0.8, "liveness": 0.9}]

    def reload_gallery(self):
        self.reloads += 1


def make_ctx(tmp_path, password=""):
    cfg = AppConfig()
    cfg.web.admin_password = password
    storage = Storage(tmp_path / "data")
    pipeline = FakePipeline()
    source = FakeSource()
    enrollment = EnrollmentService(
        cfg, VaryingEngine, source, storage, pipeline.reload_gallery, lambda: at(8),
        sleep=lambda s: None,
    )
    client = TestClient(create_app(Services(cfg, storage, source, pipeline, enrollment, lambda: at(8))))
    return client, storage, pipeline


@pytest.fixture
def ctx(tmp_path):
    client, storage, pipeline = make_ctx(tmp_path)
    yield client, storage, pipeline
    storage.close()


def add_employee(client, code="NV01", name="An"):
    response = client.post("/api/employees", json={"code": code, "name": name})
    assert response.status_code == 201
    return response.json()["id"]


def test_index_and_status(ctx):
    client, _, _ = ctx
    assert "CameraAI" in client.get("/").text
    status = client.get("/api/status").json()
    assert status["camera"]["connected"] is True and status["processing_fps"] == 7.5
    assert status["recognition_model"] == "sface" and status["liveness_mode"] == "opportunistic"
    assert status["recent"][0]["name"] == "An" and status["now"].startswith("2026-10-26T08:00")


def test_employee_crud(ctx):
    client, _, pipeline = ctx
    emp_id = add_employee(client)
    assert client.post("/api/employees", json={"code": "NV01", "name": "X"}).status_code == 409
    assert client.post("/api/employees", json={"code": " ", "name": "X"}).status_code == 400
    assert [e["code"] for e in client.get("/api/employees").json()] == ["NV01"]
    assert client.delete(f"/api/employees/{emp_id}").status_code == 200
    assert client.delete(f"/api/employees/{emp_id}").status_code == 404
    assert pipeline.reloads == 1


def test_upload_capture_and_samples(ctx):
    client, _, _ = ctx
    emp_id = add_employee(client)
    ok, png = cv2.imencode(".png", np.zeros((300, 300, 3), np.uint8))
    files = [("files", ("a.png", png.tobytes(), "image/png")), ("files", ("b.txt", b"hello", "text/plain"))]
    results = client.post(f"/api/employees/{emp_id}/upload", files=files).json()
    assert [r["ok"] for r in results] == [True, False]
    assert client.post(f"/api/employees/{emp_id}/capture").json()["saved"] == 8
    samples = client.get(f"/api/employees/{emp_id}/samples").json()
    assert len(samples) == 9
    assert client.get(f"/files/{samples[0]['image_path']}").status_code == 200
    assert client.delete(f"/api/samples/{samples[0]['id']}").status_code == 200
    assert client.post("/api/employees/999/capture").status_code == 404


def test_attendance_csv_and_rejected(ctx):
    client, storage, _ = ctx
    emp = storage.add_employee("NV01", "Nguyễn Văn An", at(7))
    storage.add_attendance(emp["id"], "laptop", at(8, 1), 0.81, 0.93, b"\xff\xd8x")
    storage.add_rejected("laptop", at(8, 2), "spoof", emp["id"], 0.6, 0.1, None)
    body = client.get("/api/attendance", params={"date": "2026-10-26"}).json()
    assert body["summary"][0]["count"] == 1 and body["events"][0]["code"] == "NV01"
    csv_response = client.get("/api/attendance.csv", params={"start": "2026-10-26", "end": "2026-10-26"})
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert csv_response.text.startswith("﻿Mã NV") and "Nguyễn Văn An" in csv_response.text
    assert client.get("/api/rejected", params={"date": "2026-10-26"}).json()["events"][0]["reason"] == "spoof"
    assert client.get("/api/attendance", params={"date": "26/10/2026"}).status_code == 400


def test_files_blocks_database(ctx):
    client, _, _ = ctx
    assert client.get("/files/cameraai.db").status_code == 404


def test_basic_auth(tmp_path):
    client, storage, _ = make_ctx(tmp_path, password="bi-mat")
    assert client.get("/api/status").status_code == 401
    good = base64.b64encode(b"admin:bi-mat").decode()
    bad = base64.b64encode(b"admin:sai").decode()
    assert client.get("/api/status", headers={"Authorization": f"Basic {good}"}).status_code == 200
    assert client.get("/api/status", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    storage.close()
