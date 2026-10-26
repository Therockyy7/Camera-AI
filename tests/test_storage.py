import numpy as np
import pytest
from helpers import at

from cameraai.storage import DuplicateEmployeeError, Storage

JPEG = b"\xff\xd8fake-jpeg"


@pytest.fixture
def storage(tmp_path):
    s = Storage(tmp_path / "data")
    yield s
    s.close()


def test_add_and_list_employees(storage):
    a = storage.add_employee(" NV01 ", "Nguyễn Văn An", at(8))
    assert a["code"] == "NV01" and a["sample_count"] == 0
    with pytest.raises(DuplicateEmployeeError):
        storage.add_employee("NV01", "Khác", at(8))
    with pytest.raises(ValueError):
        storage.add_employee("", "Tên", at(8))
    storage.add_employee("NV00", "Bình", at(8))
    assert [e["code"] for e in storage.list_employees()] == ["NV00", "NV01"]


def test_samples_roundtrip_and_gallery(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    emb = np.arange(128, dtype=np.float32)
    sample_id = storage.add_sample(emp["id"], {"sface": emb}, JPEG, "camera", at(8))
    (sample,) = storage.list_samples(emp["id"])
    assert sample["id"] == sample_id and sample["source"] == "camera"
    assert storage.resolve_file(sample["image_path"]).read_bytes() == JPEG
    ((owner, vector),) = storage.gallery_samples("sface")
    assert owner == emp["id"] and np.array_equal(vector, emb)
    assert storage.get_employee(emp["id"])["sample_count"] == 1


def test_deactivate_removes_biometrics_keeps_history(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, JPEG, "upload", at(8))
    image = storage.resolve_file(storage.list_samples(emp["id"])[0]["image_path"])
    storage.add_attendance(emp["id"], "cam", at(8, 5), 0.8, 0.9, JPEG)
    assert storage.deactivate_employee(emp["id"]) is True
    assert storage.gallery_samples("sface") == [] and not image.exists()
    assert storage.list_employees() == []
    assert len(storage.attendance_on(at(8).date())) == 1
    assert storage.deactivate_employee(emp["id"]) is False
    again = storage.add_employee("NV01", "An mới", at(9))
    assert again["id"] == emp["id"] and again["name"] == "An mới"


def test_delete_sample(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    sample_id = storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, JPEG, "camera", at(8))
    assert storage.delete_sample(sample_id) is True
    assert storage.list_samples(emp["id"]) == []
    assert storage.delete_sample(sample_id) is False


def test_daily_summary_and_day_boundaries(storage):
    a = storage.add_employee("NV01", "An", at(7))
    b = storage.add_employee("NV02", "Bình", at(7))
    for ts in (at(8, 1), at(12, 0), at(17, 30)):
        storage.add_attendance(a["id"], "cam", ts, 0.8, 0.9, None)
    storage.add_attendance(b["id"], "cam", at(9, 15), 0.7, None, JPEG)
    storage.add_attendance(a["id"], "cam", at(8, 0, day=27), 0.8, 0.9, None)
    summary = storage.daily_summary(at(0).date())
    assert [(s["code"], s["first_ts"][11:16], s["last_ts"][11:16], s["count"]) for s in summary] == [
        ("NV01", "08:01", "17:30", 3),
        ("NV02", "09:15", "09:15", 1),
    ]
    assert len(storage.attendance_on(at(0).date())) == 4
    assert len(storage.attendance_between(at(0).date(), at(0, day=27).date())) == 5


def test_rejected_and_last_times(storage):
    a = storage.add_employee("NV01", "An", at(7))
    storage.add_rejected("cam", at(8), "spoof", a["id"], 0.6, 0.1, JPEG)
    storage.add_rejected("cam", at(8, 1), "unknown", None, 0.2, 0.9, None)
    rows = storage.rejected_on(at(8).date())
    assert [(r["reason"], r["best_name"]) for r in rows] == [("unknown", None), ("spoof", "An")]
    storage.add_attendance(a["id"], "cam", at(8, 2), 0.8, 0.9, None)
    storage.add_attendance(a["id"], "other", at(8, 3), 0.8, 0.9, None)
    assert storage.last_attendance_times("cam", at(8)) == {a["id"]: at(8, 2)}


def test_pending_and_mark_synced(storage):
    a = storage.add_employee("NV01", "An", at(7))
    first = storage.add_attendance(a["id"], "cam", at(8), 0.8, 0.9, None)
    second = storage.add_attendance(a["id"], "cam", at(9), 0.8, 0.9, None)
    pending = storage.pending_events()
    assert [e["id"] for e in pending] == [first, second] and pending[0]["employee_code"] == "NV01"
    storage.mark_synced([first], at(10))
    assert [e["id"] for e in storage.pending_events()] == [second]


def test_cleanup_removes_old_snapshots(storage):
    a = storage.add_employee("NV01", "An", at(7))
    storage.add_attendance(a["id"], "cam", at(8, day=1), 0.8, 0.9, JPEG)
    storage.add_attendance(a["id"], "cam", at(8, day=26), 0.8, 0.9, JPEG)
    storage.add_rejected("cam", at(8, day=1), "unknown", None, 0.1, 0.9, JPEG)
    old_file = storage.resolve_file(storage.attendance_on(at(8, day=1).date())[0]["snapshot_path"])
    assert storage.cleanup(at(0, day=20)) == 2
    assert not old_file.exists()
    assert storage.attendance_on(at(8, day=1).date())[0]["snapshot_path"] is None
    assert storage.rejected_on(at(8, day=1).date()) == []
    assert storage.attendance_on(at(8).date())[0]["snapshot_path"] is not None


def test_embeddings_per_model_and_missing_list(storage):
    emp = storage.add_employee("NV01", "An", at(8))
    first = storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, JPEG, "camera", at(8))
    second = storage.add_sample(emp["id"], {"sface": np.ones(128, np.float32)}, None, "camera", at(8))
    assert storage.gallery_samples("arcface") == []
    missing = storage.samples_missing_embedding("arcface")
    assert [m["id"] for m in missing] == [first, second] and missing[0]["image_path"]
    storage.add_embedding(first, "arcface", np.full(512, 0.5, np.float32))
    assert [m["id"] for m in storage.samples_missing_embedding("arcface")] == [second]
    ((owner, vector),) = storage.gallery_samples("arcface")
    assert owner == emp["id"] and vector.shape == (512,)
    assert len(storage.employee_embeddings(emp["id"], "sface")) == 2
    storage.delete_sample(first)
    assert storage.gallery_samples("arcface") == []


def test_migrates_first_version_database(tmp_path):
    import sqlite3

    data = tmp_path / "old"
    data.mkdir()
    db = sqlite3.connect(data / "cameraai.db")
    db.executescript(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,"
        " active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);"
        "CREATE TABLE face_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL,"
        " embedding BLOB NOT NULL, image_path TEXT, source TEXT NOT NULL, created_at TEXT NOT NULL);"
    )
    db.execute("INSERT INTO employees(code, name, active, created_at) VALUES ('NV01', 'An', 1, 'x')")
    db.execute(
        "INSERT INTO face_samples(employee_id, embedding, image_path, source, created_at) VALUES (1, ?, NULL, 'camera', 'x')",
        (np.arange(128, dtype=np.float32).tobytes(),),
    )
    db.commit()
    db.close()

    storage = Storage(data)
    ((owner, vector),) = storage.gallery_samples("sface")
    assert owner == 1 and np.array_equal(vector, np.arange(128, dtype=np.float32))
    assert storage.list_samples(1)[0]["source"] == "camera"
    storage.close()
    assert Storage(data).gallery_samples("sface")  # mở lại không migrate lần nữa


def test_resolve_file_blocks_escape(storage):
    assert storage.resolve_file("../secret.txt") is None
    assert storage.resolve_file("cameraai.db") is None
