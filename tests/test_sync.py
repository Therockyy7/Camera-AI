from helpers import at

from cameraai.storage import Storage
from cameraai.sync import SyncWorker


class FakeSink:
    def __init__(self, fail=False, accept=None):
        self.fail = fail
        self.accept = accept
        self.batches = []

    def send(self, events):
        if self.fail:
            raise ConnectionError("mất mạng")
        self.batches.append(events)
        ids = [e["id"] for e in events]
        return ids if self.accept is None else ids[: self.accept]


def setup(tmp_path):
    storage = Storage(tmp_path)
    emp = storage.add_employee("NV01", "An", at(7))
    ids = [storage.add_attendance(emp["id"], "cam", at(8, i), 0.8, 0.9, None) for i in range(3)]
    return storage, ids


def test_run_once_marks_sent(tmp_path):
    storage, _ = setup(tmp_path)
    sink = FakeSink()
    assert SyncWorker(storage, sink, clock=lambda: at(9)).run_once() == 3
    assert storage.pending_events() == [] and sink.batches[0][0]["employee_code"] == "NV01"
    storage.close()


def test_failure_keeps_pending(tmp_path):
    storage, _ = setup(tmp_path)
    assert SyncWorker(storage, FakeSink(fail=True), clock=lambda: at(9)).run_once() == 0
    assert len(storage.pending_events()) == 3
    storage.close()


def test_partial_success(tmp_path):
    storage, ids = setup(tmp_path)
    assert SyncWorker(storage, FakeSink(accept=1), clock=lambda: at(9)).run_once() == 1
    assert [e["id"] for e in storage.pending_events()] == ids[1:]
    storage.close()


def test_nothing_pending_skips_sink(tmp_path):
    storage, _ = setup(tmp_path)
    sink = FakeSink()
    worker = SyncWorker(storage, sink, clock=lambda: at(9))
    worker.run_once()
    assert worker.run_once() == 0 and len(sink.batches) == 1
    storage.close()
