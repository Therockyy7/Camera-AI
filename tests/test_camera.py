import time

import numpy as np

from cameraai.camera import VideoSource
from cameraai.config import CameraConfig


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, prop):
        return 1000.0  # fps rất cao để test chạy nhanh

    def release(self):
        self.released = True


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_file_source_plays_all_frames_then_ends():
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    source = VideoSource(CameraConfig(source="clip.mp4"), opener=lambda spec, cfg: FakeCapture(frames))
    source.start()
    assert wait_until(lambda: source.status().ended)
    frame_id, frame = source.latest()
    assert frame_id == 3 and frame[0, 0, 0] == 2
    assert source.status().connected is False
    source.stop()


def test_stream_reconnects_and_masks_password():
    opens = []

    def opener(spec, cfg):
        opens.append(spec.value)
        return FakeCapture([np.zeros((4, 4, 3), np.uint8)])

    source = VideoSource(
        CameraConfig(source="rtsp://admin:secret@cam/x"), opener=opener, initial_backoff_s=0.01, max_backoff_s=0.02
    )
    source.start()
    assert wait_until(lambda: len(opens) >= 2)
    status = source.status()
    source.stop()
    assert "secret" not in status.source and status.width == 4
    assert source.status().ended is False


def test_unopenable_source_reports_error():
    source = VideoSource(
        CameraConfig(source="rtsp://cam/x"), opener=lambda spec, cfg: FakeCapture([], opened=False),
        initial_backoff_s=0.01, max_backoff_s=0.02,
    )
    source.start()
    assert wait_until(lambda: source.status().last_error is not None)
    source.stop()
    assert source.status().connected is False
