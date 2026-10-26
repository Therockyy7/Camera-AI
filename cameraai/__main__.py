"""Chạy CameraAI: python -m cameraai [--source ...] [--config ...] [--no-liveness]"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from . import model_store
from .camera import VideoSource
from .config import is_loopback, load_config, make_clock, mask_source
from .enrollment import EnrollmentService
from .face_engine import FaceEngine
from .gallery import Gallery
from .logsetup import setup_console
from .pipeline import Pipeline
from .recognizers import threshold_for
from .storage import Storage
from .web.app import Services, create_app

log = logging.getLogger("cameraai")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cameraai", description="Chấm công bằng camera nhận diện khuôn mặt")
    parser.add_argument("--config", help="đường dẫn file cấu hình (mặc định config.yaml)")
    parser.add_argument("--source", help="nguồn video: 0 = webcam, rtsp://..., hoặc đường dẫn file .mp4")
    parser.add_argument("--no-liveness", action="store_true", help="tắt kiểm tra mặt thật / giả")
    args = parser.parse_args(argv)
    setup_console()

    cfg = load_config(args.config)
    if args.source:
        cfg.camera.source = args.source
    if args.no_liveness:
        cfg.liveness.mode = "off"

    if not is_loopback(cfg.web.host) and not cfg.web.admin_password:
        log.error("Nghe trên %s (máy khác truy cập được) thì bắt buộc đặt ADMIN_PASSWORD", cfg.web.host)
        return 2
    problems = model_store.verify(cfg.models_path, model_store.required_models(cfg))
    if problems:
        for problem in problems:
            log.error(problem)
        log.error("Chạy: %s%s", model_store.DOWNLOAD_HINT, " --arcface" if cfg.recognition.model == "arcface" else "")
        return 1
    if cfg.recognition.model == "arcface":
        log.warning(model_store.ARCFACE_LICENSE_NOTE)

    clock = make_clock(cfg.timezone)
    storage = Storage(cfg.data_path)
    gallery = Gallery(threshold_for(cfg, cfg.recognition.model))
    source = VideoSource(cfg.camera)
    pipeline = Pipeline(cfg, source, FaceEngine(cfg, gallery), storage, gallery, clock)
    enrollment = EnrollmentService(cfg, lambda: FaceEngine(cfg), source, storage, pipeline.reload_gallery, clock)
    enrollment.reindex()  # đổi model → tính embedding mới từ ảnh mẫu trước khi nạp gallery
    app = create_app(Services(cfg, storage, source, pipeline, enrollment, clock))

    log.info(
        "Nguồn video: %s | camera_id: %s | model: %s (ngưỡng %.3f) | chống giả mạo: %s",
        mask_source(cfg.camera.source), cfg.camera.id, cfg.recognition.model, gallery.threshold, cfg.liveness.mode,
    )
    source.start()
    pipeline.start()
    shown_host = "localhost" if is_loopback(cfg.web.host) else "<địa-chỉ-IP-máy-này>"
    log.info("Mở trình duyệt: http://%s:%d  (Ctrl+C để dừng)", shown_host, cfg.web.port)
    try:
        uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        pipeline.stop()
        source.stop()
        storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
