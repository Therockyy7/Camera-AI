"""Danh sách model ONNX (ghim theo commit/phiên bản + SHA256): tải về, kiểm tra, lấy đường dẫn.

Chạy: python -m cameraai.model_store download [--arcface]   (hoặc: verify)
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str  # SHA256 của file .onnx cuối cùng (đã giải nén nếu tải từ zip)
    archive_member: str | None = None  # tên file cần lấy trong zip


_ZOO = "https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models"
_FAS = "https://github.com/facenox/face-antispoof-onnx/raw/2f8dd93dd0fd9d120f08e83d9020719f67ffdb68/models/best/98.20"

# YuNet — giấy phép MIT
DETECTOR = ModelFile(
    "face_detection_yunet_2023mar.onnx",
    f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
# SFace — giấy phép Apache-2.0
RECOGNIZER = ModelFile(
    "face_recognition_sface_2021dec.onnx",
    f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)
# MiniFASNetV2 128x128 — giấy phép Apache-2.0
LIVENESS = ModelFile(
    "liveness_minifasnetv2_128.onnx",
    f"{_FAS}/best_model.onnx",
    "af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6",
)
ALL_MODELS: tuple[ModelFile, ...] = (DETECTOR, RECOGNIZER, LIVENESS)

# ArcFace ResNet50 @ WebFace600K trong gói buffalo_l của InsightFace — CHỈ phi thương mại (đánh giá nội bộ).
ARCFACE = ModelFile(
    "w600k_r50.onnx",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    archive_member="w600k_r50.onnx",
)
OPTIONAL_MODELS: tuple[ModelFile, ...] = (ARCFACE,)
ARCFACE_LICENSE_NOTE = (
    "LƯU Ý GIẤY PHÉP: model ArcFace của InsightFace chỉ dùng cho nghiên cứu phi thương mại "
    "(\"ALL models are available for non-commercial research purposes only\"). Dùng để ĐÁNH GIÁ nội bộ; "
    "dùng thật cho sản phẩm thương mại phải mua license InsightFace hoặc giữ SFace."
)

DOWNLOAD_HINT = "python -m cameraai.model_store download"


def required_models(cfg) -> tuple[ModelFile, ...]:
    return ALL_MODELS + ((ARCFACE,) if cfg.recognition.model == "arcface" else ())


class ModelMissingError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(models_dir: Path, model: ModelFile) -> Path:
    path = Path(models_dir) / model.name
    if not path.is_file():
        raise ModelMissingError(f"Thiếu model {model.name} trong {models_dir}. Chạy: {DOWNLOAD_HINT}")
    return path


def verify(models_dir: Path, models: Iterable[ModelFile] = ALL_MODELS) -> list[str]:
    problems = []
    for model in models:
        path = Path(models_dir) / model.name
        if not path.is_file():
            problems.append(f"Thiếu {model.name}")
        elif sha256_of(path) != model.sha256:
            problems.append(f"Sai SHA256: {model.name}")
    return problems


def download(models_dir: Path, models: Iterable[ModelFile] = ALL_MODELS) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        target = models_dir / model.name
        if target.is_file() and sha256_of(target) == model.sha256:
            print(f"[có sẵn] {model.name}")
            continue
        part = target.with_name(target.name + ".part")
        print(f"[đang tải] {model.name}")
        if model.archive_member:
            archive = target.with_name(target.name + ".zip.part")
            try:
                urllib.request.urlretrieve(model.url, archive)
                with zipfile.ZipFile(archive) as zf, zf.open(model.archive_member) as src, part.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            finally:
                archive.unlink(missing_ok=True)
        else:
            urllib.request.urlretrieve(model.url, part)
        actual = sha256_of(part)
        if actual != model.sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 của {model.name} không khớp (nhận {actual})")
        part.replace(target)
        print(f"[xong] {model.name}")


def main(argv: list[str] | None = None) -> int:
    from .config import load_config
    from .logsetup import setup_console

    setup_console()
    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else "verify"
    with_arcface = "--arcface" in args
    if command not in ("download", "verify"):
        print("Cách dùng: python -m cameraai.model_store [download|verify] [--arcface]")
        return 2
    cfg = load_config()
    models = required_models(cfg) + ((ARCFACE,) if with_arcface and ARCFACE not in required_models(cfg) else ())
    if ARCFACE in models:
        print(ARCFACE_LICENSE_NOTE)
    if command == "download":
        download(cfg.models_path, models)
    problems = verify(cfg.models_path, models)
    for problem in problems:
        print(problem)
    if problems:
        print(f"Chạy: {DOWNLOAD_HINT}" + (" --arcface" if ARCFACE in models else ""))
        return 1
    print(f"Đủ {len(models)} model trong {cfg.models_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
