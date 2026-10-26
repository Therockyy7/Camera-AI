"""Đo SFace và ArcFace trên cùng các frame để chọn model + ngưỡng (không ghi gì vào DB).

    python -m cameraai.evaluate [--source 0|rtsp://...|clip.mp4] [--seconds 30] [--employee MÃ] [--record clip.mp4]

Mặt lớn nhất trong frame được coi là người cần đo (nhân viên --employee); các mặt khác là hậu cảnh (người lạ),
dùng để xem điểm cao nhất của người lạ khi chọn ngưỡng. Dừng server trước nếu dùng webcam.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable

import cv2
import numpy as np

from . import model_store
from .camera import open_capture
from .config import load_config, mask_source, parse_source
from .face_engine import FaceEngine, reject_reason, sharpness, turn_offset
from .gallery import normalize
from .logsetup import setup_console
from .recognizers import Recognizer, make_recognizer, threshold_for
from .storage import Storage

REASON_NAMES = {"small": "mặt nhỏ", "edge": "chạm mép khung hình", "blurry": "mờ"}
TURN_EDGES = (0.15, 0.30, 0.45)
SIZE_EDGES = (64, 120, 200)
VERTICAL_EDGES = (0.45, 0.60, 0.75)


def vertical_ratio(row: np.ndarray) -> float | None:
    """(mũi − mắt) / (miệng − mắt) theo trục dọc. Khoảng 0,5 khi nhìn ngang; lớn dần khi camera nhìn từ trên xuống."""
    eye_y = (float(row[5]) + float(row[7])) / 2
    mouth_y = (float(row[11]) + float(row[13])) / 2
    span = mouth_y - eye_y
    return (float(row[9]) - eye_y) / span if span > 0 else None


def _bucket(value: float, edges: tuple, fmt: Callable[[float], str], open_start: bool) -> str:
    if value < edges[0]:
        return f"< {fmt(edges[0])}" if open_start else f"{fmt(0)}–{fmt(edges[0])}"
    for low, high in zip(edges, edges[1:]):
        if value < high:
            return f"{fmt(low)}–{fmt(high)}"
    return f"≥ {fmt(edges[-1])}"


def turn_bucket(turn: float) -> str:
    return _bucket(turn, TURN_EDGES, lambda v: f"{v:.2f}", open_start=False)


def size_bucket(size: float) -> str:
    return _bucket(size, SIZE_EDGES, lambda v: f"{v:.0f}", open_start=True)


def vertical_bucket(ratio: float) -> str:
    return _bucket(ratio, VERTICAL_EDGES, lambda v: f"{v:.2f}", open_start=True)


def summarize(records: list[dict], key: str, bucket: Callable[[float], str], thresholds: dict[str, float]):
    """[(nhóm, số frame, {model: (điểm trung vị, tỷ lệ >= ngưỡng)})] theo thứ tự nhóm."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        if record.get(key) is not None:
            groups.setdefault(bucket(record[key]), []).append(record)
    rows = []
    for name in sorted(groups, key=lambda g: min(r[key] for r in groups[g])):  # theo giá trị, không theo chuỗi
        items = groups[name]
        stats = {}
        for model, threshold in thresholds.items():
            scores = [r["scores"][model] for r in items if model in r["scores"]]
            if scores:
                stats[model] = (
                    round(statistics.median(scores), 3),
                    round(sum(s >= threshold for s in scores) / len(scores), 3),
                )
        rows.append((name, len(items), stats))
    return rows


def choose_target(employees: list[dict], code: str | None) -> dict:
    if code:
        for employee in employees:
            if employee["code"] == code:
                return employee
        raise SystemExit(f"Không có nhân viên mã {code}")
    enrolled = [e for e in employees if e["sample_count"] > 0]
    if len(enrolled) == 1:
        return enrolled[0]
    raise SystemExit("Cần --employee MÃ (người đứng trước camera). Đã đăng ký: "
                     + ", ".join(e["code"] for e in enrolled))


def _target_gallery(storage: Storage, engine: FaceEngine, recognizer: Recognizer, employee_id: int) -> np.ndarray:
    """Embedding các mẫu của nhân viên cho model này: lấy từ DB, thiếu thì tính từ ảnh mẫu (chỉ trong bộ nhớ)."""
    vectors = [normalize(e) for e in storage.employee_embeddings(employee_id, recognizer.name)]
    for sample in storage.samples_missing_embedding(recognizer.name):
        if sample["employee_id"] != employee_id or not sample["image_path"]:
            continue
        path = storage.resolve_file(sample["image_path"])
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if path else None
        rows = engine.detect(image) if image is not None else []
        if len(rows):
            vectors.append(normalize(recognizer.embed(image, max(rows, key=lambda r: r[2]))))
    return np.stack(vectors) if vectors else np.zeros((0, recognizer.dim), np.float32)


def _print_table(title: str, rows, models: list[str]) -> None:
    print(f"\n{title}")
    header = f"  {'nhóm':<12}{'frame':>6}" + "".join(f"{m + ' trung vị':>18}{m + ' ≥ngưỡng':>16}" for m in models)
    print(header)
    for name, count, stats in rows:
        cells = "".join(
            f"{stats[m][0]:>18.3f}{stats[m][1]:>15.0%} " if m in stats else f"{'-':>18}{'-':>16}" for m in models
        )
        print(f"  {name:<12}{count:>6}{cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cameraai.evaluate", description="So sánh SFace và ArcFace")
    parser.add_argument("--source", help="mặc định: camera.source trong config")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--employee", help="mã nhân viên đứng trước camera")
    parser.add_argument("--record", help="lưu frame gốc ra file .mp4 để chạy lại")
    args = parser.parse_args(argv)
    setup_console()

    cfg = load_config()
    source = args.source or cfg.camera.source
    names = ["sface"] + (["arcface"] if not model_store.verify(cfg.models_path, [model_store.ARCFACE]) else [])
    if "arcface" in names:
        print(model_store.ARCFACE_LICENSE_NOTE)
    else:
        print("Chưa có ArcFace (python -m cameraai.model_store download --arcface) — chỉ đo SFace.")

    storage = Storage(cfg.data_path)
    target = choose_target(storage.list_employees(), args.employee)
    engine = FaceEngine(cfg)  # chỉ dùng bộ phát hiện + các hàm chất lượng
    recognizers = {name: make_recognizer(cfg, name) for name in names}
    galleries = {name: _target_gallery(storage, engine, rec, target["id"]) for name, rec in recognizers.items()}
    storage.close()
    thresholds = {name: threshold_for(cfg, name) for name in names}
    print(f"Nhân viên: {target['code']} {target['name']} | mẫu: "
          + ", ".join(f"{n}={len(g)}" for n, g in galleries.items()) + f" | nguồn: {mask_source(source)}")
    if any(len(g) == 0 for g in galleries.values()):
        print("Nhân viên chưa có mẫu dùng được.")
        return 1

    cap = open_capture(parse_source(source), cfg.camera)
    if not cap.isOpened():
        print("Không mở được nguồn video (webcam đang bị server dùng?)")
        return 1
    writer = None
    records, background, skipped, frames = [], {n: [] for n in names}, Counter(), 0
    deadline = time.monotonic() + args.seconds
    print(f"Đang đo {args.seconds:.0f} giây… xoay đầu, ngẩng, cúi, đi lại gần/xa camera.")
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if not ok:
            if parse_source(source).kind == "file":
                break
            continue
        frames += 1
        if args.record:
            if writer is None:
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10, frame.shape[1::-1])
            writer.write(frame)
        rows = sorted(engine.detect(frame), key=lambda r: -r[2])
        for index, row in enumerate(rows):
            box = tuple(int(round(v)) for v in row[:4])
            turn = turn_offset(row)
            # Bỏ mặt nhỏ/chạm mép/mờ như ứng dụng; GIỮ mặt nghiêng để xem model chịu được tới đâu.
            reason = reject_reason(box, 0.0, sharpness(frame, box), frame.shape, cfg.recognition.min_face_px,
                                   cfg.recognition)
            if reason:
                if index == 0:
                    skipped[reason] += 1
                continue
            scores = {n: float((galleries[n] @ normalize(r.embed(frame, row))).max()) for n, r in recognizers.items()}
            if index == 0:
                records.append({"turn": turn, "size": min(box[2], box[3]), "vertical": vertical_ratio(row),
                                "scores": scores})
            else:
                for n, s in scores.items():
                    background[n].append(s)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Đã lưu clip: {args.record}")

    reasons = ", ".join(f"{REASON_NAMES.get(r, r)} {n}" for r, n in skipped.most_common()) or "không"
    print(f"\n{frames} frame; {len(records)} frame có mặt dùng được; mặt chính bị loại: {reasons}")
    print("Ngưỡng: " + ", ".join(f"{n}={t}" for n, t in thresholds.items()))
    if not records:
        return 1
    _print_table("Theo độ quay ngang (0 = nhìn thẳng)", summarize(records, "turn", turn_bucket, thresholds), names)
    _print_table("Theo cỡ mặt (px)", summarize(records, "size", size_bucket, thresholds), names)
    _print_table("Theo tỷ lệ dọc (≈0,5 ngang tầm; lớn hơn = nhìn từ trên xuống)",
                 summarize(records, "vertical", vertical_bucket, thresholds), names)
    _print_table("Tất cả", summarize(records, "turn", lambda _: "tất cả", thresholds), names)
    print("\nMặt hậu cảnh (người khác) — điểm khớp với nhân viên này, càng thấp càng tốt:")
    for name, scores in background.items():
        if scores:
            print(f"  {name}: {len(scores)} mặt, cao nhất {max(scores):.3f}, p95 {np.percentile(scores, 95):.3f}")
        else:
            print(f"  {name}: không có mặt hậu cảnh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
