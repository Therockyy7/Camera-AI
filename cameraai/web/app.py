"""Web: API JSON, luồng MJPEG, trang quản trị tĩnh."""
from __future__ import annotations

import asyncio
import base64
import csv
import io
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import AppConfig
from ..enrollment import REASON_TEXT, EnrollmentError, EnrollmentService
from ..storage import DuplicateEmployeeError, Storage

STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class Services:
    cfg: AppConfig
    storage: Storage
    source: Any  # VideoSource: status()
    pipeline: Any  # Pipeline: latest_jpeg(), recent_events(), processed_fps, reload_gallery()
    enrollment: EnrollmentService
    clock: Callable[[], datetime]


class EmployeeIn(BaseModel):
    code: str
    name: str


class BasicAuthMiddleware:
    """HTTP Basic cho mọi route khi có ADMIN_PASSWORD (tên đăng nhập tuỳ ý)."""

    def __init__(self, app, password: str):
        self.app = app
        self._password = password.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        response = PlainTextResponse(
            "Cần đăng nhập", status_code=401, headers={"WWW-Authenticate": 'Basic realm="CameraAI"'}
        )
        await response(scope, receive, send)

    def _authorized(self, scope) -> bool:
        header = dict(scope.get("headers") or []).get(b"authorization", b"")
        if not header.lower().startswith(b"basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:], validate=True)
        except ValueError:
            return False
        _, _, password = decoded.partition(b":")
        return secrets.compare_digest(password, self._password)


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="CameraAI", docs_url=None, redoc_url=None, openapi_url=None)
    if services.cfg.web.admin_password:
        app.add_middleware(BasicAuthMiddleware, password=services.cfg.web.admin_password)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    storage, enrollment, pipeline = services.storage, services.enrollment, services.pipeline

    def parse_day(value: str | None) -> date:
        if not value:
            return services.clock().date()
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise HTTPException(400, "Ngày không hợp lệ (định dạng YYYY-MM-DD)") from None

    def require_employee(employee_id: int) -> dict:
        employee = storage.get_employee(employee_id)
        if employee is None:
            raise HTTPException(404, "Không tìm thấy nhân viên")
        return employee

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status():
        return {
            "camera": asdict(services.source.status()),
            "camera_id": services.cfg.camera.id,
            "processing_fps": pipeline.processed_fps,
            "recognition_model": services.cfg.recognition.model,
            "liveness_mode": services.cfg.liveness.mode,
            "employees": len(storage.list_employees()),
            "recent": pipeline.recent_events(),
            "now": services.clock().isoformat(),
        }

    @app.get("/stream.mjpg")
    async def stream(request: Request):
        async def frames():
            last_id = -1
            while not await request.is_disconnected():
                frame_id, jpeg = pipeline.latest_jpeg()
                if jpeg is None or frame_id == last_id:
                    await asyncio.sleep(0.03)
                    continue
                last_id = frame_id
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n" + jpeg + b"\r\n"
                )

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"}
        )

    # ---------- nhân viên + mẫu ----------
    @app.get("/api/employees")
    def list_employees():
        return storage.list_employees()

    @app.post("/api/employees", status_code=201)
    def create_employee(body: EmployeeIn):
        try:
            return storage.add_employee(body.code, body.name, services.clock())
        except DuplicateEmployeeError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.delete("/api/employees/{employee_id}")
    def delete_employee(employee_id: int):
        if not storage.deactivate_employee(employee_id):
            raise HTTPException(404, "Không tìm thấy nhân viên")
        pipeline.reload_gallery()
        return {"ok": True}

    @app.get("/api/employees/{employee_id}/samples")
    def list_samples(employee_id: int):
        require_employee(employee_id)
        return storage.list_samples(employee_id)

    @app.post("/api/employees/{employee_id}/capture")
    def capture(employee_id: int):
        require_employee(employee_id)
        try:
            return enrollment.capture(employee_id).to_dict()
        except EnrollmentError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/employees/{employee_id}/upload")
    def upload(employee_id: int, files: list[UploadFile] = File(...)):
        require_employee(employee_id)
        results = []
        for item in files:
            reason = enrollment.add_upload(employee_id, item.file.read())
            results.append({
                "filename": item.filename,
                "ok": reason is None,
                "message": "Đã lưu mẫu" if reason is None else REASON_TEXT.get(reason, reason),
            })
        return results

    @app.delete("/api/samples/{sample_id}")
    def delete_sample(sample_id: int):
        if not storage.delete_sample(sample_id):
            raise HTTPException(404, "Không tìm thấy mẫu")
        pipeline.reload_gallery()
        return {"ok": True}

    # ---------- lịch sử ----------
    @app.get("/api/attendance")
    def attendance(day: str | None = Query(None, alias="date")):
        d = parse_day(day)
        return {"date": d.isoformat(), "summary": storage.daily_summary(d), "events": storage.attendance_on(d)}

    @app.get("/api/attendance.csv")
    def attendance_csv(start: str | None = None, end: str | None = None):
        start_day = parse_day(start)
        end_day = parse_day(end) if end else start_day
        if end_day < start_day:
            raise HTTPException(400, "Ngày kết thúc phải sau ngày bắt đầu")
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Mã NV", "Tên", "Thời điểm", "Camera", "Điểm khớp", "Điểm mặt thật"])
        for e in storage.attendance_between(start_day, end_day):
            liveness = "" if e["liveness_score"] is None else f"{e['liveness_score']:.3f}"
            writer.writerow([e["code"], e["name"], e["ts"], e["camera_id"], f"{e['score']:.3f}", liveness])
        filename = f"cham-cong_{start_day}_{end_day}.csv"
        return Response(
            "﻿" + buffer.getvalue(),  # BOM để Excel đọc đúng tiếng Việt
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/rejected")
    def rejected(day: str | None = Query(None, alias="date")):
        d = parse_day(day)
        return {"date": d.isoformat(), "events": storage.rejected_on(d)}

    @app.get("/files/{rel_path:path}")
    def files(rel_path: str):
        path = storage.resolve_file(rel_path)
        if path is None:
            raise HTTPException(404, "Không tìm thấy file")
        return FileResponse(path)

    return app
