# Chấm công bằng camera (CameraAI) — Thiết kế

Ngày: 2026-10-26 · Trạng thái: đã duyệt hướng thiết kế (brainstorming)

> Bổ sung sau: camera trần nhiều góc — `2026-10-26-camera-tran-nhieu-goc-design.md` (thay `liveness.enabled` bằng
> `liveness.mode`, thêm ArcFace tuỳ chọn, embedding theo model, cổng lọc 2 mức, đăng ký nhiều góc, công cụ đo).

## 1. Bối cảnh và mục tiêu

Dùng camera nhận diện khuôn mặt nhân viên để ghi lần chấm công. Giai đoạn này chạy test trên laptop
(webcam tích hợp, Windows 11, Python 3.12). Khi triển khai thật dùng **IP camera thường phát luồng RTSP**
(Hikvision/Dahua/Imou...), code chạy trên mini PC/máy chủ cùng mạng LAN. Chuyển từ laptop sang camera thật
chỉ đổi cấu hình nguồn video (`CAMERA_SOURCE`), không đổi code.

Quyết định đã chốt với người dùng:

| Câu hỏi | Lựa chọn |
|---|---|
| Thiết bị triển khai | IP camera RTSP, code chạy trên máy cùng mạng |
| Dữ liệu chấm công | Lưu cục bộ (SQLite) + trang xem lịch sử; chừa chỗ nối Fastdo sau |
| Đăng ký khuôn mặt | Cả chụp từ camera và tải ảnh lên |
| Chống giả mạo | Có, bật/tắt bằng cấu hình |
| Công nghệ | Python + OpenCV (YuNet + SFace) + MiniFASNet ONNX, chạy CPU |

Ngoài phạm vi bản này: tính ca/công (Fastdo Timekeeping làm), nhiều camera trong một tiến trình,
đồng bộ thật lên Fastdo (chỉ chừa giao diện), GPU.

## 2. Model và giấy phép

| Vai trò | Model | Giấy phép | Nguồn (ghim commit) | SHA256 |
|---|---|---|---|---|
| Phát hiện mặt | YuNet `face_detection_yunet_2023mar.onnx` | MIT | `opencv/opencv_zoo@48b10fa505aeb51fbd127fc60a5c660040e823b7` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Nhận diện | SFace `face_recognition_sface_2021dec.onnx` (LFW 99,40%) | Apache-2.0 | `opencv/opencv_zoo@48b10fa5...` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| Chống giả mạo | MiniFASNetV2 128×128 `models/best/98.20/best_model.onnx` | Apache-2.0 | `facenox/face-antispoof-onnx@2f8dd93dd0fd9d120f08e83d9020719f67ffdb68` | `af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6` |

Không dùng InsightFace `buffalo_l` vì bộ model pretrained chỉ cho phép nghiên cứu phi thương mại.

Tiền xử lý chống giả mạo (theo mã gốc của repo model): crop vuông quanh khung mặt × 1,5, thiếu biên thì
pad `BORDER_REFLECT_101`; ảnh **RGB**; letterbox về 128×128 (`INTER_LANCZOS4` khi phóng, `INTER_AREA` khi thu);
chia 255; CHW float32. Đầu ra 2 logit `[real, spoof]`; là mặt thật khi `real − spoof ≥ ln(p/(1−p))`
với `p` = ngưỡng xác suất (mặc định 0,5). Repo khuyến nghị mặt nguồn ≥ 64 px, nhìn thẳng ±30°, đủ sáng.

## 3. Kiến trúc

```
D:\CameraAI\
  cameraai\
    __main__.py     chạy: python -m cameraai [--source ...] [--config ...]
    config.py       đọc config.yaml, biến môi trường ghi đè, parse nguồn video
    model_store.py  danh sách model + tải/kiểm SHA256 (python -m cameraai.model_store download)
    camera.py       VideoSource: webcam/RTSP/file; thread đọc giữ frame mới nhất; tự kết nối lại
    face_engine.py  FaceEngine: YuNet + SFace + liveness → danh sách FaceObservation cho 1 frame
    liveness.py     LivenessChecker (MiniFASNet ONNX)
    gallery.py      Gallery: embedding đã đăng ký, so khớp cosine
    attendance.py   FaceTracker + AttendanceDecider: bỏ phiếu nhiều frame, cooldown, quyết định ghi
    storage.py      Storage (SQLite + file ảnh)
    sync.py         AttendanceSink (Protocol) + SyncWorker (gửi bù sự kiện chưa đồng bộ)
    pipeline.py     Pipeline: thread xử lý camera → engine → decider → storage; frame có chú thích cho web
    enrollment.py   EnrollmentService: đăng ký mẫu từ camera / ảnh tải lên
    web\app.py      FastAPI: API JSON + MJPEG + file tĩnh; HTTP Basic khi có ADMIN_PASSWORD
    web\static\     index.html, app.js, app.css (HTML/JS thuần, không build)
  models\  data\  tests\  config.yaml  requirements.txt  Dockerfile  docker-compose.yml  .env.example  README.md
```

Ranh giới:
- `FaceEngine` không biết DB; `AttendanceDecider` là logic thuần (không OpenCV), test được bằng dữ liệu giả.
- Đối tượng OpenCV DNN không chia sẻ giữa thread: pipeline có 1 `FaceEngine`, enrollment có 1 `FaceEngine` riêng
  (khoá bằng lock). Enrollment lấy frame mới nhất từ `VideoSource` (getter thread-safe) — bắt buộc vì trên Windows
  webcam bị khoá cho 1 tiến trình, trình duyệt không mở song song được.
- Đổi model nhận diện sau này chỉ thay phần trong `face_engine.py`; khi đổi phải đăng ký lại mẫu (embedding khác chiều).

## 4. Luồng xử lý

1. `VideoSource` đọc liên tục, giữ frame mới nhất. RTSP dùng TCP (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp`).
2. `Pipeline` lấy frame theo `processing.fps` (mặc định 8). Nếu frame rộng hơn `processing.detect_width` (640)
   thì thu nhỏ để phát hiện, nhân toạ độ trả lại khung gốc.
3. `FaceEngine`: YuNet → lọc mặt nhỏ hơn `recognition.min_face_px` (64), chạm/tràn mép khung hình (< 5 px, như
   `detect()` của repo model chống giả mạo — mặt bị cắt làm lệch điểm mốc căn chỉnh và crop chống giả), quay nghiêng quá
   `recognition.max_turn` (0,25 — đo vị trí mũi giữa hai mắt từ điểm mốc YuNet), hoặc mờ (phương sai Laplacian
   < `recognition.min_sharpness`) → liveness (nếu bật) → SFace `alignCrop` + `feature` → `Gallery.match`.
   *Bộ lọc góc mặt thêm sau lần chạy thật đầu tiên trên webcam: mặt quay nghiêng bị chống giả mạo báo nhầm "giả".*
4. `Gallery.match`: L2-normalize, điểm của mỗi nhân viên = max cosine trên các mẫu; nhận nếu
   điểm cao nhất ≥ `recognition.threshold` (0,363 — ngưỡng cosine khuyến nghị của SFace).
5. `FaceTracker` gán quan sát vào track theo IoU ≥ 0,3; track hết hạn sau 1,5 s không thấy.
   Mỗi track giữ 5 quan sát gần nhất.
6. `AttendanceDecider` cho từng track:
   - **Chấm công**: ≥ `attendance.confirm_frames` (3) trong 5 quan sát gần nhất cùng một nhân viên, và (nếu liveness bật)
     các quan sát đó đều là mặt thật → ghi 1 lần cho track, trừ khi nhân viên đó còn trong cooldown
     (`attendance.cooldown_minutes`, mặc định 5, tính theo camera).
   - **Mặt giả**: đủ 5 quan sát và **điểm liveness trung bình** < `liveness.spoof_threshold` (0,3) → ghi
     `rejected_events` lý do `spoof`, 1 lần cho track. *Đổi từ luật "≥3/5 frame < 0,5" sau khi đo trên webcam thật:
     mặt thật có 16% frame < 0,5 và 10% frame < 0,25 (trung vị 0,988), luật cũ sẽ báo nhầm khi đứng lâu.*
   - **Người lạ**: đủ 5 quan sát, tất cả mặt thật nhưng không khớp ai → ghi `rejected_events` lý do `unknown`, 1 lần cho track.
7. Ghi sự kiện + ảnh mặt (crop ×2 quanh khung) vào `data/snapshots/YYYY-MM-DD/`. Frame có vẽ khung/tên/trạng thái
   được encode JPEG cho MJPEG. Nhãn trên hình theo **cửa sổ hiện tại** (đỏ khi trung bình liveness hiện tại dưới
   `spoof_threshold`, xanh khi track đã xác nhận nhân viên), không theo cờ đã ghi — một lần bị nghi giả lúc mới vào
   không che trạng thái đã chấm công về sau.

## 5. Dữ liệu

SQLite `data/cameraai.db`, giờ lưu ISO 8601 kèm offset theo `timezone` (mặc định `Asia/Ho_Chi_Minh`).

- `employees(id, code UNIQUE, name, active, created_at)` — xoá nhân viên = `active=0` (giữ lịch sử chấm công)
  và **xoá cứng** toàn bộ mẫu khuôn mặt + file ảnh mẫu (dữ liệu sinh trắc).
- `face_samples(id, employee_id, embedding BLOB float32[128], image_path, source 'camera'|'upload', created_at)`.
- `attendance_events(id, employee_id, camera_id, ts, score, liveness_score, snapshot_path, synced_at NULL)`.
- `rejected_events(id, camera_id, ts, reason 'unknown'|'spoof', best_employee_id NULL, score, liveness_score, snapshot_path)`.

Chỉ ghi **lần chấm thô** (như máy chấm công). Trang lịch sử tóm tắt theo ngày: giờ vào = lần đầu, giờ ra = lần cuối.
`synced_at IS NULL` = chưa gửi lên hệ thống ngoài; `SyncWorker` chỉ chạy khi có sink được cấu hình
(bản này chưa có sink thật → không chạy).

Lưu giữ: hằng ngày xoá file ảnh chụp và `rejected_events` cũ hơn `storage.snapshot_retention_days` (30);
`attendance_events` giữ nguyên, chỉ bỏ `snapshot_path`.

## 6. Đăng ký khuôn mặt

- Tạo nhân viên: mã (duy nhất) + tên.
- **Chụp từ camera**: trong tối đa 10 s, lấy frame mới nhất mỗi ~0,4 s; frame hợp lệ khi có **đúng 1 mặt**,
  điểm phát hiện ≥ 0,9, cạnh mặt ≥ `enrollment.min_face_px` (80), đủ nét, và (nếu liveness bật) là mặt thật.
  Thu đủ `enrollment.samples_per_capture` (5) mẫu hoặc hết giờ → trả số mẫu đã lưu + lý do bỏ frame.
- **Tải ảnh**: mỗi ảnh phải có đúng 1 mặt đạt cỡ/độ nét (không kiểm liveness với ảnh tải lên).
- Khuyến nghị ≥ 3 mẫu/người. Sau mỗi thay đổi mẫu, gallery của pipeline được nạp lại.

## 7. Web

FastAPI phục vụ:
- `GET /` trang một file (tab: Trực tiếp · Nhân viên · Lịch sử · Bị từ chối).
- `GET /stream.mjpg` MJPEG có chú thích. `GET /api/status` trạng thái camera (nguồn đã che mật khẩu), fps, liveness.
- `GET/POST /api/employees`, `DELETE /api/employees/{id}`, `GET /api/employees/{id}/samples`,
  `POST /api/employees/{id}/capture`, `POST /api/employees/{id}/upload`, `DELETE /api/samples/{id}`.
- `GET /api/attendance?date=`, `GET /api/attendance.csv?from=&to=`, `GET /api/rejected?date=`.
- `GET /files/...` ảnh chụp và ảnh mẫu (chỉ trong `data/`).

Bảo mật: mặc định nghe `127.0.0.1`. Nếu nghe địa chỉ khác (Docker/LAN) thì **bắt buộc** đặt `ADMIN_PASSWORD`
(HTTP Basic cho mọi route), thiếu thì từ chối khởi động.

## 8. Cấu hình

`config.yaml` (mặc định hợp lý cho laptop) + biến môi trường ghi đè: `CAMERA_SOURCE`, `CAMERA_ID`,
`CAMERAAI_HOST`, `CAMERAAI_PORT`, `ADMIN_PASSWORD`, `CAMERAAI_DATA_DIR`, `LIVENESS_ENABLED`.
`CAMERA_SOURCE`: chuỗi số → webcam index (Windows dùng backend DirectShow để mở nhanh);
`rtsp://`/`http://` → luồng mạng; còn lại → đường dẫn file video (phát theo fps của file, hết file thì dừng).

## 9. Lỗi

- Mất camera / đọc lỗi liên tiếp → giải phóng và mở lại, chờ 1 s tăng gấp đôi tới 30 s; `/api/status` báo
  `disconnected` + lỗi gần nhất; pipeline không chết.
- Thiếu model / sai SHA256 → khởi động báo rõ lệnh `python -m cameraai.model_store download`.
- Lỗi xử lý 1 frame → log, bỏ frame, chạy tiếp.

## 10. Kiểm thử

- `pytest` logic thuần, không cần camera/model: config + parse nguồn, gallery, tracker/decider (xác nhận 3/5,
  liveness, cooldown, unknown/spoof 1 lần/track), storage (CRUD, tóm tắt ngày, pending/synced, dọn dẹp),
  SyncWorker với sink giả, API qua `TestClient` với engine giả.
- Test tích hợp model (tự bỏ qua nếu chưa tải model): nạp 3 model, chạy trên ảnh tổng hợp, kiểm hình dạng đầu ra.
- `--source clip.mp4` để chạy lại đoạn video đã quay, kết quả lặp lại được.
- Test tay trên laptop: đăng ký bản thân, ngồi trước webcam → có lần chấm; giơ ảnh trên điện thoại → bị từ chối `spoof`.

## 11. Triển khai IP camera

Mini PC/máy chủ cùng LAN: `docker compose up -d`, sửa `.env`:
`CAMERA_SOURCE=rtsp://user:pass@<ip>:554/...` (nên dùng luồng phụ độ phân giải thấp), `CAMERA_ID`, `ADMIN_PASSWORD`.
Chép thư mục `data/` từ laptop để giữ danh sách đã đăng ký. Mỗi tiến trình/container phụ trách 1 camera.
Docker Desktop trên Windows không truy cập được webcam laptop → trên laptop chạy trực tiếp bằng Python.
Sau khi lắp thật cần chỉnh lại `min_face_px`, `min_sharpness`, ngưỡng liveness theo góc đặt camera.

## 12. Lưu ý pháp lý

Ảnh/đặc trưng khuôn mặt là dữ liệu cá nhân nhạy cảm (Nghị định 13/2023/NĐ-CP): cần sự đồng ý của nhân viên
trước khi đăng ký; dữ liệu chỉ lưu cục bộ; xoá nhân viên thì xoá cứng dữ liệu sinh trắc.

## 13. Mặc định AI tự chốt (người dùng có thể đổi trong config)

process fps 8 · detect_width 640 · ngưỡng cosine 0,363 · min_face_px 64 (chấm công) / 80 (đăng ký) ·
max_turn 0,25 · min_sharpness 25 (phương sai Laplacian trên ảnh mặt xám) ·
xác nhận 3/5 frame · cooldown 5 phút · ngưỡng liveness 0,5 (frame được tính chấm công) · spoof_threshold 0,3
(trung bình cửa sổ) · giữ ảnh 30 ngày · 5 mẫu/lần chụp.
