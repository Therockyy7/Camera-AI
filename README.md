# CameraAI — Chấm công bằng camera

Nhận diện khuôn mặt nhân viên qua camera để ghi lần chấm công. Chạy thử trên laptop bằng webcam; khi triển khai
thật dùng IP camera (RTSP) — **chỉ đổi cấu hình nguồn video, không đổi code**.

- Phát hiện mặt: YuNet (MIT) · Nhận diện: SFace (Apache-2.0) · Chống giả mạo: MiniFASNetV2 (Apache-2.0)
- Chạy CPU, không cần GPU. Dữ liệu lưu cục bộ trong `data/` (SQLite + ảnh).
- Thiết kế: `docs/superpowers/specs/2026-10-26-cham-cong-camera-design.md`

## 1. Chạy trên laptop (Windows, PowerShell)

Yêu cầu: Python 3.12, webcam. Tại thư mục `D:\CameraAI`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m cameraai.model_store download
.\.venv\Scripts\python.exe -m cameraai
```

Mở http://localhost:8000. Dừng bằng `Ctrl+C`. Không cần kích hoạt venv (gọi thẳng `.venv\Scripts\python.exe`).

Tham số hữu ích:

| Lệnh | Tác dụng |
|---|---|
| `python -m cameraai --source 1` | dùng webcam thứ 2 |
| `python -m cameraai --source D:\clip.mp4` | phát lại video đã quay (test lặp lại được) |
| `python -m cameraai --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"` | thử IP camera thật ngay trên laptop |
| `python -m cameraai --no-liveness` | tắt chống giả mạo |

## 2. Dùng thử

1. Tab **Nhân viên** → nhập mã + tên → **Thêm nhân viên** (cần sự đồng ý của nhân viên trước khi đăng ký khuôn mặt).
2. Bấm **Đăng ký khuôn mặt** → **Chụp từ camera** (nhìn thẳng, đủ sáng, một người trong khung) hoặc **Tải ảnh lên**.
   Nên có ≥ 3 mẫu mỗi người.
3. Tab **Trực tiếp**: ngồi trước camera → khung **xanh** có tên = đã chấm. Cùng người không bị ghi lại trong 5 phút.
4. Giơ ảnh khuôn mặt trên điện thoại (nhìn thẳng, đủ gần) khoảng 1 giây → khung **đỏ** "Nghi gia mao", ghi vào tab
   **Bị từ chối**. Tay che mặt hoặc cử động mạnh cũng có thể bị đánh giá là giả.
5. Tab **Lịch sử**: giờ vào (lần đầu) / giờ ra (lần cuối) trong ngày, ảnh từng lần chấm, **Xuất CSV** (mở được bằng Excel).

Ý nghĩa khung trên hình: xanh = đã xác nhận · vàng = đang nhận diện · đỏ = giả mạo / người lạ ·
xám = bỏ qua ("Lai gan hon" mặt nhỏ, "Lui ra / vao giua khung hinh" mặt bị cắt mép, "Nhin thang camera" mặt
nghiêng, "Giu yen" ảnh mờ).

Mẹo khi test bằng laptop: ngồi cách webcam khoảng một sải tay để thấy trọn khuôn mặt, và nhìn vào webcam (nếu dùng
màn hình ngoài, webcam laptop sẽ thấy mặt bạn quay nghiêng). Khi đăng ký nên chụp 2 lượt, hơi xoay đầu nhẹ.

## 3. Test tự động

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Test logic không cần camera; test tích hợp model tự bỏ qua nếu chưa tải model.

## 4. Tinh chỉnh (`config.yaml`)

| Khoá | Mặc định | Khi nào đổi |
|---|---|---|
| `recognition.model` | sface | `arcface` để đánh giá (xem mục 5) |
| `recognition.sface_threshold` / `arcface_threshold` | 0.363 / 0.30 | Nhận nhầm người → tăng; hay không nhận ra → đăng ký thêm mẫu nhiều góc trước, rồi mới giảm nhẹ. Chọn bằng `python -m cameraai.evaluate` |
| `recognition.min_face_px` | 40 | Cạnh mặt tối thiểu để nhận diện |
| `recognition.max_turn` | 0.45 | Hay bị "Nhin thang camera" → tăng (0.5); đo trên webcam: quay 0,30–0,45 vẫn nhận tốt |
| `recognition.min_sharpness` | 12 | Hay bị "Giu yen" → giảm |
| `liveness.mode` | opportunistic | `required`: chỉ chấm khi đã kiểm là mặt thật · `off`: tắt |
| `liveness.min_face_px` / `max_turn` | 64 / 0.25 | Vùng model chống giả còn tin cậy; ngoài vùng này frame là "chưa kiểm" |
| `liveness.threshold` | 0.5 | Frame đạt ngưỡng này được coi là thật. Ảnh giả lọt qua → tăng (0.7–0.8) |
| `liveness.spoof_threshold` | 0.3 | Ghi "giả mạo" khi điểm trung bình 5 frame đã kiểm thấp hơn. Người thật hay bị ghi giả mạo → giảm (0.2) |
| `enrollment.max_similarity` | 0.92 | Mẫu mới phải khác mẫu cũ; báo "Giống mẫu đã có" quá nhiều → tăng (0.95) |
| `processing.detect_width` | 640 | Camera trần, mặt nhỏ → 960–1280 (tốn CPU hơn) |
| `attendance.cooldown_minutes` | 5 | Khoảng tối thiểu giữa 2 lần ghi của cùng người |
| `storage.snapshot_retention_days` | 30 | Số ngày giữ ảnh chụp |

Biến môi trường ghi đè: `CAMERA_SOURCE`, `CAMERA_ID`, `CAMERAAI_HOST`, `CAMERAAI_PORT`, `ADMIN_PASSWORD`,
`CAMERAAI_DATA_DIR`, `LIVENESS_MODE`, `RECOGNITION_MODEL`.

## 5. Camera trần ở cửa / hành lang (nhiều góc)

**Vị trí lắp quyết định nhiều hơn model.** Theo hướng dẫn camera nhận diện của Hikvision: góc chúc xuống
15° ± 5°, lệch ngang < 25°, khoảng cách hai đồng tử trên ảnh > 40 px, camera 2MP phủ ≤ 3 m bề ngang. Quá ~35°
(nhìn từ trên xuống dốc) thì model nào cũng kém. Đặt camera ở chỗ người đi thẳng về phía nó (cửa, hành lang hẹp),
lùi điểm nhìn ra xa cửa để góc chúc nhỏ, đủ sáng, tránh ngược sáng.

- **Đăng ký bằng chính camera đó**: người cần đăng ký đứng ở cửa, bấm "Chụp từ camera" rồi xoay đầu chậm
  (trái, phải, ngẩng, cúi). Chỉ mẫu khác góc mới được lưu → gallery phủ nhiều góc.
- **Chống giả mạo** mặc định `opportunistic`: chỉ kiểm khi mặt ≥ 64 px và gần thẳng; mặt xa/xiên vẫn được chấm,
  ghi "chưa kiểm giả mạo" kèm ảnh để hậu kiểm (tab Lịch sử).
- **Mặt nhỏ**: tăng `processing.detect_width` lên 960–1280; dùng luồng chính (độ phân giải cao) nếu CPU đủ.

## 6. So sánh SFace và ArcFace

SFace (mặc định) có giấy phép dùng thương mại. ArcFace (InsightFace `buffalo_l`) thường chịu góc nghiêng và mặt
nhỏ tốt hơn nhưng **chỉ dùng cho nghiên cứu phi thương mại** — dùng thật cho sản phẩm phải mua license InsightFace.

```powershell
.\.venv\Scripts\python.exe -m cameraai.model_store download --arcface
```

Đo hai model trên cùng frame (dừng server trước nếu dùng webcam; đứng trước camera, xoay đầu, đi lại gần/xa):

```powershell
.\.venv\Scripts\python.exe -m cameraai.evaluate --seconds 40 --employee DEV01
```

Thêm `--record clip.mp4` để lưu đoạn quay, sau đó chạy lại `--source clip.mp4` cho các lần so sánh sau
(nên quay một đoạn từ camera trần thật). Bảng kết quả cho mỗi nhóm góc / cỡ mặt: điểm trung vị và tỷ lệ vượt
ngưỡng của từng model, kèm điểm cao nhất của người khác (hậu cảnh) để chọn ngưỡng an toàn.

Đổi model: sửa `recognition.model: arcface` trong `config.yaml` (hoặc `RECOGNITION_MODEL=arcface`) rồi khởi động
lại — embedding của model mới được tự tính từ ảnh mẫu đã lưu, không phải đăng ký lại.

## 7. Triển khai với IP camera

Camera thường chỉ phát luồng video, nên CameraAI chạy trên một máy (mini PC / máy chủ) **cùng mạng LAN** với camera.

1. Bật RTSP trên camera, lấy URL **luồng phụ** (độ phân giải thấp, nhẹ CPU):
   - Hikvision / Ezviz đời cũ: `rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/102`
   - Dahua / Imou: `rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=1`
   - Model khác: xem tài liệu của hãng. Kiểm tra trước bằng VLC (Media → Open Network Stream) hoặc chạy thử
     `python -m cameraai --source "rtsp://..."` ngay trên laptop.
2. Chép thư mục dự án sang máy triển khai (kèm `data/` để giữ danh sách đã đăng ký).
3. Tạo `.env` từ `.env.example`, sửa `CAMERA_SOURCE`, `CAMERA_ID`, `ADMIN_PASSWORD`.
4. Chạy:
   ```bash
   docker compose up -d --build
   docker compose logs -f
   ```
5. Mở `http://<ip-máy-triển-khai>:8000` (đăng nhập: tên tuỳ ý + `ADMIN_PASSWORD`).

Không dùng Docker cũng được: cài Python 3.12 rồi làm như mục 1, đặt `CAMERA_SOURCE` / `CAMERAAI_HOST=0.0.0.0` /
`ADMIN_PASSWORD` trong biến môi trường. Mỗi tiến trình phụ trách **một** camera; nhiều camera thì chạy nhiều
container với `CAMERA_ID` và cổng khác nhau.

Sau khi lắp thật nên chỉnh lại `min_face_px`, `max_turn`, `min_sharpness` và `liveness.threshold` theo góc đặt
camera (đặt ngang tầm mặt, đủ sáng, tránh ngược sáng cho kết quả tốt nhất).

Lưu ý: Docker Desktop trên Windows không truy cập được webcam laptop — trên laptop hãy chạy trực tiếp bằng Python.

## 8. Dữ liệu và bảo mật

- Ảnh / đặc trưng khuôn mặt là dữ liệu cá nhân nhạy cảm (Nghị định 13/2023/NĐ-CP): cần nhân viên đồng ý trước
  khi đăng ký. Dữ liệu chỉ nằm trong `data/`.
- Người lạ và lần giả mạo cũng được chụp ảnh lưu lại (tab Bị từ chối); ảnh tự xoá sau `snapshot_retention_days`.
- Xoá nhân viên: giữ lịch sử chấm công, **xoá vĩnh viễn** mẫu khuôn mặt.
- Mặc định web chỉ nghe `127.0.0.1`. Mở ra LAN (`CAMERAAI_HOST=0.0.0.0`) thì bắt buộc `ADMIN_PASSWORD`.

## 9. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| "Mất kết nối camera" với webcam | Đóng ứng dụng khác đang dùng webcam (Teams, Zoom, Camera) |
| Thiếu model / sai SHA256 | `python -m cameraai.model_store download` |
| RTSP giật, trễ | Dùng luồng phụ; kiểm tra mạng; CameraAI tự kết nối lại khi mất tín hiệu |
| Không nhận ra người đã đăng ký | Xem nhãn xám trên khung mặt (mặt bị cắt mép / nghiêng / mờ); đăng ký thêm mẫu tại đúng camera và tư thế thường dùng, đủ sáng |

## 10. Nối Fastdo sau này

Mỗi lần chấm có cột `synced_at` (chưa gửi = NULL). `cameraai/sync.py` có sẵn `AttendanceSink` + `SyncWorker` gửi bù
các lần chấm chưa đồng bộ; chỉ cần viết một lớp `send()` gọi API Timekeeping của Fastdo và khởi động `SyncWorker`
trong `cameraai/__main__.py`, không sửa phần nhận diện.
