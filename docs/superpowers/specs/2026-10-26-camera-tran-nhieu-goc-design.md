# Camera trần ở cửa / hành lang — nhận diện nhiều góc (CameraAI) — Thiết kế

Ngày: 2026-10-26 · Trạng thái: đã duyệt hướng thiết kế · Bổ sung cho `2026-10-26-cham-cong-camera-design.md`

## 1. Bối cảnh

Bản đầu chỉ nhận tốt mặt nhìn thẳng: bộ lọc góc/cỡ chặt (phục vụ model chống giả mạo), SFace chịu nghiêng kém,
mẫu đăng ký chỉ một góc. Thực tế sẽ lắp **camera trên trần / tường cao ở cửa ra vào hoặc hành lang, người đi về phía
camera**, ở xa hơn webcam.

Quyết định đã chốt với người dùng:

| Câu hỏi | Lựa chọn |
|---|---|
| Tình huống | Cửa / hành lang, người đi về phía camera |
| Model | Cho phép đổi model; thêm ArcFace để đo so với SFace rồi chọn. ArcFace chỉ đánh giá nội bộ — dùng thật phải mua license hoặc giữ SFace |
| Chống giả mạo | Kiểm khi đủ điều kiện (opportunistic); không kiểm được vẫn chấm, ghi "chưa kiểm" + ảnh để hậu kiểm |

Giới hạn vật lý (không phần mềm nào vượt qua): hướng dẫn camera nhận diện của Hikvision yêu cầu góc chúc 15° ± 5°,
lệch ngang < 25°, khoảng cách đồng tử > 40 px, vùng nhận diện ≤ 3 m ngang với camera 2MP; Trassir cho phép tối đa
~30–35°. README phải hướng dẫn chọn vị trí lắp.

Ngoài phạm vi: tự học thêm mẫu khi nhận đúng (rủi ro nạp nhầm người), nhiều camera trong một tiến trình, GPU.

## 2. Nhận diện đổi được model

- `cameraai/recognizers.py`: `Recognizer` (Protocol: `name`, `dim`, `threshold`, `embed(frame, row)`):
  - `SFaceRecognizer` — `cv2.FaceRecognizerSF` (`alignCrop` + `feature`), 128 chiều, mặc định.
  - `ArcFaceRecognizer` — `w600k_r50.onnx` (ResNet50 @ WebFace600K, gói `buffalo_l` của InsightFace), 512 chiều,
    onnxruntime CPU. Căn chỉnh 112×112 bằng phép đồng dạng Umeyama từ 5 điểm mốc YuNet về template ArcFace
    (cùng template OpenCV dùng cho SFace); tiền xử lý RGB, `(x − 127,5) / 127,5`, NCHW.
  - `make_recognizer(cfg, name)`.
- Cấu hình: `recognition.model: sface | arcface`, `recognition.sface_threshold` (0,363),
  `recognition.arcface_threshold` (0,30 — chốt lại bằng công cụ đo).
- `model_store`: ArcFace là model **tuỳ chọn**, không tải mặc định. `python -m cameraai.model_store download --arcface`
  in cảnh báo giấy phép ("ALL models are available for non-commercial research purposes only"), tải
  `buffalo_l.zip` (v0.7, 289 MB), chỉ giữ `w600k_r50.onnx`, kiểm SHA256. Khởi động với `model: arcface` mà thiếu
  model → báo lệnh tải.
- DB (migration `PRAGMA user_version` 0 → 1): bảng `sample_embeddings(sample_id, model, embedding)`, chuyển embedding
  hiện có sang model `sface`, bỏ cột `face_samples.embedding`. Khi khởi động, mẫu chưa có embedding của model đang
  dùng được **tính lại từ ảnh mẫu** (phát hiện lại mặt trên ảnh crop) → đổi model không phải đăng ký lại.

## 3. Thích nghi camera trần

- `liveness.mode: off | opportunistic | required` (mặc định `opportunistic`; thay `liveness.enabled`,
  env `LIVENESS_MODE`, `--no-liveness` = `off`).
  - Chỉ chạy model chống giả khi mặt ≥ `liveness.min_face_px` (64), độ quay ≤ `liveness.max_turn` (0,25).
    Không đủ điều kiện → `liveness = None`, `is_real = None` ("chưa kiểm").
  - Bỏ phiếu chấm công: `required` chỉ tính frame `is_real = True`; `opportunistic` tính frame `is_real ≠ False`;
    `off` tính mọi frame.
  - Ghi giả mạo (không đổi): trung bình điểm của 5 frame **đã kiểm** < `spoof_threshold` (0,3).
  - Lần chấm không có frame nào kiểm được → `liveness_score` rỗng, giao diện ghi "chưa kiểm giả mạo".
- Bộ lọc nhận diện nới hơn bộ lọc chống giả: `recognition.min_face_px` 40, `recognition.max_turn` 0,35,
  `recognition.min_sharpness` 12 (mặt nhỏ phóng lên 112 px cho phương sai Laplacian thấp; đo trên webcam: mặt 47 px
  được 11–19, mặt 250 px được 44–106). Loại mặt chạm mép giữ nguyên.
- `processing.detect_width` cho camera trần nên 960–1280 (README); nhận diện vẫn cắt từ frame gốc.

## 4. Đăng ký nhiều góc

- "Chụp từ camera" lấy tối đa `enrollment.samples_per_capture` (8) mẫu trong `capture_timeout_s` (15 s); chỉ nhận
  mẫu có cosine < `enrollment.max_similarity` (0,92) với **mọi** mẫu đã có của người đó (model đang dùng) — người
  dùng xoay đầu chậm trái/phải/ngẩng/cúi, mẫu tự trải ra nhiều góc. Lý do bỏ mới: `similar` ("Giống mẫu đã có —
  xoay đầu sang góc khác").
- Đếm mặt bỏ qua mặt nhỏ hơn `enrollment.min_face_px` (mặt hậu cảnh); không còn mặt đủ lớn nhưng có mặt nhỏ → `small`.
- Khi triển khai nên đăng ký bằng chính camera trần.

## 5. Công cụ đo

`python -m cameraai.evaluate [--source 0|rtsp://...|clip.mp4] [--seconds 30] [--record clip.mp4]`
(dừng server nếu dùng webcam). Mỗi frame: mặt lớn nhất (người cần đo) + các mặt khác (hậu cảnh); tính cỡ, độ quay,
tỷ lệ dọc `(mũi − mắt) / (miệng − mắt)`, chạm mép, điểm khớp của từng model có sẵn với gallery (embedding tính trong
bộ nhớ từ ảnh mẫu, không ghi DB). In bảng theo nhóm độ quay và nhóm cỡ mặt: số frame, điểm trung vị, % ≥ ngưỡng
của mỗi model; điểm cao nhất của mặt hậu cảnh (để chọn ngưỡng). `--record` lưu clip để chạy lại so sánh công bằng.

## 6. Kiểm thử

Unit: config (mode, model, kiểm giá trị), model_store (giải nén member, SHA), Umeyama, migration DB + embedding theo
model, gating liveness/nhận diện (hàm thuần), decider 3 chế độ, enrollment (đa dạng, bỏ mặt hậu cảnh, tính lại
embedding từ ảnh), bảng tổng hợp của evaluate. Tích hợp ArcFace tự bỏ qua nếu chưa tải. Đo thật trên webcam ở nhiều góc.

## 7. Mặc định AI tự chốt

recognition.min_face_px 40 · recognition.max_turn 0,45 (ban đầu 0,35; nâng sau khi đo — xem mục 8) ·
min_sharpness 12 · liveness.min_face_px 64 ·
liveness.max_turn 0,25 · arcface_threshold 0,30 (tạm, đo lại) · 8 mẫu / 15 s · max_similarity 0,92.

## 8. Kết quả đo trên webcam laptop (2026-10-26)

`python -m cameraai.evaluate --seconds 40 --employee DEV01`, 7 mẫu đăng ký (một góc), người ngồi cách ~1 sải tay
xoay đầu nhiều góc; 248/383 frame dùng được (loại: chạm mép 95, mờ 40). Frame mặt < 64 px là người hậu cảnh
(mặt lớn nhất khi người đo quay ra ngoài), không tính.

| Nhóm (mặt người đo) | SFace ≥ ngưỡng / trung vị | ArcFace ≥ ngưỡng / trung vị |
|---|---|---|
| Mặt ≥ 200 px (168) | 100% / 0,82 | 100% / 0,80 |
| Quay 0,30–0,45 (175) | 78% / 0,82 | 78% / 0,80 |
| Quay ≥ 0,45 (51, có lẫn hậu cảnh) | 80% / 0,59 | 82% / 0,59 |
| Người khác, điểm cao nhất (53 mặt) | 0,236 (ngưỡng 0,363) | 0,148 (ngưỡng 0,30) |

Kết luận: ở góc ngang tầm webcam SFace ≈ ArcFace (ArcFace tách người lạ rộng hơn một chút) → giữ SFace;
nâng `recognition.max_turn` 0,35 → 0,45. Chưa đo được góc nhìn từ trên xuống (tỷ lệ dọc chỉ 0,45–0,60) —
quyết định cuối cùng về model phải đo trên clip quay từ camera trần thật (`--record` / `--source clip.mp4`).
