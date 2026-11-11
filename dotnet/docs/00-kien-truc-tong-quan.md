# Bài 00 — Bức tranh tổng thể: CameraAI theo Clean Architecture

> Mục tiêu bài này: trước khi đọc code, biết **mỗi file Python sẽ chuyển sang tầng nào**, **vì sao**, và **mỗi
> pattern trong file ôn tập của bạn sẽ xuất hiện ở đâu** trong dự án.

---

## 1. Luồng chạy của ứng dụng (không cần hiểu thuật toán)

```text
 Camera (webcam / RTSP / file .mp4)
    │  thread nền đọc liên tục, chỉ giữ frame MỚI NHẤT
    ▼
 Frame ──► [Phát hiện mặt]  YuNet  → các khung mặt + 5 điểm mốc (mắt, mũi, miệng)
              │
              ▼
          [Cổng chất lượng]  mặt đủ to? không chạm mép? không nghiêng? không mờ?
              │ qua
              ▼
          [Chống giả]  MiniFASNet → xác suất "mặt thật" (chỉ khi mặt to + gần thẳng)
              │
              ▼
          [Nhận diện]  SFace → embedding (vector 128 số) → so cosine với gallery → "giống NV số 7, điểm 0.72"
              │
              ▼  = FaceObservation (kết quả 1 mặt / 1 frame)
          [Tracker]   ghép mặt qua các frame bằng IoU → "vệt" (Track)
              │
              ▼
          [Decider]   nhìn 5 frame gần nhất của vệt: ≥3 phiếu cùng người → CHẤM CÔNG
              │                                       điểm thật TB thấp  → GIẢ MẠO
              │                                       5 frame không khớp ai → NGƯỜI LẠ
              ▼
          [Lưu]  SQLite + ảnh chụp  ──►  [Web]  MJPEG trực tiếp, quản lý nhân viên, lịch sử, CSV
```

Bạn chỉ cần nhớ: **AI (YuNet/MiniFASNet/SFace) chỉ biến ảnh thành con số**. Mọi quyết định nghiệp vụ (bao nhiêu
phiếu, cooldown 5 phút, khi nào gọi là giả mạo) là code C# thuần — và đó chính là phần quan trọng nhất về kiến trúc.

---

## 2. Bốn tầng và quy tắc phụ thuộc

```text
        ┌──────────────────────────────────────────────┐
        │  CameraAI.Web            (ASP.NET Core)       │  Program.cs = Composition Root (nơi đăng ký DI)
        │   ┌──────────────────────────────────────┐   │
        │   │ CameraAI.Infrastructure              │   │  OpenCvSharp, ONNX Runtime, EF Core + SQLite, file ảnh
        │   │   ┌──────────────────────────────┐   │   │
        │   │   │ CameraAI.Application          │   │   │  Use case + INTERFACE (IVideoSource, IFaceAnalyzer,
        │   │   │   ┌──────────────────────┐   │   │   │   IEmployeeRepository, IUnitOfWork...)
        │   │   │   │ CameraAI.Domain      │   │   │   │  Entity, Value Object, Tracker, Decider, Gallery
        │   │   │   └──────────────────────┘   │   │   │  KHÔNG tham chiếu gói NuGet nào
        │   │   └──────────────────────────────┘   │   │
        │   └──────────────────────────────────────┘   │
        └──────────────────────────────────────────────┘
                 Mũi tên phụ thuộc luôn chỉ VÀO TRONG
```

Kiểm tra trong `.csproj` (đã nối sẵn):

| Project | Tham chiếu | Ý nghĩa |
|---|---|---|
| `CameraAI.Domain` | *(không gì cả)* | Lõi nghiệp vụ, sống được 10 năm dù đổi DB / đổi model AI / đổi web framework |
| `CameraAI.Application` | Domain | Điều phối use case, **khai báo** interface cho thứ nó cần từ bên ngoài |
| `CameraAI.Infrastructure` | Application | **Cài đặt** các interface đó bằng công nghệ cụ thể |
| `CameraAI.Web` | Application + Infrastructure | Nối mọi thứ lại bằng DI, mở HTTP |

**Đây chính là DIP trong file ôn tập của bạn:** Application (module cấp cao) cần "lấy frame camera" nhưng không
`new OpenCvVideoSource()`. Nó khai báo `IVideoSource`; Infrastructure (cấp thấp) cài đặt; Web tiêm vào qua DI.
Cả hai phụ thuộc vào abstraction → chiều phụ thuộc **bị đảo ngược** so với code gọi thẳng.

---

## 3. Bản đồ chuyển đổi Python → .NET

| File Python | Vai trò | Sang tầng | Thành |
|---|---|---|---|
| `observation.py`, `gallery.py` | dữ liệu 1 mặt, so cosine | **Domain** | `FaceObservation`, `FaceMatch`, `FaceGallery` |
| `attendance.py` | tracker + quyết định | **Domain** | `FaceTracker`, `AttendanceDecider`, `ILivenessPolicy` |
| một phần `face_engine.py` | cổng chất lượng | **Domain** | `RecognitionGate`, `LivenessGate` |
| bảng trong `storage.py` | nhân viên, mẫu, sự kiện | **Domain** | `Employee`, `FaceSample`, `AttendanceEvent`, `RejectedEvent` |
| `pipeline.py`, `enrollment.py`, `sync.py` | điều phối | **Application** | use case + `PipelineService`, interface |
| `config.py` | cấu hình YAML | **Application/Web** | `IOptions<T>` + `appsettings.json` |
| `camera.py` | OpenCV VideoCapture | **Infrastructure** | `OpenCvVideoSource : IVideoSource` |
| `face_engine.py`, `recognizers.py`, `liveness.py` | model AI | **Infrastructure** | `OpenCvFaceAnalyzer` (OpenCvSharp + ONNX Runtime) |
| `storage.py` (SQL) | SQLite | **Infrastructure** | EF Core `DbContext` + Repository |
| `web/app.py` | FastAPI | **Web** | ASP.NET Core Minimal API (giữ nguyên UI `static/`) |
| `__main__.py` | ráp nối | **Web** | `Program.cs` |

---

## 4. Bản đồ pattern (từ file ôn tập) → vị trí trong dự án

| Kiến thức | Xuất hiện ở | Tầng / Bài |
|---|---|---|
| **SRP** | Tracker chỉ bám mặt; Decider chỉ quyết định; Gallery chỉ so khớp | Domain — Bài 01 |
| **OCP + Strategy** | `ILivenessPolicy`: Off / Opportunistic / Required thay cho `if mode == ...` | Domain — Bài 01 |
| **Factory Method** | `LivenessPolicyFactory.Create(mode)`; sau này `IRecognizerFactory` | Domain, Infra |
| **Abstract Factory** | `IRecognitionModelFamily` (SFace / ArcFace): tạo **trọn bộ** aligner + embedder tương thích nhau | Infrastructure — Bài 03 |
| **Keyed Services** | chọn họ model theo chuỗi config `"sface"` / `"arcface"` | Infrastructure — Bài 03 |
| **Repository + Unit of Work** | `IEmployeeRepository`, `IAttendanceRepository` + `IUnitOfWork` (EF Core) | Application khai báo, Infra cài — Bài 02/03 |
| **Decorator** | `LoggingBehavior` bọc mọi handler; `TimingFaceAnalyzer` đo thời gian quanh `IFaceAnalyzer` | Application — Bài 02, Infra — Bài 03 |
| **Chuyển tiếp DI (forwarding)** | `IUnitOfWork` → đúng DbContext của scope; camera vừa `IVideoSource` vừa `IHostedService` | Infrastructure — Bài 03 |
| **Mediator / CQRS** | Mediator tự viết (`ISender`), 11 Command/Query; phía đọc qua `IReportQueries` → DTO | Application — Bài 02 |
| **Factory dạng interface** | `IFaceAnalyzerFactory` — so sánh với `LivenessPolicyFactory` dạng static | Application — Bài 02 |
| **Abstraction thời gian** | `TimeProvider` (.NET 8) + `LocalClock` thay cho `DateTimeOffset.Now` | Application — Bài 02 |
| **Singleton (DI)** | `FaceGallery`, `IVideoSource`, `PipelineService` (BackgroundService) | Web — Bài 04 |
| **Scoped** | `DbContext`, repository — 1 instance / 1 HTTP request | Web — Bài 04 |
| **Bẫy Captive Dependency** | Singleton (pipeline) cần Scoped (DbContext) → phải dùng `IServiceScopeFactory` | Web — Bài 04 |
| **Runtime polymorphism** | Decider gọi `policy.CountsTowardVote()` — CLR chọn hàm theo object thật lúc chạy | Domain — Bài 01 |

---

## 5. Lộ trình học

| Bài | Tầng | Trạng thái |
|---|---|---|
| 00 | Tổng quan (bài này) | ✅ |
| 01 | Domain: entity, value object, tracker, decider, Strategy | ✅ code + 36 test |
| 02 | Application: interface (port), use case, pipeline điều phối, CQRS, Mediator, DI lifetimes | ✅ code + 17 test |
| 03 | Infrastructure: OpenCvSharp, ONNX, EF Core, Repository, Decorator, Abstract Factory | ✅ code + 28 test, đối chiếu khớp bản Python |
| 04 | Web: Program.cs, DI lifetimes, BackgroundService, Minimal API, MJPEG, Basic Auth | ⏳ |

Mỗi bài: đọc code theo thứ tự gợi ý → chạy test → làm bài tập nhỏ cuối bài.
