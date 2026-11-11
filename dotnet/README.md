# CameraAI .NET — bản port Clean Architecture (kèm bài giảng)

Port của ứng dụng chấm công bằng khuôn mặt (bản Python ở thư mục cha) sang .NET 10, làm từng tầng để học
Clean Architecture + Design Patterns trong ASP.NET Core. Kết quả nhận diện khớp bản Python (cosine embedding
≥ 0,999999), dùng chung model (`../models`), dữ liệu (`../data`) và giao diện web.

```text
dotnet/
├── CameraAI.slnx
├── docs/                         ← bài giảng, đọc theo thứ tự
│   ├── 00-kien-truc-tong-quan.md
│   ├── 01-domain.md
│   ├── 02-application.md
│   ├── 03-infrastructure.md
│   ├── 04-web.md
│   └── 05-luong-chay.html        ← sơ đồ toàn bộ luồng chạy (mở bằng trình duyệt)
├── src/
│   ├── CameraAI.Domain/          lõi nghiệp vụ, không phụ thuộc gì
│   ├── CameraAI.Application/     use case + interface        (→ Domain)
│   ├── CameraAI.Infrastructure/  OpenCV, ONNX, EF Core       (→ Application)
│   └── CameraAI.Web/             ASP.NET Core, DI, giao diện (→ Application, Infrastructure)
└── tests/
    ├── CameraAI.Domain.Tests/
    ├── CameraAI.Application.Tests/
    ├── CameraAI.Infrastructure.Tests/   (SQLite thật + model thật nếu có trong ../models)
    └── CameraAI.Web.Tests/              (WebApplicationFactory — gọi HTTP thật)
```

## Chạy

Yêu cầu: .NET 10 SDK, webcam, model trong `D:\CameraAI\models` (tải bằng bản Python:
`.venv\Scripts\python.exe -m cameraai.model_store download`).

```powershell
cd D:\CameraAI\dotnet
dotnet run --project src/CameraAI.Web
```

Mở http://localhost:8000. Dừng bằng `Ctrl+C`. Tham số giống bản Python:

| Lệnh | Tác dụng |
|---|---|
| `dotnet run --project src/CameraAI.Web -- --source 1` | webcam thứ 2 |
| `dotnet run --project src/CameraAI.Web -- --source D:\clip.mp4` | phát lại video |
| `dotnet run --project src/CameraAI.Web -- --no-liveness` | tắt chống giả mạo |

Cấu hình trong `src/CameraAI.Web/appsettings.json` (mục `CameraAI`, cùng ý nghĩa với `config.yaml`). Biến môi trường
của bản Python vẫn dùng được: `CAMERA_SOURCE`, `CAMERA_ID`, `CAMERAAI_HOST`, `CAMERAAI_PORT`, `ADMIN_PASSWORD`,
`CAMERAAI_DATA_DIR`, `LIVENESS_MODE`, `RECOGNITION_MODEL`.

> Mặc định bản .NET dùng **chung** `D:\CameraAI\data` với bản Python (nhân viên, mẫu khuôn mặt, lịch sử). Muốn tách
> riêng: đặt `CameraAI:Storage:DataDir` hoặc `CAMERAAI_DATA_DIR` sang thư mục khác. Đừng chạy hai bản cùng lúc
> (webcam chỉ mở được cho một tiến trình).

## Test

```powershell
dotnet test
```

| Tầng | Trạng thái |
|---|---|
| Domain | ✅ 36 test |
| Application | ✅ 17 test |
| Infrastructure | ✅ 28 test; kết quả AI khớp bản Python (cosine ≥ 0,999999) |
| Web | ✅ 10 test tích hợp; đã chạy thật với webcam (8 fps, ~17 ms/frame) |
