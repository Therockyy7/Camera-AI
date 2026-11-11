# CameraAI .NET — bản port Clean Architecture (kèm bài giảng)

Port của ứng dụng chấm công bằng khuôn mặt (bản Python ở thư mục cha) sang .NET 10, làm từng tầng để học
Clean Architecture + Design Patterns trong ASP.NET Core.

```text
dotnet/
├── CameraAI.slnx
├── docs/                         ← bài giảng, đọc theo thứ tự
│   ├── 00-kien-truc-tong-quan.md
│   ├── 01-domain.md
│   └── 02-application.md
├── src/
│   ├── CameraAI.Domain/          lõi nghiệp vụ, không phụ thuộc gì
│   ├── CameraAI.Application/     use case + interface        (→ Domain)
│   ├── CameraAI.Infrastructure/  OpenCV, ONNX, EF Core       (→ Application)
│   └── CameraAI.Web/             ASP.NET Core, DI            (→ Application, Infrastructure)
└── tests/
    ├── CameraAI.Domain.Tests/
    └── CameraAI.Application.Tests/
```

```powershell
cd D:\CameraAI\dotnet
dotnet build
dotnet test
```

| Tầng | Trạng thái |
|---|---|
| Domain | ✅ xong, 36 test |
| Application | ✅ xong, 17 test |
| Infrastructure | ⏳ |
| Web | ⏳ (hiện là template trống) |

Model ONNX dùng chung thư mục `../models` với bản Python.
