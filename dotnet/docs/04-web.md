# Bài 04 — Tầng Web: composition root, hosting và HTTP

> Project: `src/CameraAI.Web` (→ Application + Infrastructure) · Test: `tests/CameraAI.Web.Tests` (10 test tích hợp)
>
> Chạy: `cd D:\CameraAI\dotnet` rồi `dotnet run --project src/CameraAI.Web` → mở http://localhost:8000

---

## 1. Web làm hai việc

1. **Composition root**: nơi *duy nhất* biết mọi tầng, ghép chúng thành một ứng dụng chạy được (`Program.cs`).
2. **Lớp vỏ HTTP mỏng**: đổi request HTTP thành Command/Query rồi gửi qua `ISender`. Không có dòng nghiệp vụ nào
   ở đây.

```text
src/CameraAI.Web/
├── Program.cs                       ★ đọc từ trên xuống = cả vòng đời ứng dụng
├── appsettings.json                 cấu hình (thay config.yaml)
├── Hosting/
│   ├── CameraAiConfiguration.cs     nguồn cấu hình + Options pattern + kiểm tra lúc khởi động
│   ├── StartupTasks.cs              IHostedService: tạo DB, tính lại embedding, kiểm model
│   └── PipelineWorker.cs            BackgroundService: chạy vòng xử lý camera
├── Http/
│   ├── ApiExceptionHandler.cs       ★ lỗi nghiệp vụ → mã HTTP (ProblemDetails)
│   └── BasicAuthMiddleware.cs       middleware đăng nhập
├── Endpoints/CameraAiEndpoints.cs   ★ 11 endpoint Minimal API + luồng MJPEG
└── wwwroot/                         giao diện web cũ của bản Python, giữ nguyên
```

---

## 2. `Program.cs`: năm bước

```csharp
var builder = WebApplication.CreateBuilder(args);
// 1. Cấu hình
builder.Configuration.AddCameraAiSources(args);
builder.Services.AddCameraAiOptions(builder.Configuration);
// 2. DI — mỗi tầng tự khai báo phần của mình
builder.Services.AddApplication();
builder.Services.AddInfrastructure();
builder.Services.AddHostedService<StartupTasks>();
builder.Services.AddHostedService<PipelineWorker>();
var app = builder.Build();
// 3. Middleware (THỨ TỰ quan trọng)
app.UseExceptionHandler(); app.UseMiddleware<BasicAuthMiddleware>(); app.UseStaticFiles();
// 4. Endpoint
app.MapCameraAi();
// 5. Chạy
app.Run();
```

So với `__main__.py` bên Python, ở đây **không có dòng `new` nào**. Program.cs chỉ nói "dùng những gì", còn DI tự
tạo và nối mọi thứ.

---

## 3. Cấu hình và Options pattern

### 3.1 Nguồn cấu hình: cái sau ghi đè cái trước

```text
appsettings.json
  → appsettings.Development.json
    → biến môi trường kiểu .NET          CameraAI__Camera__Source=1
      → biến môi trường kiểu Python        CAMERA_SOURCE=1, ADMIN_PASSWORD=...   (để dùng lại file .env cũ)
        → dòng lệnh                          dotnet run -- --source 1 --no-liveness
```

### 3.2 Options pattern

```csharp
services.AddOptions<CameraAiOptions>()
    .Bind(config.GetSection("CameraAI"))                         // JSON → object C# có kiểu
    .PostConfigure<IHostEnvironment>((o, env) => ...)            // đổi "../../../data" thành đường dẫn tuyệt đối
    .Validate(o => IsLoopback(o.Web.Host) || o.Web.AdminPassword is not "", "…bắt buộc ADMIN_PASSWORD")
    .ValidateOnStart();                                          // sai cấu hình → KHÔNG khởi động
```

`ValidateOnStart` giúp app **chết ngay lúc khởi động** với thông báo rõ, thay vì chạy rồi lộ ra lỗi bảo mật. Test
`Listening_on_lan_without_password_refuses_to_start` khoá chặt quy tắc này.

> 💡 Phỏng vấn hay hỏi: `IOptions<T>` (đọc 1 lần, Singleton) · `IOptionsSnapshot<T>` (đọc lại mỗi request, Scoped)
> · `IOptionsMonitor<T>` (tự cập nhật khi file đổi, có sự kiện OnChange). Dự án dùng `IOptions<T>` vì đổi cấu hình
> camera/model cần khởi động lại.

---

## 4. Host và hosted service: thứ tự khởi động

```text
app.Run()
 ├─ 1. OpenCvVideoSource.StartAsync   (đăng ký trong AddInfrastructure)  → thread đọc webcam bắt đầu
 ├─ 2. StartupTasks.StartAsync        CHỜ xong: tạo DB, tính lại embedding, nạp model (thiếu .onnx → dừng app)
 ├─ 3. PipelineWorker.StartAsync      → Task.Run(pipeline.RunAsync)      vòng xử lý 8 frame/giây
 └─ 4. Kestrel bắt đầu nhận HTTP      (cổng 8000)
Ctrl+C → ApplicationStopping → các token bị huỷ → StopAsync theo thứ tự NGƯỢC lại
```

**Hosted service khởi động theo thứ tự đăng ký**, và `StartAsync` của cái trước phải xong thì cái sau mới chạy. Vì
vậy chỉ cần đăng ký `StartupTasks` trước `PipelineWorker` là đủ đảm bảo DB có trước khi pipeline đọc.

Nhắc lại câu hỏi Singleton của bạn: `AttendancePipeline` là **Singleton** (một cái), còn **`PipelineWorker`
(BackgroundService) mới là thứ làm nó chạy**.

### Bẫy tắt máy: request "vô tận"

`/stream.mjpg` là một response **không bao giờ kết thúc**. Khi Ctrl+C, host đợi request đang chạy xong rồi mới thoát,
mặc định tới **30 giây**. Cách sửa: vòng lặp phát hình nghe **cả hai** tín hiệu dừng:

```csharp
using var stop = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted, lifetime.ApplicationStopping);
```

---

## 5. Pipeline middleware: thứ tự là ý nghĩa

```text
request ─► ExceptionHandler ─► BasicAuth ─► StaticFiles ─► Endpoint (/api/…)
             │ bọc mọi thứ        │ chưa đăng nhập      │ /static/app.js        │ ISender.Send(...)
             │ bên trong          │ → 401, DỪNG         │ → trả file, DỪNG      │
response ◄───┴────────────────────┴─────────────────────┴───────────────────────┘
```

- `ExceptionHandler` đứng **ngoài cùng** để bắt lỗi ném ra từ mọi lớp bên trong.
- `BasicAuth` đứng **trước** `StaticFiles` và endpoint. Đặt sau thì ảnh khuôn mặt ở `/files/...` lọt ra ngoài mà
  không cần mật khẩu.

Middleware cũng là một dạng **Decorator / Chain of Responsibility**: mỗi lớp quyết định *gọi tiếp* (`next(context)`)
hay *trả lời luôn*.

---

## 6. Minimal API: endpoint mỏng

```csharp
api.MapPost("/employees", async (CreateEmployeeRequest body, ISender s, CancellationToken ct) =>
{
    var created = await s.Send(new CreateEmployeeCommand(body.Code, body.Name), ct);
    return Results.Created($"/api/employees/{created.Id}", created);
});
```

ASP.NET Core **tự bind** tham số theo kiểu:

| Tham số | Lấy từ |
|---|---|
| `int id` có `{id:int}` trong route | đường dẫn |
| `DateOnly? date` | query string `?date=2026-10-26` (sai định dạng → 400) |
| `CreateEmployeeRequest body` | body JSON |
| `IFormFileCollection files` | form multipart (upload) |
| `ISender`, `IFileStore`… | DI |
| `CancellationToken` | huỷ khi trình duyệt đóng kết nối |

### JSON snake_case: dùng lại giao diện cũ

```csharp
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower);
```

`SampleCount` → `"sample_count"`, `FirstTs` → `"first_ts"`, đúng tên mà `app.js` của bản Python đang đọc. Cả giao
diện chạy **không sửa một dòng JavaScript nào**.

---

## 7. Lỗi nghiệp vụ → HTTP: `ApiExceptionHandler`

| Exception (ai ném) | HTTP | Ví dụ `detail` |
|---|---|---|
| `DuplicateEmployeeException` (Domain) | 409 | "Mã nhân viên DEV01 đã tồn tại" |
| `DomainException` (Domain) | 400 | "Cần nhập mã và tên nhân viên" |
| `InvalidRequestException` (Application) | 400 | "Ngày kết thúc phải sau ngày bắt đầu" |
| `NotFoundException` (Application) | 404 | "Không tìm thấy nhân viên" |
| lỗi khác | 500 | (không lộ chi tiết, chỉ ghi log) |

Domain và Application **không biết HTTP**, chúng chỉ ném lỗi mang nghĩa nghiệp vụ. Chỉ một file ở tầng Web biết
"trùng mã = 409". Kết quả trả về dạng **ProblemDetails** (chuẩn RFC 9457), có trường `detail` mà `app.js` hiển thị.

---

## 8. Bản đồ vòng đời DI của CẢ dự án

| Singleton (cả app một cái) | Scoped (mỗi request / mỗi scope một cái) | Hosted (chạy nền) |
|---|---|---|
| `FaceGallery`, `EmployeeNames`, `LocalClock`, `TimeProvider` | `CameraAiDbContext` = `IUnitOfWork` | `OpenCvVideoSource` (cũng là `IVideoSource`) |
| `AttendancePipeline`, `AttendanceDecider`, `ILivenessPolicy` | 3 repository, `IReportQueries` | `StartupTasks` |
| `EnrollmentAnalyzer`, `IFaceAnalyzerFactory`, `IFrameImaging`, `IFileStore` | `GalleryLoader`, `FaceEnrollmentService` | `PipelineWorker` |
| `IRecognitionModelFamily` (keyed) | `ISender`, 11 handler, `LoggingBehavior` | |

Quy tắc xuyên suốt: **Singleton không được giữ Scoped**. Khi cần thì mở scope ngắn bằng `IServiceScopeFactory`
(pipeline ghi DB, `StartupTasks` tính lại embedding).

---

## 9. ★ Một request đi qua cả 4 tầng

`POST /api/employees {"code":"DEV01","name":"An"}`:

```text
WEB            Kestrel → ExceptionHandler → BasicAuth → endpoint: bind body JSON (snake_case)
               → sender.Send(new CreateEmployeeCommand("DEV01","An"))
APPLICATION    Sender (Mediator) → LoggingBehavior (Decorator) → CreateEmployeeHandler
               → employees.FindByCodeAsync("DEV01")                 [port IEmployeeRepository]
INFRASTRUCTURE   → EmployeeRepository → EF Core → SELECT … FROM employees WHERE code = 'DEV01'
DOMAIN         → Employee.Create("DEV01","An", now)                 validate, trim (rich model)
APPLICATION    → employees.Add(e); uow.SaveChangesAsync()
INFRASTRUCTURE   → CameraAiDbContext.SaveChanges → INSERT … (created_at = '2026-11-11T14:50:00+07:00')
APPLICATION    → EmployeeDto(...)
WEB            ← 201 Created  {"id":4,"code":"DEV01","name":"An","created_at":"…+07:00","sample_count":0}
```

Nếu mã đã tồn tại: Domain ném `DuplicateEmployeeException` → lỗi bay ngược ra qua các tầng → `ApiExceptionHandler`
đổi thành **409** + `detail`.

---

## 10. Test tích hợp với `WebApplicationFactory`

```csharp
public sealed class CameraAiApp : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureAppConfiguration(... DataDir tạm, camera = file không tồn tại ...);
        builder.ConfigureTestServices(s => s.AddSingleton<IFaceAnalyzerFactory>(new StubAnalyzerFactory(...)));
    }
}
```

Dựng **nguyên** ứng dụng thật trong bộ nhớ (Program.cs, middleware, DI, SQLite) rồi gọi HTTP thật. Chỉ thay model AI
bằng bản giả. Đây chính là lợi ích cuối cùng của việc đi qua interface: muốn thay chỗ nào thì chỉ cần **đăng ký
đè một dòng**. `public partial class Program;` ở cuối Program.cs là để test "nhìn thấy" class Program sinh tự động.

---

## 11. Kết quả chạy thật

| Kiểm tra | Kết quả |
|---|---|
| Webcam | kết nối, 1280×720, ~10 fps |
| Vòng xử lý | đúng 8 frame/giây; ~17 ms/frame (theo log của `TimingFaceAnalyzer`) |
| Dữ liệu cũ của bản Python | nạp 12 mẫu khuôn mặt / 3 nhân viên; tab Lịch sử hiện đủ 8 lần chấm + 8 ảnh |
| Giao diện | `/stream.mjpg` hiển thị video 1280×720; mọi tab hoạt động, không sửa JS |

---

## 12. Câu hỏi tự kiểm tra

1. Vì sao `BasicAuthMiddleware` phải đứng trước `UseStaticFiles`?
   <details><summary>Đáp án</summary>Middleware chạy theo thứ tự. StaticFiles trả file rồi dừng chuỗi; nếu nó đứng trước thì ảnh/giao diện được trả mà chưa kiểm mật khẩu.</details>
2. `PipelineWorker` và `AttendancePipeline` khác nhau thế nào? Sao không gộp một?
   <details><summary>Đáp án</summary>Pipeline (Application) là logic điều phối, không biết gì về hosting nên test được bằng cách gọi thẳng `ProcessFrameAsync`. Worker (Web) chỉ là "động cơ" của host gọi `RunAsync` và huỷ khi tắt app. Tách ra theo SRP và giữ Application không phụ thuộc ASP.NET Core.</details>
3. Nếu bỏ `ApplicationStopping` khỏi luồng MJPEG, chuyện gì xảy ra khi Ctrl+C lúc đang mở trang?
   <details><summary>Đáp án</summary>Host đợi request MJPEG (không bao giờ tự kết thúc) tới hết thời gian chờ tắt (mặc định 30 giây) mới thoát.</details>
4. Vì sao kiểm "mở LAN phải có mật khẩu" bằng `ValidateOnStart` thay vì `if` trong middleware?
   <details><summary>Đáp án</summary>Cấu hình sai thì app không được phép chạy chút nào: thất bại sớm, rõ ràng, trước khi có request nào lọt qua.</details>
5. Kể đường đi của `POST /api/employees` qua 4 tầng, và chỗ nào ném lỗi 409?

## 13. Bài tập

1. Thêm endpoint `PUT /api/employees/{id}` đổi tên (dùng `RenameEmployeeCommand` từ bài tập Bài 02). Mấy file phải sửa?
2. Thêm `GET /health` trả `{ "camera": true/false, "db": true/false }` dùng `AddHealthChecks()` có sẵn của ASP.NET Core.
3. Đổi `IOptions<CameraAiOptions>` trong `BasicAuthMiddleware` sang `IOptionsMonitor` để đổi mật khẩu không cần khởi
   động lại. Middleware là Singleton hay Scoped? Có dùng được `IOptionsSnapshot` không, vì sao?
