// =====================================================================================================
//  COMPOSITION ROOT — nơi DUY NHẤT trong solution "biết hết": ghép Domain + Application + Infrastructure
//  lại thành một ứng dụng chạy được. Tương đương cameraai/__main__.py bên Python.
//
//  Đọc từ trên xuống là thấy cả vòng đời ứng dụng:
//    1. Cấu hình     2. Đăng ký DI (mỗi tầng tự khai qua AddXxx())     3. Pipeline HTTP (middleware)
//    4. Endpoint     5. Chạy (hosted services khởi động: camera → DB/model → vòng xử lý)
// =====================================================================================================
using System.Text.Json;
using CameraAI.Application;
using CameraAI.Infrastructure;
using CameraAI.Web.Endpoints;
using CameraAI.Web.Hosting;
using CameraAI.Web.Http;

var builder = WebApplication.CreateBuilder(args.Where(a => a != "--no-liveness").ToArray());

// ---------- 1. Cấu hình ----------
builder.Configuration.AddCameraAiSources(args);           // biến môi trường kiểu Python + --source / --no-liveness
builder.Services.AddCameraAiOptions(builder.Configuration);
var web = WebOptionsReader.Read(builder.Configuration);
builder.WebHost.UseUrls($"http://{web.Host}:{web.Port}");

// ---------- 2. Đăng ký DI ----------
builder.Services.AddApplication();                         // use case, pipeline, mediator (Bài 02)
builder.Services.AddInfrastructure();                      // EF Core, camera, model AI (Bài 03) — camera là hosted service #1
builder.Services.AddHostedService<StartupTasks>();         // hosted service #2: DB + kiểm model, CHỜ xong mới tới #3
builder.Services.AddHostedService<PipelineWorker>();       // hosted service #3: vòng xử lý camera chạy nền

// JSON snake_case: SampleCount → "sample_count" — khớp giao diện web cũ của bản Python.
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower);
builder.Services.AddProblemDetails();
builder.Services.AddExceptionHandler<ApiExceptionHandler>();

var app = builder.Build();

// ---------- 3. Pipeline HTTP: request đi qua theo ĐÚNG thứ tự này ----------
app.UseExceptionHandler();                 // ngoài cùng: bắt lỗi từ mọi lớp bên trong → ProblemDetails
app.UseMiddleware<BasicAuthMiddleware>();  // chưa đăng nhập thì dừng ở đây (401), không tới được file hay API
app.UseStaticFiles();                      // /static/app.js, /static/app.css từ wwwroot

// ---------- 4. Endpoint ----------
app.MapCameraAi();

// ---------- 5. Chạy ----------
app.Run();

/// <summary>Cho project test (WebApplicationFactory&lt;Program&gt;) thấy được class Program sinh tự động.</summary>
public partial class Program;
