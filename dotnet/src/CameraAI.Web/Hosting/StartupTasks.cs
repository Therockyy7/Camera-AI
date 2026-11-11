using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Enrollment;
using CameraAI.Infrastructure.Persistence;
using Microsoft.Extensions.Options;

namespace CameraAI.Web.Hosting;

/// <summary>
/// Việc phải xong TRƯỚC khi pipeline chạy và web nhận request: tạo DB, tính embedding còn thiếu (khi đổi model).
///
/// IHostedService.StartAsync được host chạy LẦN LƯỢT theo thứ tự đăng ký và CHỜ xong mới chạy cái sau → đăng ký
/// StartupTasks trước PipelineWorker là đủ đảm bảo thứ tự. Ném lỗi ở đây (thiếu model...) = app dừng ngay với
/// thông báo rõ ràng, giống bản Python kiểm model lúc khởi động.
/// </summary>
internal sealed class StartupTasks(
    IServiceProvider services, IVideoSource source, IOptions<CameraAiOptions> options, ILogger<StartupTasks> log)
    : IHostedService
{
    public async Task StartAsync(CancellationToken ct)
    {
        var o = options.Value;
        await services.InitializeDatabaseAsync(ct);
        try
        {
            await using var scope = services.CreateAsyncScope();
            // Đổi model (sface ↔ arcface) → tính embedding mới từ ảnh mẫu đã lưu. Đồng thời nạp model lần đầu:
            // thiếu file .onnx thì lỗi NGAY lúc khởi động chứ không đợi tới frame đầu tiên.
            await scope.ServiceProvider.GetRequiredService<FaceEnrollmentService>().ReindexAsync(ct);
        }
        catch (FileNotFoundException ex)
        {
            log.LogCritical("{Message}", ex.Message);
            throw;
        }

        if (o.Recognition.Model == "arcface")
            log.LogWarning("ArcFace (InsightFace) chỉ dùng cho nghiên cứu phi thương mại — dùng thật phải mua license");
        log.LogInformation(
            "Nguồn video: {Source} | camera_id: {CameraId} | model: {Model} (ngưỡng {Threshold:F3}) | chống giả mạo: {Liveness}",
            source.Status.Source, o.Camera.Id, o.Recognition.Model, o.Recognition.ActiveThreshold, o.Liveness.Mode);
        var shownHost = CameraAiConfiguration.IsLoopback(o.Web.Host) ? "localhost" : "<địa-chỉ-IP-máy-này>";
        log.LogInformation("Mở trình duyệt: http://{Host}:{Port}  (Ctrl+C để dừng)", shownHost, o.Web.Port);
    }

    public Task StopAsync(CancellationToken ct) => Task.CompletedTask;
}
