using CameraAI.Application.Pipeline;

namespace CameraAI.Web.Hosting;

/// <summary>
/// Chạy vòng xử lý camera NỀN suốt đời ứng dụng. Đây là thứ làm pipeline "chạy" — Singleton chỉ đảm bảo có một
/// pipeline; BackgroundService mới là thứ gọi RunAsync và huỷ nó khi Ctrl+C (stoppingToken).
/// </summary>
internal sealed class PipelineWorker(AttendancePipeline pipeline, ILogger<PipelineWorker> log) : BackgroundService
{
    protected override Task ExecuteAsync(CancellationToken stoppingToken) =>
        // Task.Run: phân tích ảnh là việc CPU nặng chạy liên tục → đưa ra khỏi luồng khởi động của host.
        Task.Run(async () =>
        {
            try
            {
                await pipeline.RunAsync(stoppingToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                log.LogCritical(ex, "Vòng xử lý camera dừng do lỗi");
                throw;
            }
        }, stoppingToken);
}
