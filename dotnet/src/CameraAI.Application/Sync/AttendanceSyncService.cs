using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using Microsoft.Extensions.Logging;

namespace CameraAI.Application.Sync;

/// <summary>
/// Gửi bù các lần chấm chưa đồng bộ sang hệ thống ngoài qua <see cref="IAttendanceSink"/>.
/// Chưa có sink thật nên chưa được chạy định kỳ; nối Fastdo = cài IAttendanceSink + gọi RunOnceAsync định kỳ.
/// </summary>
public sealed class AttendanceSyncService(
    IReportQueries queries, IAttendanceRepository attendance, IUnitOfWork uow, IAttendanceSink sink,
    LocalClock clock, ILogger<AttendanceSyncService> log)
{
    public async Task<int> RunOnceAsync(int batchSize = 100, CancellationToken ct = default)
    {
        var pending = await queries.PendingSyncAsync(batchSize, ct);
        if (pending.Count == 0) return 0;

        IReadOnlyCollection<int> sent;
        try
        {
            sent = await sink.SendAsync(pending, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            log.LogError(ex, "Gửi lần chấm công thất bại, sẽ thử lại sau");
            return 0;
        }

        var now = clock.Now.TrimToSeconds();
        foreach (var e in await attendance.GetManyAsync(sent, ct)) e.MarkSynced(now);
        await uow.SaveChangesAsync(ct);
        return sent.Count;
    }
}
