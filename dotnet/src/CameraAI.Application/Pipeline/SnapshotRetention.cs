using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CameraAI.Application.Pipeline;

/// <summary>
/// Xoá ảnh chụp cũ hơn N ngày (dữ liệu sinh trắc — không giữ vô thời hạn).
/// Lần chấm công GIỮ LẠI, chỉ bỏ ảnh; sự kiện bị từ chối thì xoá hẳn.
/// </summary>
public sealed class SnapshotRetention(
    IAttendanceRepository attendance, IUnitOfWork uow, IFileStore files, LocalClock clock,
    IOptions<CameraAiOptions> options, ILogger<SnapshotRetention> log)
{
    public async Task<int> CleanupAsync(CancellationToken ct = default)
    {
        var cutoff = clock.Now - TimeSpan.FromDays(options.Value.Storage.SnapshotRetentionDays);
        var kept = await attendance.WithSnapshotBeforeAsync(cutoff, ct);
        var rejected = await attendance.RejectedBeforeAsync(cutoff, ct);

        var paths = kept.Select(a => a.SnapshotPath).Concat(rejected.Select(r => r.SnapshotPath))
            .OfType<string>().ToList();
        foreach (var a in kept) a.ClearSnapshot();
        foreach (var r in rejected) attendance.Remove(r);
        await uow.SaveChangesAsync(ct);

        // Xoá file SAU khi DB đã ghi xong: nếu DB lỗi thì ảnh vẫn còn, không có bản ghi trỏ tới file đã mất.
        foreach (var path in paths) files.Delete(path);
        if (paths.Count > 0) log.LogInformation("Đã xoá {Count} ảnh chụp cũ hơn {Cutoff:yyyy-MM-dd}", paths.Count, cutoff);
        return paths.Count;
    }
}
