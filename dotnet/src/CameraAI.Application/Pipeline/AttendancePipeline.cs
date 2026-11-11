using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Faces;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CameraAI.Application.Pipeline;

/// <summary>Sự kiện gần đây hiển thị ở tab Trực tiếp.</summary>
public sealed record RecentEvent(string Kind, int? EmployeeId, string Name, DateTimeOffset Ts, double Score, double? Liveness);

/// <summary>
/// Vòng xử lý nền — tương đương pipeline.py:  frame mới nhất → AI → gallery → tracker → decider → lưu → vẽ.
///
/// ĐÂY LÀ "ỨNG DỤNG" CỦA TẦNG APPLICATION: nó không tự làm việc nặng nào, chỉ ĐIỀU PHỐI:
///   • việc AI       → IFaceAnalyzer        (Infrastructure)
///   • việc so khớp  → FaceGallery          (Domain)
///   • việc quyết    → FaceTracker/Decider  (Domain)
///   • việc lưu      → Repository + UoW     (Infrastructure)
///   • việc vẽ       → IFrameImaging        (Infrastructure)
///
/// Vòng đời SINGLETON (một vòng lặp cho cả ứng dụng, giữ trạng thái tracker). Nhưng repository là SCOPED
/// (dùng DbContext) → KHÔNG được tiêm repository vào constructor (lỗi "captive dependency": DbContext bị giữ
/// mãi, không bao giờ giải phóng, và không an toàn đa luồng). Thay vào đó tiêm IServiceScopeFactory và
/// mở một scope ngắn mỗi lần cần ghi DB.
/// </summary>
public sealed class AttendancePipeline : IDisposable
{
    private const int MaxRecent = 20;

    private readonly IVideoSource _source;
    private readonly IFaceAnalyzerFactory _analyzers;
    private readonly IFrameImaging _imaging;
    private readonly FaceGallery _gallery;
    private readonly EmployeeNames _names;
    private readonly AttendanceDecider _decider;
    private readonly FaceLabeler _labeler;
    private readonly IServiceScopeFactory _scopes;
    private readonly LocalClock _clock;
    private readonly TimeProvider _time;
    private readonly CameraAiOptions _options;
    private readonly ILogger<AttendancePipeline> _log;

    private readonly FaceTracker _tracker;
    private readonly RecognitionGate _gate;
    private readonly LinkedList<RecentEvent> _recent = new();
    private IFaceAnalyzer? _analyzer;
    private volatile LatestJpeg? _latest;
    private long _jpegId;
    private DateOnly? _lastCleanup;

    public AttendancePipeline(
        IVideoSource source, IFaceAnalyzerFactory analyzers, IFrameImaging imaging, FaceGallery gallery,
        EmployeeNames names, AttendanceDecider decider, FaceLabeler labeler, IServiceScopeFactory scopes,
        LocalClock clock, TimeProvider time, IOptions<CameraAiOptions> options, ILogger<AttendancePipeline> log)
    {
        (_source, _analyzers, _imaging, _gallery, _names, _decider, _labeler, _scopes, _clock, _time, _log) =
            (source, analyzers, imaging, gallery, names, decider, labeler, scopes, clock, time, log);
        _options = options.Value;
        _tracker = new FaceTracker(window: _options.Attendance.WindowFrames);
        _gate = _options.Recognition.ToGate();
    }

    private sealed record LatestJpeg(long Id, byte[] Jpeg);

    // ---------------- cho web đọc ----------------
    public (long Id, byte[]? Jpeg) Latest => _latest is { } l ? (l.Id, l.Jpeg) : (0, null);
    public double ProcessedFps { get; private set; }

    public IReadOnlyList<RecentEvent> RecentEvents
    {
        get { lock (_recent) return _recent.ToList(); }
    }

    // ---------------- vòng đời ----------------

    /// <summary>Nạp gallery + lần chấm gần nhất từ DB (để cooldown đúng sau khi khởi động lại).</summary>
    public async Task PrimeAsync(CancellationToken ct = default)
    {
        await using var scope = _scopes.CreateAsyncScope();
        var services = scope.ServiceProvider;
        var since = _clock.Now - TimeSpan.FromMinutes(_options.Attendance.CooldownMinutes);
        _decider.Seed(await services.GetRequiredService<IAttendanceRepository>()
            .LastTimesAsync(_options.Camera.Id, since, ct));
        await services.GetRequiredService<GalleryLoader>().ReloadAsync(ct);
    }

    /// <summary>Chạy tới khi bị huỷ. Tầng Web gọi từ một BackgroundService.</summary>
    public async Task RunAsync(CancellationToken ct)
    {
        await PrimeAsync(ct);
        var interval = TimeSpan.FromSeconds(1 / Math.Max(_options.Processing.Fps, 0.1));
        long lastId = -1;
        int counted = 0;
        long windowStart = _time.GetTimestamp();
        try
        {
            while (!ct.IsCancellationRequested)
            {
                long started = _time.GetTimestamp();
                using (var frame = _source.TryGetLatest(lastId))
                {
                    if (frame is null)
                    {
                        await Task.Delay(TimeSpan.FromMilliseconds(20), _time, ct);
                        continue;
                    }
                    lastId = frame.Id;
                    try
                    {
                        await ProcessFrameAsync(frame, _clock.Now, ct);
                        await MaybeCleanupAsync(ct);
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException)
                    {
                        _log.LogError(ex, "Lỗi xử lý frame — bỏ qua frame này");
                    }
                }

                counted++;
                var window = _time.GetElapsedTime(windowStart);
                if (window >= TimeSpan.FromSeconds(2))
                {
                    ProcessedFps = Math.Round(counted / window.TotalSeconds, 1);
                    (counted, windowStart) = (0, _time.GetTimestamp());
                }
                var remaining = interval - _time.GetElapsedTime(started);
                if (remaining > TimeSpan.Zero) await Task.Delay(remaining, _time, ct);
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // dừng bình thường (Ctrl+C)
        }
    }

    // ---------------- xử lý 1 frame ----------------
    public async Task<IReadOnlyList<Decision>> ProcessFrameAsync(IVideoFrame frame, DateTimeOffset now, CancellationToken ct = default)
    {
        _analyzer ??= _analyzers.Create(); // tạo lười: model chỉ nạp khi có frame đầu tiên

        // 1. AI: ảnh → các FaceObservation (có embedding)   2. Domain: so khớp gallery
        var observations = _analyzer.Analyze(frame, _gate)
            .Select(o => o.Embedding is null ? o : o with { Match = _gallery.Match(o.Embedding) })
            .ToList();

        // 3. Domain: bám vệt + quyết định
        var pairs = _tracker.Update(observations, now);
        var decisions = new List<Decision>();
        foreach (var (track, _) in pairs)
        {
            if (_decider.Evaluate(track, now) is not { } decision) continue;
            await RecordAsync(frame, decision, now, ct);
            decisions.Add(decision);
        }

        // 4. Vẽ khung + nhãn, giữ JPEG mới nhất cho luồng MJPEG
        var labels = pairs.Select(p => _labeler.For(p.Track, p.Observation)).ToList();
        _latest = new LatestJpeg(Interlocked.Increment(ref _jpegId), _imaging.EncodeAnnotated(frame, labels));
        return decisions;
    }

    private async Task RecordAsync(IVideoFrame frame, Decision decision, DateTimeOffset now, CancellationToken ct)
    {
        var at = now.TrimToSeconds();
        var snapshot = _imaging.EncodeCrop(frame, decision.Observation.Box, scale: 2.0);
        var cameraId = _options.Camera.Id;

        await using (var scope = _scopes.CreateAsyncScope()) // scope ngắn: 1 DbContext cho đúng 1 lần ghi
        {
            var services = scope.ServiceProvider;
            var path = await services.GetRequiredService<IFileStore>().SaveAsync($"snapshots/{at:yyyy-MM-dd}", snapshot, ct);
            var repo = services.GetRequiredService<IAttendanceRepository>();
            if (decision.Kind == DecisionKind.Attendance)
                repo.Add(AttendanceEvent.Record(decision.EmployeeId!.Value, cameraId, at, decision.Score, decision.Liveness, path));
            else
                repo.Add(RejectedEvent.Record(cameraId, at, decision.Kind, decision.EmployeeId, decision.Score, decision.Liveness, path));
            await services.GetRequiredService<IUnitOfWork>().SaveChangesAsync(ct);
        }

        var name = _names.NameOf(decision.EmployeeId);
        if (decision.Kind == DecisionKind.Attendance)
            _log.LogInformation("Chấm công: {Name} (điểm khớp {Score:F2})", name, decision.Score);
        else
            _log.LogInformation("Từ chối ({Kind}), gần nhất: {Name} (điểm {Score:F2})", decision.Kind, name, decision.Score);

        var recent = new RecentEvent(decision.Kind.ToApiString(), decision.EmployeeId, name, at,
            Math.Round(decision.Score, 3), decision.Liveness is { } l ? Math.Round(l, 3) : null);
        lock (_recent)
        {
            _recent.AddFirst(recent);
            while (_recent.Count > MaxRecent) _recent.RemoveLast();
        }
    }

    private async Task MaybeCleanupAsync(CancellationToken ct)
    {
        var today = _clock.Today;
        if (_lastCleanup == today) return; // mỗi ngày dọn một lần
        _lastCleanup = today;
        await using var scope = _scopes.CreateAsyncScope();
        await scope.ServiceProvider.GetRequiredService<SnapshotRetention>().CleanupAsync(ct);
    }

    public void Dispose() => _analyzer?.Dispose();
}
