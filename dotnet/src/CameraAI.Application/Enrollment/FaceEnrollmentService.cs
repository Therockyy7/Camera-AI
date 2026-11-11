using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Domain.Employees;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CameraAI.Application.Enrollment;

/// <summary>
/// Giữ MỘT analyzer riêng cho việc đăng ký mẫu + khoá để mỗi lúc chỉ một lượt chụp chạy.
/// SINGLETON: model nạp một lần, dùng mãi. Tách khỏi FaceEnrollmentService (Scoped) vì service cần
/// repository (Scoped) — Scoped được phép giữ Singleton, ngược lại thì không.
/// </summary>
public sealed class EnrollmentAnalyzer(IFaceAnalyzerFactory factory) : IDisposable
{
    private readonly SemaphoreSlim _lock = new(1, 1);
    private IFaceAnalyzer? _analyzer;

    public async Task<T> UseAsync<T>(Func<IFaceAnalyzer, Task<T>> work, CancellationToken ct)
    {
        await _lock.WaitAsync(ct);
        try
        {
            _analyzer ??= factory.Create(); // tạo lười: chưa ai đăng ký thì không tốn RAM cho model
            return await work(_analyzer);
        }
        finally
        {
            _lock.Release();
        }
    }

    public void Dispose()
    {
        _analyzer?.Dispose();
        _lock.Dispose();
    }
}

/// <summary>
/// Đăng ký mẫu khuôn mặt: chụp từ camera đang chạy, hoặc từ ảnh tải lên; tính lại embedding khi đổi model.
/// Chụp phía server (lấy frame của IVideoSource) vì trên Windows webcam bị khoá cho một tiến trình.
/// </summary>
public sealed class FaceEnrollmentService(
    EnrollmentAnalyzer analyzer,
    IVideoSource source,
    IFrameImaging imaging,
    IEmployeeRepository employees,
    IFaceSampleRepository samples,
    IUnitOfWork uow,
    IFileStore files,
    LocalClock clock,
    TimeProvider time,
    IOptions<CameraAiOptions> options,
    ILogger<FaceEnrollmentService> log)
{
    private readonly CameraAiOptions _options = options.Value;

    /// <summary>Lấy tối đa N mẫu KHÁC NHAU trong thời gian chờ. Người dùng xoay đầu chậm để mẫu phủ nhiều góc.</summary>
    public async Task<CaptureResult> CaptureAsync(int employeeId, CancellationToken ct = default)
    {
        await RequireEmployeeAsync(employeeId, ct);
        return await analyzer.UseAsync(a => CaptureLoopAsync(a, employeeId, ct), ct);
    }

    private async Task<CaptureResult> CaptureLoopAsync(IFaceAnalyzer a, int employeeId, CancellationToken ct)
    {
        var cfg = _options.Enrollment;
        var issues = new Dictionary<EnrollmentIssue, int>();
        int saved = 0, attempts = 0;
        long lastId = -1;

        var existing = (await samples.ListByEmployeeAsync(employeeId, ct))
            .SelectMany(s => s.Embeddings).Where(e => e.Model == a.ModelName)
            .Select(e => VectorMath.Normalize(e.Vector)).ToList();
        var deadline = time.GetUtcNow() + TimeSpan.FromSeconds(cfg.CaptureTimeoutSeconds);

        while (saved < cfg.SamplesPerCapture && time.GetUtcNow() < deadline)
        {
            using var frame = source.TryGetLatest(lastId);
            if (frame is null)
            {
                await DelayAsync(cfg.PollIntervalMs, ct);
                continue;
            }
            lastId = frame.Id;
            attempts++;

            var (face, issue) = Check(a, frame, requireReal: true);
            float[]? normalized = null;
            if (face is not null)
            {
                normalized = VectorMath.Normalize(face.Embedding!);
                // Chỉ nhận mẫu khác đủ so với mọi mẫu đã có → gallery trải nhiều góc mặt.
                if (existing.Count > 0 && existing.Max(e => VectorMath.Dot(e, normalized)) >= cfg.MaxSimilarity)
                    issue = EnrollmentIssue.Similar;
            }

            if (issue is { } i) issues[i] = issues.GetValueOrDefault(i) + 1;
            else
            {
                await SaveAsync(employeeId, frame, face!, SampleSource.Camera, a.ModelName, ct);
                existing.Add(normalized!);
                saved++;
            }
            await DelayAsync(cfg.SampleIntervalMs, ct);
        }

        if (attempts == 0) issues[EnrollmentIssue.NoCamera] = 1;
        return new CaptureResult(saved, attempts,
            issues.Select(p => new IssueCount(p.Key.Code(), p.Value, p.Key.Text())).ToList());
    }

    /// <summary>Lưu mẫu từ ảnh tải lên. null = thành công, ngược lại là lý do.</summary>
    public async Task<EnrollmentIssue?> AddUploadAsync(int employeeId, byte[] data, CancellationToken ct = default)
    {
        await RequireEmployeeAsync(employeeId, ct);
        using var image = data.Length > 0 ? imaging.Decode(data) : null;
        if (image is null) return EnrollmentIssue.NotImage;

        return await analyzer.UseAsync<EnrollmentIssue?>(async a =>
        {
            var (face, issue) = Check(a, image, requireReal: false); // ảnh tải lên vốn là "ảnh" → không kiểm giả
            if (face is not null) await SaveAsync(employeeId, image, face, SampleSource.Upload, a.ModelName, ct);
            return issue;
        }, ct);
    }

    /// <summary>Đổi model → tính embedding còn thiếu từ ảnh mẫu đã lưu (không phải đăng ký lại). Trả về số mẫu đã tính.</summary>
    public Task<int> ReindexAsync(CancellationToken ct = default) => analyzer.UseAsync(async a =>
    {
        int done = 0;
        foreach (var sample in await samples.MissingEmbeddingAsync(a.ModelName, ct))
        {
            var bytes = sample.ImagePath is null ? null : await files.ReadAsync(sample.ImagePath, ct);
            using var image = bytes is null ? null : imaging.Decode(bytes);
            var embedding = image is null ? null : a.EmbedLargestFace(image);
            if (embedding is null)
            {
                log.LogWarning("Mẫu #{Id}: không tính được embedding {Model} (thiếu ảnh hoặc không thấy mặt)", sample.Id, a.ModelName);
                continue;
            }
            sample.SetEmbedding(a.ModelName, embedding);
            done++;
        }
        if (done > 0)
        {
            await uow.SaveChangesAsync(ct);
            log.LogInformation("Đã tính embedding {Model} cho {Count} mẫu", a.ModelName, done);
        }
        return done;
    }, ct);

    private (FaceObservation? Face, EnrollmentIssue? Issue) Check(IFaceAnalyzer a, IVideoFrame frame, bool requireReal)
    {
        var cfg = _options.Enrollment;
        var gate = _options.Recognition.ToGate() with { MinFacePx = cfg.MinFacePx }; // cổng chặt hơn lúc nhận diện
        var faces = a.Analyze(frame, gate);
        if (faces.Count == 0) return (null, EnrollmentIssue.NoFace);

        var large = faces.Where(f => f.RejectReason != RejectReason.Small).ToList(); // mặt nhỏ = người ở hậu cảnh
        if (large.Count == 0) return (null, EnrollmentIssue.Small);
        if (large.Count > 1) return (null, EnrollmentIssue.ManyFaces);

        var face = large[0];
        if (face.RejectReason is { } reason) return (null, EnrollmentIssueText.From(reason));
        if (face.DetScore < cfg.MinDetScore) return (null, EnrollmentIssue.LowScore);
        if (requireReal && face.IsReal is false) return (null, EnrollmentIssue.Spoof);
        return (face, null);
    }

    private async Task SaveAsync(int employeeId, IVideoFrame frame, FaceObservation face, SampleSource from, string model, CancellationToken ct)
    {
        var path = await files.SaveAsync($"faces/{employeeId}", imaging.EncodeCrop(frame, face.Box, scale: 1.6), ct);
        samples.Add(FaceSample.Create(employeeId, path, from, clock.Now.TrimToSeconds(), model, face.Embedding!));
        await uow.SaveChangesAsync(ct);
    }

    private async Task RequireEmployeeAsync(int employeeId, CancellationToken ct)
    {
        if (await employees.GetActiveAsync(employeeId, ct) is null) throw new NotFoundException("Không tìm thấy nhân viên");
    }

    private Task DelayAsync(int ms, CancellationToken ct) =>
        ms > 0 ? Task.Delay(TimeSpan.FromMilliseconds(ms), time, ct) : Task.CompletedTask;
}
