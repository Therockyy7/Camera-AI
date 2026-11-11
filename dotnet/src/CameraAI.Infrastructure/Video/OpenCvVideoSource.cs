using System.Diagnostics;
using System.Text.RegularExpressions;
using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Video;

internal enum SourceKind { Webcam, Stream, File }

internal sealed record VideoSourceSpec(SourceKind Kind, string Value)
{
    /// <summary>"0" → webcam số 0 · "rtsp://..." → luồng mạng · còn lại → đường dẫn file video.</summary>
    public static VideoSourceSpec Parse(string source)
    {
        var s = source.Trim();
        if (int.TryParse(s, out _)) return new(SourceKind.Webcam, s);
        if (s.Contains("://")) return new(SourceKind.Stream, s);
        return new(SourceKind.File, Path.GetFullPath(s));
    }

    /// <summary>Giấu mật khẩu trong URL trước khi ghi log / trả về web: rtsp://admin:***@192.168.1.64/...</summary>
    public static string Mask(string source) => Regex.Replace(source, @"(://[^:/@]+):[^@/]+@", "$1:***@");
}

/// <summary>
/// Nguồn video bằng OpenCV — tương đương camera.py. Một thread nền đọc LIÊN TỤC, chỉ giữ frame MỚI NHẤT
/// (không xếp hàng) → xử lý chậm hơn tốc độ camera thì bỏ frame cũ, không bị trễ dồn của RTSP.
/// Mất tín hiệu → tự kết nối lại, chờ tăng dần 1s, 2s, 4s… tối đa 30s.
///
/// Vừa là <see cref="IVideoSource"/> (cho Application đọc) vừa là <see cref="BackgroundService"/> (ASP.NET Core
/// tự Start khi app chạy, tự Stop khi Ctrl+C). DependencyInjection.cs "chuyển tiếp" cả hai về CÙNG một object.
/// </summary>
internal sealed class OpenCvVideoSource : BackgroundService, IVideoSource
{
    private const int MaxReadFailures = 10;
    private static readonly TimeSpan MaxBackoff = TimeSpan.FromSeconds(30);

    private readonly CameraOptions _cfg;
    private readonly VideoSourceSpec _spec;
    private readonly ILogger<OpenCvVideoSource> _log;
    private readonly Lock _lock = new();
    private Mat? _latest;
    private long _frameId;
    private bool _connected, _ended;
    private double _fps;
    private string? _lastError;

    static OpenCvVideoSource()
    {
        // RTSP qua TCP ổn định hơn UDP khi mạng nội bộ mất gói. Phải đặt TRƯỚC khi mở VideoCapture.
        if (Environment.GetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS") is null)
            Environment.SetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp");
    }

    public OpenCvVideoSource(IOptions<CameraAiOptions> options, ILogger<OpenCvVideoSource> log)
    {
        _cfg = options.Value.Camera;
        _spec = VideoSourceSpec.Parse(_cfg.Source);
        _log = log;
    }

    // ---------------- IVideoSource ----------------
    public IVideoFrame? TryGetLatest(long afterFrameId)
    {
        lock (_lock)
        {
            if (_latest is null || _frameId <= afterFrameId) return null;
            return new MatFrame(_frameId, _latest.Clone()); // BẢN SAO: người gọi Dispose, thread đọc vẫn thay frame tự do
        }
    }

    public CameraStatus Status
    {
        get
        {
            lock (_lock)
                return new CameraStatus(_connected, _ended, VideoSourceSpec.Mask(_cfg.Source),
                    _latest?.Width ?? 0, _latest?.Height ?? 0, Math.Round(_fps, 1), _lastError);
        }
    }

    // ---------------- BackgroundService ----------------
    // cap.Read() là lệnh CHẶN (đợi camera) → chạy trên thread riêng (LongRunning), không chiếm thread pool.
    protected override Task ExecuteAsync(CancellationToken ct) =>
        Task.Factory.StartNew(() => Run(ct), ct, TaskCreationOptions.LongRunning, TaskScheduler.Default);

    private void Run(CancellationToken ct)
    {
        var backoff = TimeSpan.FromSeconds(1);
        while (!ct.IsCancellationRequested)
        {
            VideoCapture? cap = null;
            try
            {
                cap = Open();
                if (cap.IsOpened())
                {
                    backoff = TimeSpan.FromSeconds(1);
                    ReadLoop(cap, ct);
                }
                else SetError($"Không mở được nguồn video {VideoSourceSpec.Mask(_cfg.Source)}");
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "Lỗi nguồn video");
                SetError(ex.Message);
            }
            finally
            {
                cap?.Dispose();
                lock (_lock) _connected = false;
            }

            if (_spec.Kind == SourceKind.File)
            {
                lock (_lock) _ended = true; // file phát hết thì dừng, không mở lại
                return;
            }
            if (ct.WaitHandle.WaitOne(backoff)) return;
            backoff = TimeSpan.FromTicks(Math.Min(backoff.Ticks * 2, MaxBackoff.Ticks));
        }
    }

    private VideoCapture Open()
    {
        switch (_spec.Kind)
        {
            case SourceKind.Webcam:
                // DirectShow mở webcam trên Windows nhanh hơn backend mặc định (MSMF).
                var cap = new VideoCapture(int.Parse(_spec.Value),
                    OperatingSystem.IsWindows() ? VideoCaptureAPIs.DSHOW : VideoCaptureAPIs.ANY);
                cap.Set(VideoCaptureProperties.FourCC, VideoWriter.FourCC('M', 'J', 'P', 'G'));
                if (_cfg.Width > 0 && _cfg.Height > 0)
                {
                    cap.Set(VideoCaptureProperties.FrameWidth, _cfg.Width);
                    cap.Set(VideoCaptureProperties.FrameHeight, _cfg.Height);
                }
                return cap;
            case SourceKind.Stream:
                var stream = new VideoCapture(_spec.Value, VideoCaptureAPIs.FFMPEG);
                stream.Set(VideoCaptureProperties.BufferSize, 1);
                return stream;
            default:
                return new VideoCapture(_spec.Value);
        }
    }

    private void ReadLoop(VideoCapture cap, CancellationToken ct)
    {
        var interval = TimeSpan.Zero;
        if (_spec.Kind == SourceKind.File) // phát file đúng tốc độ gốc, không tua nhanh
        {
            var fps = cap.Get(VideoCaptureProperties.Fps);
            interval = TimeSpan.FromSeconds(1 / (fps > 0 ? fps : 25));
        }
        lock (_lock)
        {
            _connected = true;
            _lastError = null;
        }
        _log.LogInformation("Đã kết nối nguồn video {Source}", VideoSourceSpec.Mask(_cfg.Source));

        int failures = 0, counted = 0;
        var window = Stopwatch.StartNew();
        while (!ct.IsCancellationRequested)
        {
            long started = Stopwatch.GetTimestamp();
            var frame = new Mat();
            if (!cap.Read(frame) || frame.Empty())
            {
                frame.Dispose();
                if (_spec.Kind == SourceKind.File) return;
                if (++failures >= MaxReadFailures)
                {
                    SetError("Mất tín hiệu camera (đọc frame lỗi liên tiếp)");
                    return;
                }
                Thread.Sleep(50);
                continue;
            }
            failures = 0;
            counted++;
            lock (_lock)
            {
                var old = _latest;
                (_latest, _frameId) = (frame, _frameId + 1);
                old?.Dispose(); // an toàn: TryGetLatest chỉ đọc _latest bên trong lock và trả về bản sao
                if (window.Elapsed >= TimeSpan.FromSeconds(1))
                {
                    _fps = counted / window.Elapsed.TotalSeconds;
                    counted = 0;
                    window.Restart();
                }
            }
            var remaining = interval - Stopwatch.GetElapsedTime(started);
            if (remaining > TimeSpan.Zero) Thread.Sleep(remaining);
        }
    }

    private void SetError(string message)
    {
        _log.LogWarning("{Message}", message);
        lock (_lock) _lastError = message;
    }

    public override void Dispose()
    {
        base.Dispose();
        lock (_lock) _latest?.Dispose();
    }
}
