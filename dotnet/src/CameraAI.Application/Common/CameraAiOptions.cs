using CameraAI.Domain.Attendance;
using CameraAI.Domain.Faces;

namespace CameraAI.Application.Common;

/// <summary>
/// Cấu hình ứng dụng — tương đương config.yaml bên Python. Tầng Web bind từ appsettings.json
/// (mục "CameraAI") + biến môi trường, rồi tiêm vào nơi cần qua <c>IOptions&lt;CameraAiOptions&gt;</c>.
///
/// Options là "đồ của tầng ngoài" (chuỗi, số tuỳ ý). Các method To...() đổi chúng thành record của Domain
/// → Domain không bao giờ phụ thuộc vào IOptions hay appsettings.
/// </summary>
public sealed class CameraAiOptions
{
    public const string Section = "CameraAI";

    public string TimeZone { get; set; } = "Asia/Ho_Chi_Minh";
    public string ModelsDir { get; set; } = "models";
    public CameraOptions Camera { get; set; } = new();
    public ProcessingOptions Processing { get; set; } = new();
    public RecognitionOptions Recognition { get; set; } = new();
    public LivenessOptions Liveness { get; set; } = new();
    public AttendanceOptions Attendance { get; set; } = new();
    public EnrollmentOptions Enrollment { get; set; } = new();
    public StorageOptions Storage { get; set; } = new();
    public WebOptions Web { get; set; } = new();
}

public sealed class CameraOptions
{
    public string Source { get; set; } = "0";   // "0" = webcam, rtsp://..., hoặc file .mp4
    public string Id { get; set; } = "laptop";  // ghi vào mỗi lần chấm
    public int Width { get; set; } = 1280;
    public int Height { get; set; } = 720;
}

public sealed class ProcessingOptions
{
    public double Fps { get; set; } = 8;           // số frame xử lý mỗi giây
    public int DetectWidth { get; set; } = 640;    // thu nhỏ frame về bề rộng này để phát hiện mặt
}

public sealed class RecognitionOptions
{
    public string Model { get; set; } = "sface";   // sface | arcface
    public double SfaceThreshold { get; set; } = 0.363;
    public double ArcfaceThreshold { get; set; } = 0.30;
    public double DetectScore { get; set; } = 0.8;
    public int MinFacePx { get; set; } = 40;
    public double MinSharpness { get; set; } = 12;
    public double MaxTurn { get; set; } = 0.45;

    public double ActiveThreshold => Model == "arcface" ? ArcfaceThreshold : SfaceThreshold;
    public RecognitionGate ToGate() => new(MinFacePx, MaxTurn, MinSharpness);
}

public sealed class LivenessOptions
{
    public LivenessMode Mode { get; set; } = LivenessMode.Opportunistic;
    public double Threshold { get; set; } = 0.5;
    public double SpoofThreshold { get; set; } = 0.3;
    public int MinFacePx { get; set; } = 64;
    public double MaxTurn { get; set; } = 0.25;

    public LivenessGate ToGate() => new(MinFacePx, MaxTurn, Threshold);
}

public sealed class AttendanceOptions
{
    public int WindowFrames { get; set; } = 5;
    public int ConfirmFrames { get; set; } = 3;
    public int CooldownMinutes { get; set; } = 5;

    public AttendanceRules ToRules(double spoofThreshold) =>
        new(WindowFrames, ConfirmFrames, CooldownMinutes, spoofThreshold);
}

public sealed class EnrollmentOptions
{
    public int SamplesPerCapture { get; set; } = 8;
    public double CaptureTimeoutSeconds { get; set; } = 15;
    public int MinFacePx { get; set; } = 80;
    public double MinDetScore { get; set; } = 0.9;
    public double MaxSimilarity { get; set; } = 0.92;
    public int PollIntervalMs { get; set; } = 100;    // chờ frame mới
    public int SampleIntervalMs { get; set; } = 300;  // nghỉ giữa 2 lần thử → người dùng kịp xoay đầu
}

public sealed class StorageOptions
{
    public string DataDir { get; set; } = "data";
    public int SnapshotRetentionDays { get; set; } = 30;
}

public sealed class WebOptions
{
    public string Host { get; set; } = "127.0.0.1";
    public int Port { get; set; } = 8000;
    public string? AdminPassword { get; set; }
}
