using CameraAI.Domain.Faces;

namespace CameraAI.Domain.Attendance;

public enum LivenessMode { Off, Opportunistic, Required }

/// <summary>
/// STRATEGY PATTERN: mỗi chế độ chống giả là một class riêng, cùng một interface.
/// Bản Python viết <c>if mode == "required" ... elif mode == "opportunistic" ...</c> rải ở nhiều chỗ;
/// ở đây AttendanceDecider chỉ gọi <c>policy.CountsTowardVote(obs)</c> — thêm chế độ mới = thêm class mới,
/// KHÔNG sửa Decider (Open/Closed Principle).
/// </summary>
public interface ILivenessPolicy
{
    LivenessMode Mode { get; }

    /// <summary>Frame này có được bỏ phiếu chấm công không.</summary>
    bool CountsTowardVote(FaceObservation obs);

    /// <summary>Chế độ này có ghi "giả mạo" không.</summary>
    bool CanFlagSpoof { get; }
}

/// <summary>Tắt chống giả: mọi frame dùng được đều bỏ phiếu.</summary>
public sealed class LivenessOffPolicy : ILivenessPolicy
{
    public LivenessMode Mode => LivenessMode.Off;
    public bool CanFlagSpoof => false;
    public bool CountsTowardVote(FaceObservation obs) => true;
}

/// <summary>Camera trần: frame "chưa kiểm" (null) vẫn bỏ phiếu; chỉ loại frame đã kiểm là giả.</summary>
public sealed class LivenessOpportunisticPolicy : ILivenessPolicy
{
    public LivenessMode Mode => LivenessMode.Opportunistic;
    public bool CanFlagSpoof => true;
    public bool CountsTowardVote(FaceObservation obs) => obs.IsReal is not false;
}

/// <summary>Chặt nhất: chỉ frame đã kiểm là thật mới bỏ phiếu.</summary>
public sealed class LivenessRequiredPolicy : ILivenessPolicy
{
    public LivenessMode Mode => LivenessMode.Required;
    public bool CanFlagSpoof => true;
    public bool CountsTowardVote(FaceObservation obs) => obs.IsReal is true;
}

/// <summary>
/// FACTORY METHOD: chọn class cụ thể theo giá trị lúc runtime (đọc từ config) thay vì <c>new</c> rải rác.
/// Đây là chỗ DUY NHẤT còn switch trên mode.
/// </summary>
public static class LivenessPolicyFactory
{
    public static ILivenessPolicy Create(LivenessMode mode) => mode switch
    {
        LivenessMode.Off => new LivenessOffPolicy(),
        LivenessMode.Opportunistic => new LivenessOpportunisticPolicy(),
        LivenessMode.Required => new LivenessRequiredPolicy(),
        _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Chế độ chống giả không hợp lệ"),
    };
}
