namespace CameraAI.Application.Common;

/// <summary>Tham số request sai (vd ngày kết thúc trước ngày bắt đầu) → tầng Web dịch thành HTTP 400.</summary>
public sealed class InvalidRequestException(string message) : Exception(message);

public static class DecisionKindExtensions
{
    /// <summary>"attendance" / "spoof" / "unknown" — đúng chuỗi giao diện web đang dùng.</summary>
    public static string ToApiString(this Domain.Attendance.DecisionKind kind) => kind.ToString().ToLowerInvariant();
}
