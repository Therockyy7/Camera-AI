using CameraAI.Application.Abstractions;
using CameraAI.Application.Faces;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Faces;

namespace CameraAI.Application.Pipeline;

/// <summary>
/// Chọn chữ + màu cho khung mặt trên hình trực tiếp. Tách khỏi pipeline (SRP): pipeline lo luồng chạy,
/// class này lo "hiển thị gì" — test riêng được, đổi câu chữ không đụng vòng xử lý.
/// </summary>
public sealed class FaceLabeler(AttendanceDecider decider, EmployeeNames names)
{
    private static readonly Dictionary<RejectReason, string> ReasonText = new()
    {
        [RejectReason.Small] = "Lại gần hơn",
        [RejectReason.Edge] = "Lùi ra / vào giữa khung hình",
        [RejectReason.Blurry] = "Giữ yên",
        [RejectReason.Turned] = "Nhìn thẳng camera",
    };

    public FaceLabel For(Track track, FaceObservation obs)
    {
        var (text, color) = Describe(track, obs);
        return new FaceLabel(obs.Box, text, color);
    }

    private (string, LabelColor) Describe(Track track, FaceObservation obs)
    {
        if (obs.RejectReason is { } reason) return (ReasonText[reason], LabelColor.Gray);

        // Theo cửa sổ HIỆN TẠI, không theo cờ đã ghi: một lần bị nghi giả lúc mới bước vào
        // không được che trạng thái đã chấm công về sau.
        if (decider.LooksSpoof(track)) return ("Nghi giả mạo", LabelColor.Red);

        if (track.EmployeeId is { } confirmed)
        {
            var last = decider.LastRecorded(confirmed);
            var suffix = last is { } at ? $" - đã chấm {at:HH:mm}" : "";
            return ($"{NameOr(confirmed)}{suffix}", LabelColor.Green);
        }
        if (obs.IsReal is false) return ("Đang kiểm tra...", LabelColor.Yellow);
        if (obs.Match?.EmployeeId is { } candidate) return ($"{NameOr(candidate)}...", LabelColor.Yellow);
        if (track.Done.Contains(DecisionKind.Unknown)) return ("Người lạ", LabelColor.Red);
        return ("Đang nhận diện...", LabelColor.Yellow);
    }

    private string NameOr(int id) => names.NameOf(id) is { Length: > 0 } name ? name : "?";
}
