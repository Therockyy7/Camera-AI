using CameraAI.Domain.Faces;

namespace CameraAI.Application.Enrollment;

/// <summary>Lý do một lần thử lấy mẫu khuôn mặt không được lưu.</summary>
public enum EnrollmentIssue
{
    NoFace, ManyFaces, Small, Edge, Blurry, Turned, LowScore, Spoof, NotImage, NoCamera, Similar,
}

public static class EnrollmentIssueText
{
    private static readonly Dictionary<EnrollmentIssue, (string Code, string Text)> Map = new()
    {
        [EnrollmentIssue.NoFace] = ("no_face", "Không thấy khuôn mặt"),
        [EnrollmentIssue.ManyFaces] = ("many_faces", "Có nhiều hơn một khuôn mặt"),
        [EnrollmentIssue.Small] = ("small", "Mặt quá nhỏ — lại gần camera hơn"),
        [EnrollmentIssue.Edge] = ("edge", "Mặt bị cắt ở mép khung hình — lùi ra hoặc vào giữa"),
        [EnrollmentIssue.Blurry] = ("blurry", "Ảnh mờ — giữ yên"),
        [EnrollmentIssue.Turned] = ("turned", "Mặt quay nghiêng — nhìn thẳng camera"),
        [EnrollmentIssue.LowScore] = ("low_score", "Khuôn mặt không rõ"),
        [EnrollmentIssue.Spoof] = ("spoof", "Nghi mặt giả (ảnh / màn hình)"),
        [EnrollmentIssue.NotImage] = ("not_image", "Không đọc được file ảnh"),
        [EnrollmentIssue.NoCamera] = ("no_camera", "Chưa có hình từ camera"),
        [EnrollmentIssue.Similar] = ("similar", "Giống mẫu đã có — xoay đầu sang góc khác"),
    };

    public static string Code(this EnrollmentIssue issue) => Map[issue].Code;
    public static string Text(this EnrollmentIssue issue) => Map[issue].Text;

    public static EnrollmentIssue From(RejectReason reason) => reason switch
    {
        RejectReason.Small => EnrollmentIssue.Small,
        RejectReason.Edge => EnrollmentIssue.Edge,
        RejectReason.Blurry => EnrollmentIssue.Blurry,
        RejectReason.Turned => EnrollmentIssue.Turned,
        _ => throw new ArgumentOutOfRangeException(nameof(reason)),
    };
}

public sealed record IssueCount(string Code, int Count, string Text);

public sealed record CaptureResult(int Saved, int Attempts, IReadOnlyList<IssueCount> Reasons);
