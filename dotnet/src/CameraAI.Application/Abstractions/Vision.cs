using CameraAI.Domain.Faces;

namespace CameraAI.Application.Abstractions;

// ============================================================================================
//  PORT = interface do Application KHAI BÁO, Infrastructure CÀI ĐẶT (OpenCvSharp + ONNX Runtime).
//  Application không biết OpenCV tồn tại: nó chỉ thấy IVideoFrame, IFaceAnalyzer...
//  Đây là Dependency Inversion: module cấp cao (pipeline) định nghĩa "tôi cần gì", module cấp thấp tuân theo.
// ============================================================================================

/// <summary>
/// Một frame ảnh — "hộp đen" với Application. Bên Infrastructure nó bọc một <c>Mat</c> của OpenCV
/// (bộ nhớ native, vài MB) nên PHẢI Dispose: luôn dùng <c>using var frame = ...</c>.
/// </summary>
public interface IVideoFrame : IDisposable
{
    long Id { get; }
    int Width { get; }
    int Height { get; }
}

/// <summary>Nguồn video (webcam / RTSP / file). Thread nền đọc liên tục, chỉ giữ frame mới nhất.</summary>
public interface IVideoSource
{
    /// <summary>
    /// Frame mới hơn <paramref name="afterFrameId"/> hoặc null nếu chưa có. Trả về BẢN SAO —
    /// người gọi sở hữu và phải Dispose (nhờ vậy pipeline và đăng ký mẫu đọc song song không giẫm nhau).
    /// </summary>
    IVideoFrame? TryGetLatest(long afterFrameId);

    CameraStatus Status { get; }
}

public sealed record CameraStatus(
    bool Connected, bool Ended, string Source, int Width, int Height, double Fps, string? LastError);

/// <summary>
/// Phát hiện + chống giả + tính embedding trên một frame. KHÔNG so khớp gallery — việc đó là của Domain
/// (FaceGallery), pipeline sẽ tự gọi. Nhờ vậy Infrastructure chỉ lo "ảnh → con số".
///
/// Đối tượng DNN của OpenCV không an toàn khi dùng chung giữa các thread → mỗi nơi dùng (pipeline,
/// đăng ký mẫu) tạo analyzer riêng qua <see cref="IFaceAnalyzerFactory"/>.
/// </summary>
public interface IFaceAnalyzer : IDisposable
{
    /// <summary>Tên model nhận diện ("sface" / "arcface") — embedding được lưu theo tên này.</summary>
    string ModelName { get; }

    IReadOnlyList<FaceObservation> Analyze(IVideoFrame frame, RecognitionGate gate);

    /// <summary>Embedding của mặt lớn nhất (tính lại embedding từ ảnh mẫu khi đổi model). null = không thấy mặt.</summary>
    float[]? EmbedLargestFace(IVideoFrame frame);
}

/// <summary>
/// Factory dạng INTERFACE (khác LivenessPolicyFactory dạng static ở Domain). Vì sao? Theo 3 câu hỏi đã bàn:
/// (1) pipeline và enrollment — class nghiệp vụ — gọi nó; (2) test cần bản giả không nạp model thật;
/// (3) bản thật cần IOptions (đường dẫn model) và nạp file ONNX.
/// </summary>
public interface IFaceAnalyzerFactory
{
    IFaceAnalyzer Create();
}

public enum LabelColor { Green, Yellow, Red, Gray }

/// <summary>Một khung + nhãn cần vẽ lên frame trực tiếp.</summary>
public sealed record FaceLabel(FaceBox Box, string Text, LabelColor Color);

/// <summary>Xử lý ảnh: vẽ khung, cắt, encode/decode JPEG.</summary>
public interface IFrameImaging
{
    byte[] EncodeAnnotated(IVideoFrame frame, IReadOnlyList<FaceLabel> labels);

    /// <summary>Cắt vùng quanh khung mặt (phóng <paramref name="scale"/> lần), thu nhỏ cạnh dài ≤ 400 px, encode JPEG.</summary>
    byte[] EncodeCrop(IVideoFrame frame, FaceBox box, double scale);

    /// <summary>Đọc ảnh tải lên. null = không phải ảnh.</summary>
    IVideoFrame? Decode(byte[] data);
}
