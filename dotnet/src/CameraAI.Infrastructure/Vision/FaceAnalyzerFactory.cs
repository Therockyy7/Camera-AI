using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Domain.Attendance;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CameraAI.Infrastructure.Vision;

/// <summary>
/// Cài đặt <see cref="IFaceAnalyzerFactory"/>: mỗi lần Create() nạp một bộ model MỚI (detector + họ nhận diện +
/// chống giả). Đối tượng DNN không an toàn khi chia sẻ giữa các thread → pipeline và đăng ký mẫu mỗi bên một bộ.
///
/// Họ model (<see cref="IRecognitionModelFamily"/>) được DI chọn theo config bằng KEYED SERVICES (.NET 8):
///   AddKeyedSingleton&lt;IRecognitionModelFamily, SFaceFamily&gt;("sface") … rồi lấy theo khoá "sface"/"arcface".
/// </summary>
internal sealed class FaceAnalyzerFactory(
    IRecognitionModelFamily family, IOptions<CameraAiOptions> options, ILoggerFactory loggers) : IFaceAnalyzerFactory
{
    public IFaceAnalyzer Create()
    {
        var o = options.Value;
        var dir = Path.GetFullPath(o.ModelsDir);
        bool withLiveness = o.Liveness.Mode != LivenessMode.Off;
        ModelFiles.Require(dir, withLiveness
            ? [ModelFiles.Detector, family.ModelFile, ModelFiles.Liveness]
            : [ModelFiles.Detector, family.ModelFile]);

        var analyzer = new OpenCvFaceAnalyzer(
            new YuNetDetector(Path.Combine(dir, ModelFiles.Detector), (float)o.Recognition.DetectScore, o.Processing.DetectWidth),
            family.CreateAligner(),                      // ┐ cả bộ lấy từ CÙNG một họ (Abstract Factory)
            family.CreateEmbedder(dir),                  // ┘
            withLiveness ? new MiniFasLiveness(Path.Combine(dir, ModelFiles.Liveness)) : null,
            o.Liveness.ToGate(),
            family.Name);
        return new TimingFaceAnalyzer(analyzer, loggers.CreateLogger<TimingFaceAnalyzer>()); // bọc Decorator
    }
}
