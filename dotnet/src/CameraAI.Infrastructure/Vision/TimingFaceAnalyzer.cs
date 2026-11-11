using System.Diagnostics;
using CameraAI.Application.Abstractions;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.Logging;

namespace CameraAI.Infrastructure.Vision;

/// <summary>
/// DECORATOR (Performance Tracking): bọc một <see cref="IFaceAnalyzer"/> bất kỳ, đo thời gian mỗi lần Analyze và
/// cứ 200 frame ghi log trung bình một lần. Cùng interface với lớp bên trong → pipeline không biết mình đang
/// gọi qua lớp bọc. Muốn tắt đo? Factory bỏ lớp bọc đi, không sửa OpenCvFaceAnalyzer hay pipeline.
/// </summary>
internal sealed class TimingFaceAnalyzer(IFaceAnalyzer inner, ILogger<TimingFaceAnalyzer> log) : IFaceAnalyzer
{
    private const int ReportEvery = 200;
    private long _ticks;
    private int _calls, _faces;

    public string ModelName => inner.ModelName;

    public IReadOnlyList<FaceObservation> Analyze(IVideoFrame frame, RecognitionGate gate)
    {
        long started = Stopwatch.GetTimestamp();
        var result = inner.Analyze(frame, gate);       // ← việc thật do lớp bên trong làm
        _ticks += Stopwatch.GetTimestamp() - started;
        _faces += result.Count;
        if (++_calls == ReportEvery)
        {
            log.LogInformation("Phân tích ảnh ({Model}): trung bình {Ms:F1} ms/frame, {Faces:F1} mặt/frame",
                ModelName, Stopwatch.GetElapsedTime(0, _ticks).TotalMilliseconds / _calls, (double)_faces / _calls);
            (_ticks, _calls, _faces) = (0, 0, 0);
        }
        return result;
    }

    public float[]? EmbedLargestFace(IVideoFrame frame) => inner.EmbedLargestFace(frame);

    public void Dispose() => inner.Dispose();
}
