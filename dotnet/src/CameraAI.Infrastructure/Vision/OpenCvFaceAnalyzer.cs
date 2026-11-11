using CameraAI.Application.Abstractions;
using CameraAI.Domain.Faces;
using CameraAI.Infrastructure.Video;

namespace CameraAI.Infrastructure.Vision;

/// <summary>
/// Cài đặt <see cref="IFaceAnalyzer"/> — tương đương face_engine.py: phát hiện → cổng chất lượng (Domain) →
/// chống giả (nếu mặt đủ to & gần thẳng) → căn mặt + embedding. KHÔNG so khớp gallery (việc của Domain).
///
/// Quy tắc "mặt thế nào thì dùng được" KHÔNG nằm ở đây: lớp này chỉ ĐO (khung, độ quay, độ nét) rồi hỏi
/// RecognitionGate / LivenessGate của Domain. Infrastructure lo "đo", Domain lo "quyết".
/// </summary>
internal sealed class OpenCvFaceAnalyzer(
    YuNetDetector detector, IFaceAligner aligner, IFaceEmbedder embedder,
    MiniFasLiveness? liveness, LivenessGate livenessGate, string modelName) : IFaceAnalyzer
{
    public string ModelName => modelName;

    public IReadOnlyList<FaceObservation> Analyze(IVideoFrame frame, RecognitionGate gate)
    {
        var mat = MatFrame.Of(frame);
        var observations = new List<FaceObservation>();
        foreach (var face in detector.Detect(mat))
        {
            var box = face.Box;
            double turn = RecognitionGate.TurnOffset(face.Landmarks[0].X, face.Landmarks[1].X, face.Landmarks[2].X);
            double sharpness = ImageOps.Sharpness(mat, box);
            var observation = new FaceObservation
            {
                Box = box,
                DetScore = face.Score,
                Sharpness = sharpness,
                Turn = turn,
                RejectReason = gate.Evaluate(box, turn, sharpness, mat.Width, mat.Height),
            };
            if (observation.Usable)
            {
                double? real = liveness is not null && livenessGate.CanCheck(box, turn) ? liveness.Score(mat, box) : null;
                using var aligned = aligner.Align(mat, face.Landmarks);
                observation = observation with
                {
                    Liveness = real,
                    IsReal = real is { } p ? livenessGate.IsReal(p) : null,
                    Embedding = embedder.Embed(aligned),
                };
            }
            observations.Add(observation);
        }
        return observations;
    }

    public float[]? EmbedLargestFace(IVideoFrame frame)
    {
        var mat = MatFrame.Of(frame);
        var largest = detector.Detect(mat).MaxBy(f => f.Box.Width);
        if (largest is null) return null;
        using var aligned = aligner.Align(mat, largest.Landmarks);
        return embedder.Embed(aligned);
    }

    public void Dispose()
    {
        detector.Dispose();
        embedder.Dispose();
        liveness?.Dispose();
    }
}
