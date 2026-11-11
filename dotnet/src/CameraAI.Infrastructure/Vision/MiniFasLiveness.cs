using CameraAI.Domain.Faces;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Vision;

/// <summary>
/// Chống giả mạo bằng MiniFASNetV2 (ONNX) — tương đương liveness.py. Tiền xử lý bám đúng repo gốc của model:
/// crop vuông ×1,5 quanh mặt (thiếu biên thì pad phản chiếu), RGB, letterbox 128×128, chia 255, CHW.
/// Đầu ra 2 logit [thật, giả] → xác suất mặt thật = sigmoid(thật − giả).
/// </summary>
internal sealed class MiniFasLiveness : IDisposable
{
    private const int InputSize = 128;
    private const double BoxExpansion = 1.5;
    private readonly InferenceSession _session;
    private readonly string _input;

    public MiniFasLiveness(string modelPath)
    {
        _session = new InferenceSession(modelPath, OnnxSessions.Options(threads: 2));
        _input = _session.InputMetadata.Keys.First();
    }

    /// <summary>Xác suất khuôn mặt trong <paramref name="box"/> là mặt thật (0..1).</summary>
    public double Score(Mat frameBgr, FaceBox box)
    {
        using var square = CropSquare(frameBgr, box);
        using var input = Letterbox(square);
        var tensor = ImageOps.ToRgbTensor(input, mean: 0f, scale: 1 / 255f);
        using var outputs = _session.Run([NamedOnnxValue.CreateFromTensor(_input, tensor)]);
        var logits = outputs[0].AsEnumerable<float>().ToArray();
        return RealProbability(logits[0], logits[1]);
    }

    internal static Mat CropSquare(Mat image, FaceBox box)
    {
        int size = (int)(Math.Max(box.Width, box.Height) * BoxExpansion);
        if (box.Width <= 0 || box.Height <= 0 || size <= 0) throw new ArgumentException("Khung mặt không hợp lệ");
        int x1 = (int)(box.X + box.Width / 2.0 - size / 2.0);
        int y1 = (int)(box.Y + box.Height / 2.0 - size / 2.0);
        int cx1 = Math.Max(0, x1), cy1 = Math.Max(0, y1);
        int cx2 = Math.Min(image.Width, x1 + size), cy2 = Math.Min(image.Height, y1 + size);
        if (cx2 <= cx1 || cy2 <= cy1) throw new ArgumentException("Khung mặt nằm ngoài ảnh");

        using var crop = new Mat(image, new Rect(cx1, cy1, cx2 - cx1, cy2 - cy1));
        var padded = new Mat();
        Cv2.CopyMakeBorder(crop, padded, cy1 - y1, y1 + size - cy2, cx1 - x1, x1 + size - cx2, BorderTypes.Reflect101);
        return padded;
    }

    internal static Mat Letterbox(Mat face)
    {
        double ratio = (double)InputSize / Math.Max(face.Width, face.Height);
        int w = (int)(face.Width * ratio), h = (int)(face.Height * ratio);
        using var resized = new Mat();
        Cv2.Resize(face, resized, new Size(w, h), 0, 0, ratio > 1 ? InterpolationFlags.Lanczos4 : InterpolationFlags.Area);
        int dh = InputSize - h, dw = InputSize - w;
        var output = new Mat();
        Cv2.CopyMakeBorder(resized, output, dh / 2, dh - dh / 2, dw / 2, dw - dw / 2, BorderTypes.Reflect101);
        return output;
    }

    /// <summary>sigmoid(thật − giả), viết ổn định số học (không tràn exp với số âm lớn).</summary>
    internal static double RealProbability(double realLogit, double spoofLogit)
    {
        double diff = realLogit - spoofLogit;
        if (diff >= 0) return 1 / (1 + Math.Exp(-diff));
        double e = Math.Exp(diff);
        return e / (1 + e);
    }

    public void Dispose() => _session.Dispose();
}
