using CameraAI.Domain.Faces;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Vision;

/// <summary>Các phép xử lý ảnh nhỏ dùng chung (thuần kỹ thuật, không có nghiệp vụ).</summary>
internal static class ImageOps
{
    /// <summary>
    /// Toạ độ 5 điểm (mắt phải, mắt trái, mũi, khoé miệng phải, khoé miệng trái) trên ảnh 112×112 — template
    /// của ArcFace; OpenCV cũng dùng đúng template này trong FaceRecognizerSF.alignCrop cho SFace.
    /// </summary>
    public static readonly Point2f[] ArcFaceTemplate =
    [
        new(38.2946f, 51.6963f), new(73.5318f, 51.5014f), new(56.0252f, 71.7366f),
        new(41.5493f, 92.3655f), new(70.7299f, 92.2041f),
    ];

    /// <summary>
    /// Phép đồng dạng (xoay + tỉ lệ + tịnh tiến) bình phương tối thiểu đưa <paramref name="src"/> về <paramref name="dst"/>.
    /// Dạng đóng cho 2D: với điểm đã trừ trọng tâm p, q:  a = Σ(p·q)/Σ|p|²,  b = Σ(p×q)/Σ|p|²
    /// → ma trận [[a, −b, tx], [b, a, ty]]. (Cho cùng kết quả với thuật toán Umeyama bên Python khi không có phản xạ.)
    /// </summary>
    public static double[] SimilarityTransform(IReadOnlyList<Point2f> src, IReadOnlyList<Point2f> dst)
    {
        int n = src.Count;
        double msx = src.Average(p => p.X), msy = src.Average(p => p.Y);
        double mdx = dst.Average(p => p.X), mdy = dst.Average(p => p.Y);
        double dot = 0, cross = 0, norm = 0;
        for (int i = 0; i < n; i++)
        {
            double px = src[i].X - msx, py = src[i].Y - msy, qx = dst[i].X - mdx, qy = dst[i].Y - mdy;
            dot += px * qx + py * qy;
            cross += px * qy - py * qx;
            norm += px * px + py * py;
        }
        double a = dot / norm, b = cross / norm;
        return [a, -b, mdx - (a * msx - b * msy), b, a, mdy - (b * msx + a * msy)];
    }

    /// <summary>Xoay-thu phóng khuôn mặt về 112×112 theo template — đầu vào chuẩn của SFace/ArcFace.</summary>
    public static Mat AlignTo112(Mat frame, IReadOnlyList<Point2f> landmarks)
    {
        using var m = Mat.FromPixelData(2, 3, MatType.CV_64FC1, SimilarityTransform(landmarks, ArcFaceTemplate));
        var aligned = new Mat();
        Cv2.WarpAffine(frame, aligned, m, new Size(112, 112), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.All(0));
        return aligned;
    }

    /// <summary>Ảnh BGR (OpenCV) → tensor [1, 3, H, W] kênh RGB, mỗi điểm ảnh (giá trị − mean) × scale.</summary>
    public static DenseTensor<float> ToRgbTensor(Mat bgr, float mean, float scale)
    {
        int h = bgr.Rows, w = bgr.Cols;
        var tensor = new DenseTensor<float>([1, 3, h, w]);
        using var continuous = bgr.IsContinuous() ? null : bgr.Clone();
        (continuous ?? bgr).GetArray(out Vec3b[] pixels);
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            var p = pixels[y * w + x]; // Item0 = B, Item1 = G, Item2 = R
            tensor[0, 0, y, x] = (p.Item2 - mean) * scale;
            tensor[0, 1, y, x] = (p.Item1 - mean) * scale;
            tensor[0, 2, y, x] = (p.Item0 - mean) * scale;
        }
        return tensor;
    }

    /// <summary>Độ nét = phương sai Laplacian trên mặt xám đưa về 112×112 (ít phụ thuộc khoảng cách).</summary>
    public static double Sharpness(Mat frame, FaceBox box)
    {
        var rect = Clamp(frame, box.X, box.Y, box.Right, box.Bottom);
        if (rect.Width <= 0 || rect.Height <= 0) return 0;
        using var roi = new Mat(frame, rect);
        using var gray = new Mat();
        Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
        using var small = new Mat();
        Cv2.Resize(gray, small, new Size(112, 112), 0, 0, gray.Width > 112 ? InterpolationFlags.Area : InterpolationFlags.Linear);
        using var lap = new Mat();
        Cv2.Laplacian(small, lap, MatType.CV_64F);
        Cv2.MeanStdDev(lap, out _, out var std);
        return std.Val0 * std.Val0;
    }

    /// <summary>Cắt vùng quanh khung mặt phóng <paramref name="scale"/> lần (để ảnh lưu thấy cả đầu, vai).</summary>
    public static Mat CropAround(Mat frame, FaceBox box, double scale)
    {
        double cx = box.X + box.Width / 2.0, cy = box.Y + box.Height / 2.0;
        double hw = box.Width * scale / 2, hh = box.Height * scale / 2;
        var rect = Clamp(frame, (int)(cx - hw), (int)(cy - hh), (int)(cx + hw), (int)(cy + hh));
        return rect.Width <= 0 || rect.Height <= 0 ? frame.Clone() : new Mat(frame, rect).Clone();
    }

    /// <summary>Thu nhỏ để cạnh dài ≤ <paramref name="maxSide"/> (ảnh lưu nhẹ).</summary>
    public static Mat FitWithin(Mat image, int maxSide = 400)
    {
        int longest = Math.Max(image.Width, image.Height);
        if (longest <= maxSide) return image.Clone();
        double s = (double)maxSide / longest;
        var resized = new Mat();
        Cv2.Resize(image, resized, new Size(Math.Max(1, (int)Math.Round(image.Width * s)), Math.Max(1, (int)Math.Round(image.Height * s))),
            0, 0, InterpolationFlags.Area);
        return resized;
    }

    private static Rect Clamp(Mat frame, int x1, int y1, int x2, int y2)
    {
        x1 = Math.Max(0, x1);
        y1 = Math.Max(0, y1);
        x2 = Math.Min(frame.Width, x2);
        y2 = Math.Min(frame.Height, y2);
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }
}
