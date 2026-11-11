using System.Globalization;
using System.Text;
using CameraAI.Application.Abstractions;
using CameraAI.Domain.Faces;
using CameraAI.Infrastructure.Video;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Vision;

/// <summary>Vẽ khung + nhãn, cắt ảnh, encode/decode JPEG — tương đương imaging.py.</summary>
internal sealed class OpenCvImaging : IFrameImaging
{
    private static readonly Dictionary<LabelColor, Scalar> Colors = new() // thứ tự BGR của OpenCV
    {
        [LabelColor.Green] = new(80, 200, 80),
        [LabelColor.Yellow] = new(0, 200, 255),
        [LabelColor.Red] = new(60, 60, 230),
        [LabelColor.Gray] = new(160, 160, 160),
    };

    public byte[] EncodeAnnotated(IVideoFrame frame, IReadOnlyList<FaceLabel> labels)
    {
        using var canvas = MatFrame.Of(frame).Clone(); // không vẽ lên frame gốc (còn dùng để cắt ảnh lưu)
        foreach (var label in labels) DrawBox(canvas, label);
        return Encode(canvas, quality: 75);
    }

    public byte[] EncodeCrop(IVideoFrame frame, FaceBox box, double scale)
    {
        using var crop = ImageOps.CropAround(MatFrame.Of(frame), box, scale);
        using var small = ImageOps.FitWithin(crop);
        return Encode(small, quality: 85);
    }

    public IVideoFrame? Decode(byte[] data)
    {
        if (data.Length == 0) return null;
        var mat = Cv2.ImDecode(data, ImreadModes.Color);
        if (!mat.Empty()) return new MatFrame(0, mat);
        mat.Dispose();
        return null;
    }

    private static void DrawBox(Mat canvas, FaceLabel label)
    {
        var color = Colors[label.Color];
        var b = label.Box;
        Cv2.Rectangle(canvas, new Rect(b.X, b.Y, b.Width, b.Height), color, 2);
        var text = AsciiLabel(label.Text);
        if (text.Length == 0) return;

        var size = Cv2.GetTextSize(text, HersheyFonts.HersheySimplex, 0.6, 1, out int baseline);
        int top = Math.Max(0, b.Y - size.Height - baseline - 6);
        Cv2.Rectangle(canvas, new Rect(b.X, top, size.Width + 8, size.Height + baseline + 6), color, -1);
        Cv2.PutText(canvas, text, new Point(b.X + 4, top + size.Height + 3), HersheyFonts.HersheySimplex, 0.6,
            Scalar.Black, 1, LineTypes.AntiAlias);
    }

    /// <summary>Font Hershey của OpenCV chỉ có ASCII → bỏ dấu tiếng Việt: "Nghi giả mạo" → "Nghi gia mao".</summary>
    internal static string AsciiLabel(string text)
    {
        var decomposed = text.Replace('đ', 'd').Replace('Đ', 'D').Normalize(NormalizationForm.FormD);
        var sb = new StringBuilder(decomposed.Length);
        foreach (var c in decomposed)
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark && c < 128) sb.Append(c);
        return sb.ToString();
    }

    private static byte[] Encode(Mat image, int quality)
    {
        Cv2.ImEncode(".jpg", image, out var buffer, new ImageEncodingParam(ImwriteFlags.JpegQuality, quality));
        return buffer;
    }
}
