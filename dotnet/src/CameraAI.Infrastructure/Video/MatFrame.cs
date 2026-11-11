using CameraAI.Application.Abstractions;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Video;

/// <summary>
/// Cài đặt <see cref="IVideoFrame"/>: bọc một <see cref="Mat"/> của OpenCV. Application chỉ thấy "IVideoFrame",
/// chỉ các class trong Infrastructure mới mở hộp lấy Mat ra (<see cref="Of"/>).
/// Mat giữ bộ nhớ NATIVE (C++) mà GC .NET không thấy → bắt buộc Dispose.
/// </summary>
internal sealed class MatFrame(long id, Mat mat) : IVideoFrame
{
    public long Id => id;
    public int Width => mat.Width;
    public int Height => mat.Height;
    public Mat Mat => mat;

    public void Dispose() => mat.Dispose();

    public static Mat Of(IVideoFrame frame) =>
        frame is MatFrame m ? m.Mat : throw new ArgumentException("Frame không do Infrastructure tạo ra", nameof(frame));
}
