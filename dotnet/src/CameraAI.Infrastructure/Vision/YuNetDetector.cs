using CameraAI.Domain.Faces;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Vision;

/// <summary>Một mặt YuNet tìm thấy: khung + 5 điểm mốc (mắt phải, mắt trái, mũi, khoé miệng phải, trái) theo frame gốc.</summary>
internal sealed record DetectedFace(FaceBox Box, Point2f[] Landmarks, float Score);

/// <summary>
/// Phát hiện mặt bằng YuNet (OpenCV FaceDetectorYN). Thu nhỏ frame về <c>detectWidth</c> cho nhanh rồi nhân toạ độ ngược lại.
///
/// OpenCV bắt ảnh đúng kích thước khai báo khi tạo detector, và OpenCvSharp chưa có SetInputSize
/// → kích thước đổi thì tạo lại detector (camera kích thước cố định nên thực tế chỉ tạo 1 lần; model chỉ ~300 KB).
/// </summary>
internal sealed class YuNetDetector(string modelPath, float scoreThreshold, int detectWidth) : IDisposable
{
    private FaceDetectorYN? _net;
    private Size _size;

    public IReadOnlyList<DetectedFace> Detect(Mat frame)
    {
        double scale = 1;
        Mat? resized = null;
        if (detectWidth > 0 && frame.Width > detectWidth)
        {
            scale = (double)detectWidth / frame.Width;
            resized = new Mat();
            Cv2.Resize(frame, resized, new Size(detectWidth, Math.Max(1, (int)Math.Round(frame.Height * scale))),
                0, 0, InterpolationFlags.Area);
        }
        try
        {
            var image = resized ?? frame;
            if (_net is null || _size != image.Size())
            {
                _net?.Dispose();
                _net = FaceDetectorYN.Create(modelPath, "", image.Size(), scoreThreshold, 0.3f, 5000);
                _size = image.Size();
            }
            using var rows = new Mat();
            _net.Detect(image, rows);

            int count = rows.Rows; // mỗi lần đọc thuộc tính Mat là một lời gọi sang C++ → đọc 1 lần
            if (count == 0) return [];
            rows.GetArray(out float[] data); // chép cả ma trận N×15 ra mảng .NET một lần

            var faces = new List<DetectedFace>(count);
            for (int i = 0; i < count; i++)
            {
                // Mỗi dòng 15 số: x, y, w, h, 5 cặp (x, y) điểm mốc, điểm tin cậy
                float V(int col) => data[i * 15 + col];
                var box = new FaceBox(R(V(0) / scale), R(V(1) / scale), R(V(2) / scale), R(V(3) / scale));
                var landmarks = Enumerable.Range(0, 5)
                    .Select(k => new Point2f((float)(V(4 + 2 * k) / scale), (float)(V(5 + 2 * k) / scale))).ToArray();
                faces.Add(new DetectedFace(box, landmarks, V(14)));
            }
            return faces;
        }
        finally
        {
            resized?.Dispose();
        }
    }

    private static int R(double v) => (int)Math.Round(v);

    public void Dispose() => _net?.Dispose();
}
