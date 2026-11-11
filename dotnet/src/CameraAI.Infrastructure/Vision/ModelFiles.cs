namespace CameraAI.Infrastructure.Vision;

/// <summary>Tên file model trong thư mục models/ (dùng chung với bản Python — tải bằng cameraai.model_store).</summary>
internal static class ModelFiles
{
    public const string Detector = "face_detection_yunet_2023mar.onnx";
    public const string SFace = "face_recognition_sface_2021dec.onnx";
    public const string Liveness = "liveness_minifasnetv2_128.onnx";
    public const string ArcFace = "w600k_r50.onnx";

    /// <summary>Báo lỗi RÕ RÀNG ngay lúc nạp, thay vì lỗi native khó hiểu của OpenCV/ONNX Runtime.</summary>
    public static void Require(string modelsDir, IEnumerable<string> files)
    {
        var missing = files.Where(f => !File.Exists(Path.Combine(modelsDir, f))).ToList();
        if (missing.Count == 0) return;
        throw new FileNotFoundException(
            $"Thiếu model trong {modelsDir}: {string.Join(", ", missing)}. " +
            @"Tải bằng bản Python: .venv\Scripts\python.exe -m cameraai.model_store download" +
            (missing.Contains(ArcFace) ? " --arcface" : ""));
    }
}
