using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Vision;

// ============================================================================================
//  ABSTRACT FACTORY — tạo trọn bộ một HỌ sản phẩm đi cùng nhau.
//
//  Muốn ra một embedding đúng, cần HAI thứ khớp nhau:
//    (1) IFaceAligner : căn mặt về ảnh chuẩn (kích thước + template điểm mốc mà model được huấn luyện)
//    (2) IFaceEmbedder: chạy model với ĐÚNG cách chuẩn hoá điểm ảnh của model đó
//  Lắp lệch (aligner 112×112 của model này + embedder chờ 160×160 của model khác, hoặc chuẩn hoá sai)
//  thì vẫn chạy, không lỗi, nhưng embedding vô nghĩa → nhận nhầm người. Abstract Factory đảm bảo luôn lấy
//  cả bộ từ CÙNG một họ.
//
//  Hai họ hiện có tình cờ dùng chung kiểu căn 112×112 (ArcFace template), khác nhau ở chuẩn hoá + file model.
//  Thêm model mới (vd FaceNet 160×160) = thêm một họ mới, không sửa analyzer.
// ============================================================================================

internal interface IFaceAligner
{
    Mat Align(Mat frame, IReadOnlyList<Point2f> landmarks);
}

internal interface IFaceEmbedder : IDisposable
{
    float[] Embed(Mat aligned);
}

/// <summary>Abstract Factory: mỗi họ model tạo ra aligner + embedder tương thích với nhau.</summary>
internal interface IRecognitionModelFamily
{
    string Name { get; }
    string ModelFile { get; }
    IFaceAligner CreateAligner();
    IFaceEmbedder CreateEmbedder(string modelsDir);
}

/// <summary>SFace (OpenCV Zoo, Apache-2.0, dùng thương mại được). Ảnh RGB, giá trị điểm ảnh giữ nguyên 0..255.</summary>
internal sealed class SFaceFamily : IRecognitionModelFamily
{
    public string Name => "sface";
    public string ModelFile => ModelFiles.SFace;
    public IFaceAligner CreateAligner() => new ArcFaceTemplateAligner();
    public IFaceEmbedder CreateEmbedder(string modelsDir) =>
        new OnnxFaceEmbedder(Path.Combine(modelsDir, ModelFile), mean: 0f, scale: 1f, threads: 2);
}

/// <summary>ArcFace (InsightFace buffalo_l — CHỈ nghiên cứu, dùng thật cần mua license). Ảnh RGB, (x − 127.5) / 127.5.</summary>
internal sealed class ArcFaceFamily : IRecognitionModelFamily
{
    public string Name => "arcface";
    public string ModelFile => ModelFiles.ArcFace;
    public IFaceAligner CreateAligner() => new ArcFaceTemplateAligner();
    public IFaceEmbedder CreateEmbedder(string modelsDir) =>
        new OnnxFaceEmbedder(Path.Combine(modelsDir, ModelFile), mean: 127.5f, scale: 1 / 127.5f, threads: 4);
}

internal sealed class ArcFaceTemplateAligner : IFaceAligner
{
    public Mat Align(Mat frame, IReadOnlyList<Point2f> landmarks) => ImageOps.AlignTo112(frame, landmarks);
}

/// <summary>Chạy model nhận diện ONNX bằng ONNX Runtime: ảnh 112×112 → vector đặc trưng (128 hoặc 512 số).</summary>
internal sealed class OnnxFaceEmbedder : IFaceEmbedder
{
    private readonly InferenceSession _session;
    private readonly string _input;
    private readonly float _mean, _scale;

    public OnnxFaceEmbedder(string modelPath, float mean, float scale, int threads)
    {
        _session = new InferenceSession(modelPath, new SessionOptions { IntraOpNumThreads = threads });
        _input = _session.InputMetadata.Keys.First();
        (_mean, _scale) = (mean, scale);
    }

    public float[] Embed(Mat aligned)
    {
        var tensor = ImageOps.ToRgbTensor(aligned, _mean, _scale);
        using var outputs = _session.Run([NamedOnnxValue.CreateFromTensor(_input, tensor)]);
        return outputs[0].AsEnumerable<float>().ToArray();
    }

    public void Dispose() => _session.Dispose();
}
