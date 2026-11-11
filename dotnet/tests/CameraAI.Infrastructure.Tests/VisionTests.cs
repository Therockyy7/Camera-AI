using CameraAI.Application;
using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Domain.Faces;
using CameraAI.Infrastructure.Video;
using CameraAI.Infrastructure.Vision;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OpenCvSharp;

namespace CameraAI.Infrastructure.Tests;

public class ImageOpsTests
{
    [Fact]
    public void Similarity_transform_of_template_onto_itself_is_identity()
    {
        var m = ImageOps.SimilarityTransform(ImageOps.ArcFaceTemplate, ImageOps.ArcFaceTemplate);
        double[] identity = [1, 0, 0, 0, 1, 0];
        for (int i = 0; i < 6; i++) Assert.Equal(identity[i], m[i], precision: 5);
    }

    [Fact]
    public void Similarity_transform_recovers_rotation_scale_and_shift()
    {
        // Tạo điểm "trên ảnh" bằng cách xoay 30°, phóng 2 lần, dịch (100, 50) template → phép ngược phải đưa về template.
        double angle = Math.PI / 6, s = 2;
        var src = ImageOps.ArcFaceTemplate.Select(p => new Point2f(
            (float)(s * (Math.Cos(angle) * p.X - Math.Sin(angle) * p.Y) + 100),
            (float)(s * (Math.Sin(angle) * p.X + Math.Cos(angle) * p.Y) + 50))).ToArray();
        var m = ImageOps.SimilarityTransform(src, ImageOps.ArcFaceTemplate);
        for (int i = 0; i < src.Length; i++)
        {
            double x = m[0] * src[i].X + m[1] * src[i].Y + m[2];
            double y = m[3] * src[i].X + m[4] * src[i].Y + m[5];
            Assert.Equal(ImageOps.ArcFaceTemplate[i].X, x, tolerance: 1e-3); // dung sai, không phải "làm tròn tới 3 số"
            Assert.Equal(ImageOps.ArcFaceTemplate[i].Y, y, tolerance: 1e-3);
        }
    }

    [Fact]
    public void Rgb_tensor_swaps_channels_and_normalizes()
    {
        using var bgr = new Mat(1, 1, MatType.CV_8UC3, new Scalar(10, 20, 255)); // B=10, G=20, R=255
        var t = ImageOps.ToRgbTensor(bgr, mean: 127.5f, scale: 1 / 127.5f);
        Assert.Equal(1f, t[0, 0, 0, 0], precision: 4);                 // kênh 0 = R
        Assert.Equal((20 - 127.5f) / 127.5f, t[0, 1, 0, 0], precision: 4);
        Assert.Equal((10 - 127.5f) / 127.5f, t[0, 2, 0, 0], precision: 4);
    }

    [Fact]
    public void Sharpness_is_zero_on_flat_image_and_high_on_texture()
    {
        using var flat = new Mat(200, 200, MatType.CV_8UC3, Scalar.All(128));
        using var noisy = new Mat(200, 200, MatType.CV_8UC3);
        Cv2.Randu(noisy, Scalar.All(0), Scalar.All(255));
        var box = new FaceBox(50, 50, 100, 100);
        Assert.Equal(0, ImageOps.Sharpness(flat, box), precision: 6);
        Assert.True(ImageOps.Sharpness(noisy, box) > 1000);
    }

    [Fact]
    public void Liveness_crop_is_square_with_reflected_padding_then_128px()
    {
        using var image = new Mat(100, 100, MatType.CV_8UC3, Scalar.All(50));
        using var square = MiniFasLiveness.CropSquare(image, new FaceBox(10, 20, 40, 60)); // ×1,5 → 90, tràn mép trên
        Assert.Equal((90, 90), (square.Width, square.Height));
        using var input = MiniFasLiveness.Letterbox(square);
        Assert.Equal((128, 128), (input.Width, input.Height));
    }

    [Theory]
    [InlineData(0, 0, 0.5)]
    [InlineData(5, -5, 0.9999546)]
    [InlineData(-800, 0, 0)]         // không tràn số với logit âm lớn
    public void Real_probability_is_stable_sigmoid(double real, double spoof, double expected)
    {
        Assert.Equal(expected, MiniFasLiveness.RealProbability(real, spoof), precision: 6);
    }

    [Theory]
    [InlineData("Nghi giả mạo", "Nghi gia mao")]
    [InlineData("Đặng Văn Đức - đã chấm 08:00", "Dang Van Duc - da cham 08:00")]
    [InlineData("Lùi ra / vào giữa khung hình", "Lui ra / vao giua khung hinh")]
    public void Labels_are_stripped_to_ascii_for_hershey_font(string text, string expected)
    {
        Assert.Equal(expected, OpenCvImaging.AsciiLabel(text));
    }
}

public class ImagingAndSourceTests
{
    [Fact]
    public void Decode_rejects_non_images_and_round_trips_jpeg()
    {
        var imaging = new OpenCvImaging();
        Assert.Null(imaging.Decode([1, 2, 3]));
        Assert.Null(imaging.Decode([]));

        using var mat = new Mat(240, 320, MatType.CV_8UC3, new Scalar(0, 128, 255));
        using var frame = new MatFrame(1, mat.Clone());
        var jpeg = imaging.EncodeAnnotated(frame, [new FaceLabel(new FaceBox(50, 60, 80, 80), "Người lạ", LabelColor.Red)]);
        Assert.Equal([0xFF, 0xD8], jpeg[..2]);                     // chữ ký JPEG
        using var back = imaging.Decode(jpeg)!;
        Assert.Equal((320, 240), (back.Width, back.Height));

        var crop = imaging.EncodeCrop(frame, new FaceBox(100, 80, 100, 100), scale: 2.0);
        using var cropped = imaging.Decode(crop)!;
        Assert.Equal((200, 200), (cropped.Width, cropped.Height));
    }

    [Theory]
    [InlineData("0", "Webcam")]
    [InlineData("rtsp://admin:pw@192.168.1.64:554/Streaming/Channels/102", "Stream")]
    [InlineData(@"D:\clip.mp4", "File")]
    public void Parses_video_source(string source, string kind) => // SourceKind là internal → test public nhận chuỗi
        Assert.Equal(Enum.Parse<SourceKind>(kind), VideoSourceSpec.Parse(source).Kind);

    [Fact]
    public void Masks_password_in_stream_url()
    {
        Assert.Equal("rtsp://admin:***@192.168.1.64:554/Streaming/Channels/102",
            VideoSourceSpec.Mask("rtsp://admin:S3cret!@192.168.1.64:554/Streaming/Channels/102"));
        Assert.Equal("0", VideoSourceSpec.Mask("0"));
    }
}

public class VisionRegistrationTests
{
    private static ServiceProvider Build(Action<CameraAiOptions>? configure = null)
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.Configure<CameraAiOptions>(o =>
        {
            o.ModelsDir = ModelsDir;
            o.Storage.DataDir = Path.Combine(Path.GetTempPath(), "cameraai-vision-" + Guid.NewGuid().ToString("N")[..8]);
            configure?.Invoke(o);
        });
        services.AddApplication();
        services.AddInfrastructure();
        return services.BuildServiceProvider();
    }

    /// <summary>Thư mục models/ của repo (dùng chung với bản Python, không commit file .onnx).</summary>
    private static string ModelsDir => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../../models"));

    [Fact]
    public void Camera_is_one_object_behind_both_interfaces()
    {
        using var sp = Build();
        var source = sp.GetRequiredService<IVideoSource>();
        var hosted = sp.GetServices<IHostedService>().OfType<OpenCvVideoSource>().Single();
        Assert.Same(source, hosted); // nếu là 2 object thì webcam bị mở 2 lần
    }

    [Fact]
    public void Unknown_recognition_model_fails_with_clear_message()
    {
        using var sp = Build(o => o.Recognition.Model = "facenet");
        var ex = Assert.Throws<InvalidOperationException>(() => sp.GetRequiredService<IFaceAnalyzerFactory>());
        Assert.Contains("facenet", ex.Message);
    }

    [Fact]
    public void Missing_model_files_fail_with_download_hint()
    {
        using var sp = Build(o => o.ModelsDir = Path.GetTempPath());
        var ex = Assert.Throws<FileNotFoundException>(() => sp.GetRequiredService<IFaceAnalyzerFactory>().Create());
        Assert.Contains("model_store download", ex.Message);
    }

    [Fact]
    public void Real_models_load_and_find_no_face_on_blank_image()
    {
        // Test tích hợp cần model thật: bỏ qua nếu chưa tải (giống bản Python).
        if (!File.Exists(Path.Combine(ModelsDir, ModelFiles.SFace))) return;
        using var sp = Build();
        using var analyzer = sp.GetRequiredService<IFaceAnalyzerFactory>().Create();
        using var frame = new MatFrame(1, new Mat(480, 640, MatType.CV_8UC3, Scalar.All(127)));

        Assert.Equal("sface", analyzer.ModelName);
        Assert.IsType<TimingFaceAnalyzer>(analyzer);   // factory luôn bọc Decorator đo thời gian
        Assert.Empty(analyzer.Analyze(frame, new RecognitionGate()));
        Assert.Null(analyzer.EmbedLargestFace(frame));
    }
}
