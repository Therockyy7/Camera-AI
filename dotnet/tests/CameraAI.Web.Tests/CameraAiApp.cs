using CameraAI.Application.Abstractions;
using CameraAI.Domain.Faces;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace CameraAI.Web.Tests;

/// <summary>
/// Dựng NGUYÊN ứng dụng thật (Program.cs, DI, middleware, endpoint, SQLite) trong bộ nhớ, gọi bằng HttpClient thật.
/// Chỉ thay 2 thứ: model AI (bản giả, không cần file .onnx) và nguồn video (file không tồn tại → không mở webcam).
/// </summary>
public sealed class CameraAiApp : WebApplicationFactory<Program>
{
    public string DataDir { get; } = Path.Combine(Path.GetTempPath(), "cameraai-web-" + Guid.NewGuid().ToString("N")[..8]);
    public StubAnalyzer Analyzer { get; } = new();
    private readonly Dictionary<string, string?> _settings;

    public CameraAiApp(Dictionary<string, string?>? settings = null) => _settings = settings ?? [];

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Development");
        builder.ConfigureAppConfiguration((_, config) => config.AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["CameraAI:Storage:DataDir"] = DataDir,
            ["CameraAI:Camera:Source"] = Path.Combine(DataDir, "khong-co-video.mp4"),
        }.Concat(_settings)));
        // ConfigureTestServices chạy SAU Program.cs → đăng ký sau ghi đè đăng ký trước (GetRequiredService lấy cái cuối).
        builder.ConfigureTestServices(services => services.AddSingleton<IFaceAnalyzerFactory>(new StubAnalyzerFactory(Analyzer)));
    }

    public override async ValueTask DisposeAsync()
    {
        await base.DisposeAsync();
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        if (Directory.Exists(DataDir)) Directory.Delete(DataDir, recursive: true);
    }
}

public sealed class StubAnalyzer : IFaceAnalyzer
{
    public List<FaceObservation> Faces { get; } = [];
    public string ModelName => "sface";
    public IReadOnlyList<FaceObservation> Analyze(IVideoFrame frame, RecognitionGate gate) => Faces;
    public float[]? EmbedLargestFace(IVideoFrame frame) => null;
    public void Dispose() { }
}

public sealed class StubAnalyzerFactory(StubAnalyzer analyzer) : IFaceAnalyzerFactory
{
    public IFaceAnalyzer Create() => analyzer;
}
