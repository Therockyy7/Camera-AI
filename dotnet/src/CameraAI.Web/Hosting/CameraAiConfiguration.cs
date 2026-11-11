using System.Net;
using CameraAI.Application.Common;
using CameraAI.Domain.Attendance;
using Microsoft.Extensions.Options;

namespace CameraAI.Web.Hosting;

/// <summary>
/// Nguồn cấu hình, theo thứ tự ưu tiên tăng dần (cái sau ghi đè cái trước):
///   appsettings.json → appsettings.{Môi trường}.json → biến môi trường kiểu .NET (CameraAI__Camera__Source)
///   → biến môi trường KIỂU CŨ của bản Python (CAMERA_SOURCE, ADMIN_PASSWORD...) → tham số dòng lệnh.
/// Nhờ vậy file .env / docker-compose của bản Python dùng lại được nguyên vẹn.
/// </summary>
public static class CameraAiConfiguration
{
    private const string P = CameraAiOptions.Section;

    /// <summary>Biến môi trường của bản Python → khoá cấu hình .NET.</summary>
    private static readonly Dictionary<string, string> LegacyEnvironment = new()
    {
        ["CAMERA_SOURCE"] = $"{P}:Camera:Source",
        ["CAMERA_ID"] = $"{P}:Camera:Id",
        ["CAMERAAI_HOST"] = $"{P}:Web:Host",
        ["CAMERAAI_PORT"] = $"{P}:Web:Port",
        ["ADMIN_PASSWORD"] = $"{P}:Web:AdminPassword",
        ["CAMERAAI_DATA_DIR"] = $"{P}:Storage:DataDir",
        ["LIVENESS_MODE"] = $"{P}:Liveness:Mode",
        ["RECOGNITION_MODEL"] = $"{P}:Recognition:Model",
    };

    /// <summary>Tham số dòng lệnh giống bản Python: <c>dotnet run -- --source 1</c>.</summary>
    private static readonly Dictionary<string, string> Switches = new()
    {
        ["--source"] = $"{P}:Camera:Source",
        ["--model"] = $"{P}:Recognition:Model",
    };

    public static void AddCameraAiSources(this ConfigurationManager config, string[] args)
    {
        var legacy = LegacyEnvironment
            .Select(p => (Key: p.Value, Value: Environment.GetEnvironmentVariable(p.Key)))
            .Where(p => !string.IsNullOrEmpty(p.Value))
            .ToDictionary(p => p.Key, p => p.Value);
        if (args.Contains("--no-liveness")) legacy[$"{P}:Liveness:Mode"] = nameof(LivenessMode.Off); // cờ không có giá trị
        config.AddInMemoryCollection(legacy);
        config.AddCommandLine(args.Where(a => a != "--no-liveness").ToArray(), Switches);
    }

    /// <summary>
    /// OPTIONS PATTERN: bind mục "CameraAI" → <see cref="CameraAiOptions"/>, đổi đường dẫn tương đối thành tuyệt đối,
    /// và KIỂM TRA ngay lúc khởi động (ValidateOnStart) — cấu hình sai thì app không chạy, thay vì chạy rồi hỏng giữa chừng.
    /// </summary>
    public static IServiceCollection AddCameraAiOptions(this IServiceCollection services, IConfiguration config)
    {
        services.AddOptions<CameraAiOptions>()
            .Bind(config.GetSection(P))
            .PostConfigure<IHostEnvironment>((o, env) =>
            {
                // Đường dẫn tính từ thư mục project Web (content root), không phụ thuộc thư mục đang đứng khi chạy.
                o.ModelsDir = Path.GetFullPath(o.ModelsDir, env.ContentRootPath);
                o.Storage.DataDir = Path.GetFullPath(o.Storage.DataDir, env.ContentRootPath);
            })
            .Validate(o => IsLoopback(o.Web.Host) || !string.IsNullOrEmpty(o.Web.AdminPassword),
                "Nghe trên địa chỉ máy khác truy cập được (Web:Host) thì bắt buộc đặt ADMIN_PASSWORD")
            .Validate(o => o.Recognition.Model is "sface" or "arcface",
                "Recognition:Model phải là sface hoặc arcface")
            .Validate(o => o.Processing.Fps > 0 && o.Attendance.ConfirmFrames <= o.Attendance.WindowFrames,
                "Processing:Fps phải > 0 và Attendance:ConfirmFrames không được lớn hơn WindowFrames")
            .ValidateOnStart();
        return services;
    }

    public static bool IsLoopback(string host) =>
        host is "localhost" || (IPAddress.TryParse(host, out var ip) && IPAddress.IsLoopback(ip));
}

/// <summary>Đọc nhanh mục Web trước khi Build (để biết nghe cổng nào).</summary>
public static class WebOptionsReader
{
    public static WebOptions Read(IConfiguration config) =>
        config.GetSection($"{CameraAiOptions.Section}:Web").Get<WebOptions>() ?? new WebOptions();
}
