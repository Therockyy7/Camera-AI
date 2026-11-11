using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Infrastructure.Files;
using CameraAI.Infrastructure.Persistence;
using CameraAI.Infrastructure.Video;
using CameraAI.Infrastructure.Vision;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace CameraAI.Infrastructure;

/// <summary>
/// Nối mỗi PORT của Application với bản cài đặt thật. Program.cs chỉ gọi <c>services.AddInfrastructure()</c>.
/// Đây là nơi duy nhất trong cả solution biết "IEmployeeRepository thật ra là EF Core".
/// </summary>
public static class DependencyInjection
{
    public static IServiceCollection AddInfrastructure(this IServiceCollection services) =>
        services.AddPersistence().AddVision();

    public static IServiceCollection AddVision(this IServiceCollection services)
    {
        // Camera: MỘT object đóng 2 vai — IVideoSource (Application đọc frame) + IHostedService (host tự Start/Stop).
        // Đăng ký class một lần rồi CHUYỂN TIẾP cả hai interface về nó; viết AddHostedService<OpenCvVideoSource>()
        // riêng sẽ tạo object THỨ HAI mở camera lần nữa (webcam bị khoá → lỗi).
        services.AddSingleton<OpenCvVideoSource>();
        services.AddSingleton<IVideoSource>(sp => sp.GetRequiredService<OpenCvVideoSource>());
        services.AddHostedService(sp => sp.GetRequiredService<OpenCvVideoSource>());

        services.AddSingleton<IFrameImaging, OpenCvImaging>();

        // KEYED SERVICES: đăng ký mọi họ model, mỗi họ một khoá; chọn họ theo config lúc chạy.
        services.AddKeyedSingleton<IRecognitionModelFamily, SFaceFamily>("sface");
        services.AddKeyedSingleton<IRecognitionModelFamily, ArcFaceFamily>("arcface");
        services.AddSingleton<IFaceAnalyzerFactory>(sp =>
        {
            var model = sp.GetRequiredService<IOptions<CameraAiOptions>>().Value.Recognition.Model;
            var family = sp.GetKeyedService<IRecognitionModelFamily>(model)
                ?? throw new InvalidOperationException($"Không có model nhận diện '{model}' (chọn: sface | arcface)");
            return ActivatorUtilities.CreateInstance<FaceAnalyzerFactory>(sp, family);
        });
        return services;
    }

    public static IServiceCollection AddPersistence(this IServiceCollection services)
    {
        // DbContext: SCOPED (mặc định của AddDbContext) — 1 DbContext / 1 request, KHÔNG thread-safe.
        services.AddDbContext<CameraAiDbContext>((sp, db) =>
        {
            var dataDir = Path.GetFullPath(sp.GetRequiredService<IOptions<CameraAiOptions>>().Value.Storage.DataDir);
            Directory.CreateDirectory(dataDir);
            db.UseSqlite($"Data Source={Path.Combine(dataDir, "cameraai.db")}");
        });

        // ★ "Chuyển tiếp" (forwarding): IUnitOfWork KHÔNG đăng ký thành object mới, mà trỏ về ĐÚNG DbContext của scope.
        //   Nếu viết AddScoped<IUnitOfWork, CameraAiDbContext>() thì DI tạo DbContext THỨ HAI → repository Add vào
        //   context A, UnitOfWork Save context B → không có gì được lưu.
        services.AddScoped<IUnitOfWork>(sp => sp.GetRequiredService<CameraAiDbContext>());

        // Repository + query: Scoped vì phụ thuộc DbContext (Scoped).
        services.AddScoped<IEmployeeRepository, EmployeeRepository>();
        services.AddScoped<IFaceSampleRepository, FaceSampleRepository>();
        services.AddScoped<IAttendanceRepository, AttendanceRepository>();
        services.AddScoped<IReportQueries, ReportQueries>();

        // File store: không trạng thái thay đổi, an toàn đa luồng → Singleton.
        services.AddSingleton<IFileStore>(sp =>
            new LocalFileStore(sp.GetRequiredService<IOptions<CameraAiOptions>>().Value.Storage.DataDir));
        return services;
    }
}
