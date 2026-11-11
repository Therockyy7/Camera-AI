using CameraAI.Application.Common;
using CameraAI.Application.Enrollment;
using CameraAI.Application.Faces;
using CameraAI.Application.Messaging;
using CameraAI.Application.Pipeline;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;

namespace CameraAI.Application;

/// <summary>
/// Tầng Application tự khai báo "tôi cần đăng ký những gì". Program.cs chỉ gọi <c>services.AddApplication()</c>
/// → Program.cs không phải biết từng class bên trong (mỗi tầng tự lo phần đăng ký của mình).
///
/// Đọc kỹ cột vòng đời — đây là phần hay bị hỏi phỏng vấn nhất:
/// </summary>
public static class DependencyInjection
{
    public static IServiceCollection AddApplication(this IServiceCollection services)
    {
        // ---------- SINGLETON: một instance cho cả đời ứng dụng ----------
        services.TryAddSingleton(TimeProvider.System);   // TryAdd: test có thể đăng ký FakeTimeProvider trước
        services.AddSingleton<LocalClock>();
        services.AddSingleton(sp => new FaceGallery(Opt(sp).Recognition.ActiveThreshold)); // gallery trong RAM, dùng chung
        services.AddSingleton<EmployeeNames>();

        // Factory (static) chọn Strategy theo config — gọi đúng 1 lần ở đây (composition), như đã bàn ở Bài 01.
        services.AddSingleton(sp => LivenessPolicyFactory.Create(Opt(sp).Liveness.Mode));
        services.AddSingleton(sp => new AttendanceDecider(
            Opt(sp).Attendance.ToRules(Opt(sp).Liveness.SpoofThreshold),
            sp.GetRequiredService<ILivenessPolicy>()));
        services.AddSingleton<FaceLabeler>();
        services.AddSingleton<AttendancePipeline>();   // giữ trạng thái tracker xuyên suốt → phải Singleton
        services.AddSingleton<EnrollmentAnalyzer>();   // model nạp 1 lần + khoá dùng chung

        // ---------- SCOPED: một instance cho mỗi HTTP request (hoặc mỗi scope tự mở) ----------
        services.AddScoped<GalleryLoader>();
        services.AddScoped<FaceEnrollmentService>();
        services.AddScoped<SnapshotRetention>();
        services.AddScoped<ISender, Sender>();
        // AttendanceSyncService: chỉ đăng ký khi có IAttendanceSink thật (Fastdo) — để trống thì DI báo lỗi lúc khởi động.

        // ---------- Handler + Behavior ----------
        RegisterHandlers(services);
        services.AddScoped(typeof(IPipelineBehavior<,>), typeof(LoggingBehavior<,>)); // open generic: áp cho MỌI request
        return services;
    }

    private static CameraAiOptions Opt(IServiceProvider sp) => sp.GetRequiredService<IOptions<CameraAiOptions>>().Value;

    /// <summary>Quét assembly: mọi class cài IRequestHandler&lt;,&gt; → đăng ký Scoped. Thêm use case mới không phải sửa file này.</summary>
    private static void RegisterHandlers(IServiceCollection services)
    {
        var handlers =
            from type in typeof(DependencyInjection).Assembly.GetTypes()
            where type is { IsClass: true, IsAbstract: false }
            from contract in type.GetInterfaces()
            where contract.IsGenericType && contract.GetGenericTypeDefinition() == typeof(IRequestHandler<,>)
            select (contract, type);
        foreach (var (contract, type) in handlers) services.AddScoped(contract, type);
    }
}
