using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Messaging;
using CameraAI.Domain.Employees;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Time.Testing;

namespace CameraAI.Application.Tests;

/// <summary>
/// Dựng DI container THẬT giống Program.cs, chỉ thay các port bằng bản giả.
/// Nhờ vậy test kiểm luôn cả phần đăng ký DI (thiếu đăng ký / sai vòng đời là test đỏ).
/// </summary>
public sealed class TestHost : IDisposable
{
    public InMemoryDb Db { get; } = new();
    public FakeFileStore Files { get; } = new();
    public FakeVideoSource Source { get; } = new();
    public FakeAnalyzer Analyzer { get; } = new();
    public FakeImaging Imaging { get; } = new();
    public FakeTimeProvider Time { get; } = new(new DateTimeOffset(2026, 10, 26, 1, 0, 0, TimeSpan.Zero)); // = 08:00 giờ VN
    public ServiceProvider Services { get; }

    public TestHost(Action<CameraAiOptions>? configure = null)
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.Configure<CameraAiOptions>(o =>
        {
            o.Enrollment.SamplesPerCapture = 3;
            o.Enrollment.PollIntervalMs = 0;   // test không ngủ thật
            o.Enrollment.SampleIntervalMs = 0;
            configure?.Invoke(o);
        });
        services.AddSingleton<TimeProvider>(Time);   // đăng ký TRƯỚC AddApplication → TryAddSingleton bên trong bỏ qua
        services.AddApplication();

        // Các port: một object giả phục vụ nhiều interface.
        // Đăng ký ĐÚNG vòng đời của bản thật (repository = Scoped vì dùng DbContext) → nếu ai đó lỡ tiêm
        // repository vào một Singleton, ValidateOnBuild bên dưới báo lỗi ngay, như khi chạy app thật.
        services.AddScoped<IEmployeeRepository>(_ => Db);
        services.AddScoped<IFaceSampleRepository>(_ => Db);
        services.AddScoped<IAttendanceRepository>(_ => Db);
        services.AddScoped<IUnitOfWork>(_ => Db);
        services.AddScoped<IReportQueries>(_ => Db);
        services.AddScoped<IFileStore>(_ => Files);
        services.AddSingleton<IVideoSource>(Source);
        services.AddSingleton<IFrameImaging>(Imaging);
        services.AddSingleton<IFaceAnalyzerFactory>(new FakeAnalyzerFactory(Analyzer));

        // ValidateScopes + ValidateOnBuild: như môi trường Development của ASP.NET Core —
        // bắt lỗi Singleton giữ Scoped (captive dependency) ngay khi dựng container.
        Services = services.BuildServiceProvider(new ServiceProviderOptions { ValidateScopes = true, ValidateOnBuild = true });
    }

    /// <summary>Gửi request trong một scope mới — giống mỗi HTTP request một scope.</summary>
    public async Task<T> Send<T>(IRequest<T> request)
    {
        await using var scope = Services.CreateAsyncScope();
        return await scope.ServiceProvider.GetRequiredService<ISender>().Send(request);
    }

    public T Get<T>() where T : notnull => Services.GetRequiredService<T>();

    /// <summary>Thêm nhân viên có sẵn 1 mẫu khuôn mặt (embedding = trục <paramref name="axis"/>).</summary>
    public Employee SeedEmployee(string code, string name, int axis)
    {
        var e = Employee.Create(code, name, Time.GetUtcNow());
        Db.Add(e);
        Db.SaveChangesAsync().GetAwaiter().GetResult();
        Db.Add(FaceSample.Create(e.Id, null, SampleSource.Upload, Time.GetUtcNow(), "sface", Vectors.Unit(axis)));
        Db.SaveChangesAsync().GetAwaiter().GetResult();
        return e;
    }

    public static FaceObservation Face(float[] embedding, FaceBox? box = null, bool? real = true,
        RejectReason? reject = null, double detScore = 0.95) => new()
    {
        Box = box ?? new FaceBox(100, 100, 120, 120),
        DetScore = detScore,
        Sharpness = 100,
        RejectReason = reject,
        Liveness = real is null ? null : real.Value ? 0.9 : 0.1,
        IsReal = real,
        Embedding = reject is null ? embedding : null,
    };

    public void Dispose() => Services.Dispose();
}
