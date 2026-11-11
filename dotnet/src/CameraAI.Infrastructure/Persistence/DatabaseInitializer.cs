using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace CameraAI.Infrastructure.Persistence;

public static class DatabaseInitializer
{
    /// <summary>
    /// Tạo DB nếu chưa có (cùng schema với bản Python) và bật WAL (đọc/ghi song song tốt hơn: luồng camera ghi,
    /// web đọc). File DB đã có sẵn — kể cả do bản Python tạo — thì giữ nguyên.
    ///
    /// Vì sao EnsureCreated mà không dùng Migrations? Schema đang do bản Python quản lý chung. Dự án .NET thuần
    /// thì nên dùng <c>dotnet ef migrations add</c> để có lịch sử thay đổi schema.
    /// </summary>
    public static async Task InitializeDatabaseAsync(this IServiceProvider services, CancellationToken ct = default)
    {
        await using var scope = services.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CameraAiDbContext>();
        var log = scope.ServiceProvider.GetRequiredService<ILogger<CameraAiDbContext>>();

        if (await db.Database.EnsureCreatedAsync(ct))
        {
            await db.Database.ExecuteSqlRawAsync("PRAGMA user_version = 1", ct); // = SCHEMA_VERSION của bản Python
            log.LogInformation("Đã tạo cơ sở dữ liệu mới");
        }
        await db.Database.ExecuteSqlRawAsync("PRAGMA journal_mode=WAL", ct);
    }
}
