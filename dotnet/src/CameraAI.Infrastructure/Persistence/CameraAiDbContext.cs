using System.Text;
using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Employees;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace CameraAI.Infrastructure.Persistence;

/// <summary>
/// DbContext của EF Core = "phiên làm việc" với DB: theo dõi entity đã đọc/thêm/sửa (change tracker) và ghi
/// tất cả trong MỘT transaction khi SaveChanges.
///
/// Nó cài luôn <see cref="IUnitOfWork"/>: <c>IUnitOfWork.SaveChangesAsync</c> CHÍNH LÀ <c>DbContext.SaveChangesAsync</c>.
/// Đăng ký SCOPED, và mọi repository + IUnitOfWork trong cùng scope trỏ về CÙNG một DbContext (xem DependencyInjection.cs)
/// → repository Add, unit of work Save — thấy nhau.
/// </summary>
public sealed class CameraAiDbContext(DbContextOptions<CameraAiDbContext> options, IOptions<CameraAiOptions> settings)
    : DbContext(options), IUnitOfWork
{
    public DbSet<Employee> Employees => Set<Employee>();
    public DbSet<FaceSample> FaceSamples => Set<FaceSample>();
    public DbSet<AttendanceEvent> AttendanceEvents => Set<AttendanceEvent>();
    public DbSet<RejectedEvent> RejectedEvents => Set<RejectedEvent>();

    protected override void OnModelCreating(ModelBuilder model)
    {
        // Mỗi entity một class cấu hình (Fluent API) — Domain không có attribute [Table], [Column] nào.
        model.ApplyConfigurationsFromAssembly(typeof(CameraAiDbContext).Assembly);

        var zone = TimeZoneInfo.FindSystemTimeZoneById(settings.Value.TimeZone);
        var time = new LocalIsoTimeConverter(zone);
        foreach (var entity in model.Model.GetEntityTypes())
        {
            foreach (var property in entity.GetProperties())
            {
                // Tên cột snake_case như bản Python (EmployeeId → employee_id), trừ cột đã đặt tên riêng.
                if (property.GetColumnName() == property.Name) property.SetColumnName(ToSnakeCase(property.Name));
                if (property.ClrType == typeof(DateTimeOffset) || property.ClrType == typeof(DateTimeOffset?))
                    property.SetValueConverter(time);
            }
        }
    }

    private static string ToSnakeCase(string name)
    {
        var sb = new StringBuilder(name.Length + 4);
        for (int i = 0; i < name.Length; i++)
        {
            if (char.IsUpper(name[i]) && i > 0) sb.Append('_');
            sb.Append(char.ToLowerInvariant(name[i]));
        }
        return sb.ToString();
    }
}
