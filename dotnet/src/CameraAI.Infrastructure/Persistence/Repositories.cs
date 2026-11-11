using CameraAI.Application.Abstractions;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Employees;
using Microsoft.EntityFrameworkCore;

namespace CameraAI.Infrastructure.Persistence;

// ============================================================================================
//  REPOSITORY = cài đặt port của Application bằng EF Core. Mỗi repository chỉ là lớp mỏng quanh DbContext:
//  Add/Remove chỉ "ghi chú" vào change tracker, KHÔNG chạm DB. DB chỉ được ghi khi ai đó gọi
//  IUnitOfWork.SaveChangesAsync (= DbContext.SaveChangesAsync) — cùng DbContext vì cùng scope.
// ============================================================================================

internal sealed class EmployeeRepository(CameraAiDbContext db) : IEmployeeRepository
{
    public Task<Employee?> GetActiveAsync(int id, CancellationToken ct = default) =>
        db.Employees.FirstOrDefaultAsync(e => e.Id == id && e.Active, ct);

    public Task<Employee?> FindByCodeAsync(string code, CancellationToken ct = default) =>
        db.Employees.FirstOrDefaultAsync(e => e.Code == code, ct);

    public Task<Dictionary<int, string>> ActiveNamesAsync(CancellationToken ct = default) =>
        db.Employees.AsNoTracking().Where(e => e.Active).ToDictionaryAsync(e => e.Id, e => e.Name, ct);

    public Task<int> ActiveCountAsync(CancellationToken ct = default) => db.Employees.CountAsync(e => e.Active, ct);

    public void Add(Employee employee) => db.Employees.Add(employee);
}

internal sealed class FaceSampleRepository(CameraAiDbContext db) : IFaceSampleRepository
{
    public Task<FaceSample?> GetAsync(int id, CancellationToken ct = default) =>
        db.FaceSamples.FirstOrDefaultAsync(s => s.Id == id, ct);

    public Task<List<FaceSample>> ListByEmployeeAsync(int employeeId, CancellationToken ct = default) =>
        db.FaceSamples.Where(s => s.EmployeeId == employeeId).OrderBy(s => s.Id).ToListAsync(ct);

    public async Task<List<(int EmployeeId, float[] Vector)>> GalleryAsync(string model, CancellationToken ct = default)
    {
        // AsNoTracking: chỉ đọc để nạp RAM → không bắt change tracker theo dõi hàng nghìn vector.
        var rows = await (
            from s in db.FaceSamples.AsNoTracking()
            join e in db.Employees on s.EmployeeId equals e.Id
            where e.Active
            from m in s.Embeddings
            where m.Model == model
            select new { s.EmployeeId, m.Vector }).ToListAsync(ct);
        return rows.Select(r => (r.EmployeeId, r.Vector)).ToList();
    }

    public Task<List<FaceSample>> MissingEmbeddingAsync(string model, CancellationToken ct = default) =>
        db.FaceSamples
            .Where(s => db.Employees.Any(e => e.Id == s.EmployeeId && e.Active))
            .Where(s => !s.Embeddings.Any(m => m.Model == model))
            .OrderBy(s => s.Id).ToListAsync(ct);

    public void Add(FaceSample sample) => db.FaceSamples.Add(sample);
    public void Remove(FaceSample sample) => db.FaceSamples.Remove(sample); // embedding (owned) bị xoá theo
}

internal sealed class AttendanceRepository(CameraAiDbContext db) : IAttendanceRepository
{
    public void Add(AttendanceEvent attendance) => db.AttendanceEvents.Add(attendance);
    public void Add(RejectedEvent rejected) => db.RejectedEvents.Add(rejected);
    public void Remove(RejectedEvent rejected) => db.RejectedEvents.Remove(rejected);

    public async Task<Dictionary<int, DateTimeOffset>> LastTimesAsync(
        string cameraId, DateTimeOffset since, CancellationToken ct = default)
    {
        // Cửa sổ chỉ vài phút → kéo về rồi gom nhóm trong RAM (đơn giản hơn ép SQL MAX() trên cột chuỗi).
        var rows = await db.AttendanceEvents.AsNoTracking()
            .Where(a => a.CameraId == cameraId && a.Timestamp >= since)
            .Select(a => new { a.EmployeeId, a.Timestamp }).ToListAsync(ct);
        return rows.GroupBy(r => r.EmployeeId).ToDictionary(g => g.Key, g => g.Max(r => r.Timestamp));
    }

    public Task<List<AttendanceEvent>> GetManyAsync(IReadOnlyCollection<int> ids, CancellationToken ct = default) =>
        db.AttendanceEvents.Where(a => ids.Contains(a.Id)).ToListAsync(ct);

    public Task<List<AttendanceEvent>> WithSnapshotBeforeAsync(DateTimeOffset cutoff, CancellationToken ct = default) =>
        db.AttendanceEvents.Where(a => a.Timestamp < cutoff && a.SnapshotPath != null).ToListAsync(ct);

    public Task<List<RejectedEvent>> RejectedBeforeAsync(DateTimeOffset cutoff, CancellationToken ct = default) =>
        db.RejectedEvents.Where(r => r.Timestamp < cutoff).ToListAsync(ct);
}
