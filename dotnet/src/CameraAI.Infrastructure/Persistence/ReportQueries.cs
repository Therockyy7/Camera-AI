using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using Microsoft.EntityFrameworkCore;

namespace CameraAI.Infrastructure.Persistence;

/// <summary>
/// CQRS phía ĐỌC: chiếu (project) thẳng từ bảng ra DTO bằng LINQ → EF sinh câu SELECT … JOIN chỉ lấy đúng cột cần,
/// không dựng Entity, không change tracking.
/// </summary>
internal sealed class ReportQueries(CameraAiDbContext db) : IReportQueries
{
    public Task<List<EmployeeDto>> EmployeesAsync(CancellationToken ct = default) =>
        db.Employees.AsNoTracking()
            .Where(e => e.Active)
            .OrderBy(e => e.Code)
            .Select(e => new EmployeeDto(e.Id, e.Code, e.Name, e.CreatedAt,
                db.FaceSamples.Count(s => s.EmployeeId == e.Id)))          // → subquery COUNT(*)
            .ToListAsync(ct);

    public async Task<List<SampleDto>> SamplesAsync(int employeeId, CancellationToken ct = default)
    {
        var rows = await db.FaceSamples.AsNoTracking()
            .Where(s => s.EmployeeId == employeeId).OrderBy(s => s.Id)
            .Select(s => new { s.Id, s.ImagePath, s.Source, s.CreatedAt }).ToListAsync(ct);
        return rows.Select(s => new SampleDto(s.Id, s.ImagePath, s.Source.ToString().ToLowerInvariant(), s.CreatedAt)).ToList();
    }

    public Task<List<AttendanceRow>> AttendanceAsync(DateTimeOffset start, DateTimeOffset end, CancellationToken ct = default) =>
        (from a in db.AttendanceEvents.AsNoTracking()
         where a.Timestamp >= start && a.Timestamp < end
         join e in db.Employees on a.EmployeeId equals e.Id
         orderby a.Timestamp          // sắp theo CỘT GỐC; sắp theo r.Ts SAU khi tạo DTO thì EF không dịch ra SQL được
         select new AttendanceRow(a.Id, a.EmployeeId, e.Code, e.Name, a.CameraId, a.Timestamp, a.Score,
             a.LivenessScore, a.SnapshotPath, a.SyncedAt))
        .ToListAsync(ct);

    public async Task<List<DailySummaryRow>> DailySummaryAsync(DateTimeOffset from, DateTimeOffset to, CancellationToken ct = default)
    {
        // Gom trong RAM: 1 ngày chỉ vài trăm lần chấm; tránh MIN/MAX trên cột thời gian dạng chuỗi.
        var events = await AttendanceAsync(from, to, ct);
        return events
            .GroupBy(e => (e.EmployeeId, e.Code, e.Name))
            .Select(g => new DailySummaryRow(g.Key.EmployeeId, g.Key.Code, g.Key.Name,
                g.Min(e => e.Ts), g.Max(e => e.Ts), g.Count()))
            .OrderBy(r => r.FirstTs).ToList();
    }

    // Tham số đặt tên start/end: trong cú pháp LINQ dạng query, "from" là từ khoá.
    public async Task<List<RejectedRow>> RejectedAsync(DateTimeOffset start, DateTimeOffset end, CancellationToken ct = default)
    {
        var rows = await (
            from r in db.RejectedEvents.AsNoTracking()
            where r.Timestamp >= start && r.Timestamp < end
            join e in db.Employees on r.BestEmployeeId equals (int?)e.Id into best
            from e in best.DefaultIfEmpty()                                   // LEFT JOIN
            orderby r.Timestamp descending
            select new { r, Code = e == null ? null : e.Code, Name = e == null ? null : e.Name }).ToListAsync(ct);
        return rows.Select(x => new RejectedRow(x.r.Id, x.r.CameraId, x.r.Timestamp, x.r.Reason.ToApiString(),
            x.r.BestEmployeeId, x.r.Score, x.r.LivenessScore, x.r.SnapshotPath, x.Code, x.Name)).ToList();
    }

    public Task<List<PendingAttendance>> PendingSyncAsync(int limit, CancellationToken ct = default) =>
        db.AttendanceEvents.AsNoTracking()
            .Where(a => a.SyncedAt == null)
            .OrderBy(a => a.Id).Take(limit)
            .Join(db.Employees, a => a.EmployeeId, e => e.Id, (a, e) => new PendingAttendance(
                a.Id, a.EmployeeId, e.Code, e.Name, a.CameraId, a.Timestamp, a.Score, a.LivenessScore))
            .ToListAsync(ct);
}
