namespace CameraAI.Application.Abstractions;

// ============================================================================================
//  CQRS — phía ĐỌC (Query). Màn hình lịch sử / danh sách cần dữ liệu JOIN nhiều bảng, đã "phẳng" sẵn.
//  Đi qua Entity + Repository rồi map lại là thừa và chậm → Infrastructure chiếu (project) thẳng
//  từ DB ra DTO. Phía GHI vẫn đi qua Entity để giữ quy tắc nghiệp vụ.
//
//  Tên thuộc tính PascalCase; tầng Web cấu hình JSON snake_case → "sample_count", "first_ts"...
//  khớp đúng giao diện web cũ (static/app.js) mà không phải sửa JS.
//  Mọi thời điểm là giờ ĐỊA PHƯƠNG (app.js cắt chuỗi ISO lấy HH:mm:ss).
// ============================================================================================

public sealed record EmployeeDto(int Id, string Code, string Name, DateTimeOffset CreatedAt, int SampleCount);

public sealed record SampleDto(int Id, string? ImagePath, string Source, DateTimeOffset CreatedAt);

public sealed record AttendanceRow(
    int Id, int EmployeeId, string Code, string Name, string CameraId, DateTimeOffset Ts,
    double Score, double? LivenessScore, string? SnapshotPath, DateTimeOffset? SyncedAt);

public sealed record DailySummaryRow(
    int EmployeeId, string Code, string Name, DateTimeOffset FirstTs, DateTimeOffset LastTs, int Count);

public sealed record RejectedRow(
    int Id, string CameraId, DateTimeOffset Ts, string Reason, int? BestEmployeeId, double? Score,
    double? LivenessScore, string? SnapshotPath, string? BestCode, string? BestName);

public sealed record PendingAttendance(
    int Id, int EmployeeId, string EmployeeCode, string EmployeeName, string CameraId, DateTimeOffset Ts,
    double Score, double? LivenessScore);

public interface IReportQueries
{
    Task<List<EmployeeDto>> EmployeesAsync(CancellationToken ct = default);
    Task<List<SampleDto>> SamplesAsync(int employeeId, CancellationToken ct = default);

    /// <summary>Mỗi nhân viên: lần đầu (giờ vào), lần cuối (giờ ra), số lần chấm trong khoảng.</summary>
    Task<List<DailySummaryRow>> DailySummaryAsync(DateTimeOffset from, DateTimeOffset to, CancellationToken ct = default);

    Task<List<AttendanceRow>> AttendanceAsync(DateTimeOffset from, DateTimeOffset to, CancellationToken ct = default);
    Task<List<RejectedRow>> RejectedAsync(DateTimeOffset from, DateTimeOffset to, CancellationToken ct = default);
    Task<List<PendingAttendance>> PendingSyncAsync(int limit, CancellationToken ct = default);
}

/// <summary>
/// Chỗ cắm gửi lần chấm sang hệ thống ngoài (vd Fastdo Timekeeping). Chưa có bản cài thật —
/// nối Fastdo sau này = viết 1 class cài interface này + đăng ký DI, không sửa phần nhận diện (OCP).
/// </summary>
public interface IAttendanceSink
{
    /// <returns>Id các lần chấm đã gửi thành công.</returns>
    Task<IReadOnlyCollection<int>> SendAsync(IReadOnlyList<PendingAttendance> events, CancellationToken ct = default);
}
