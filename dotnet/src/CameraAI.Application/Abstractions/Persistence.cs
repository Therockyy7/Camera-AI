using CameraAI.Domain.Attendance;
using CameraAI.Domain.Employees;

namespace CameraAI.Application.Abstractions;

// ============================================================================================
//  REPOSITORY PATTERN (phía GHI — Command). Làm việc với ENTITY của Domain như một "danh sách trong bộ nhớ":
//  Add / Remove / Get. Không có SaveChanges ở từng repository — việc chốt giao dịch là của IUnitOfWork.
//  Infrastructure cài bằng EF Core; mọi repository trong một HTTP request dùng chung MỘT DbContext (Scoped)
//  → một lần SaveChangesAsync ghi tất cả trong một transaction.
// ============================================================================================

public interface IEmployeeRepository
{
    Task<Employee?> GetActiveAsync(int id, CancellationToken ct = default);

    /// <summary>Tìm theo mã, KỂ CẢ nhân viên đã ngừng (để kích hoạt lại thay vì báo trùng mã).</summary>
    Task<Employee?> FindByCodeAsync(string code, CancellationToken ct = default);

    /// <summary>id → tên của nhân viên đang hoạt động (vẽ nhãn trên hình).</summary>
    Task<Dictionary<int, string>> ActiveNamesAsync(CancellationToken ct = default);

    Task<int> ActiveCountAsync(CancellationToken ct = default);

    void Add(Employee employee);
}

public interface IFaceSampleRepository
{
    Task<FaceSample?> GetAsync(int id, CancellationToken ct = default);
    Task<List<FaceSample>> ListByEmployeeAsync(int employeeId, CancellationToken ct = default);

    /// <summary>Mọi embedding của <paramref name="model"/> thuộc nhân viên đang hoạt động — nạp vào FaceGallery.</summary>
    Task<List<(int EmployeeId, float[] Vector)>> GalleryAsync(string model, CancellationToken ct = default);

    /// <summary>Mẫu chưa có embedding của <paramref name="model"/> (sau khi đổi model).</summary>
    Task<List<FaceSample>> MissingEmbeddingAsync(string model, CancellationToken ct = default);

    void Add(FaceSample sample);
    void Remove(FaceSample sample);
}

public interface IAttendanceRepository
{
    void Add(AttendanceEvent attendance);
    void Add(RejectedEvent rejected);
    void Remove(RejectedEvent rejected);

    /// <summary>Lần chấm gần nhất của từng nhân viên tại camera này kể từ <paramref name="since"/> (khôi phục cooldown).</summary>
    Task<Dictionary<int, DateTimeOffset>> LastTimesAsync(string cameraId, DateTimeOffset since, CancellationToken ct = default);

    Task<List<AttendanceEvent>> GetManyAsync(IReadOnlyCollection<int> ids, CancellationToken ct = default);
    Task<List<AttendanceEvent>> WithSnapshotBeforeAsync(DateTimeOffset cutoff, CancellationToken ct = default);
    Task<List<RejectedEvent>> RejectedBeforeAsync(DateTimeOffset cutoff, CancellationToken ct = default);
}

/// <summary>Chốt mọi thay đổi của các repository trong cùng phạm vi (scope) thành một giao dịch.</summary>
public interface IUnitOfWork
{
    Task<int> SaveChangesAsync(CancellationToken ct = default);
}

/// <summary>File ảnh (mẫu khuôn mặt, ảnh chụp lúc chấm) trong thư mục data/. Đường dẫn trả về là TƯƠNG ĐỐI.</summary>
public interface IFileStore
{
    Task<string> SaveAsync(string folder, byte[] data, CancellationToken ct = default);
    Task<byte[]?> ReadAsync(string relativePath, CancellationToken ct = default);
    void Delete(string? relativePath);

    /// <summary>Đường dẫn tuyệt đối để web trả file; null nếu không tồn tại hoặc trỏ ra ngoài data/ (chặn "../").</summary>
    string? ResolveFullPath(string relativePath);
}
