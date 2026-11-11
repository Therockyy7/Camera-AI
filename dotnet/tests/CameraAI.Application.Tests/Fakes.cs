using CameraAI.Application.Abstractions;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Employees;
using CameraAI.Domain.Faces;

namespace CameraAI.Application.Tests;

// ============================================================================================
//  Bản giả cho từng PORT. Vì Application chỉ phụ thuộc interface, ta thay cả OpenCV, ONNX, SQLite
//  bằng vài chục dòng C# → test chạy toàn bộ luồng trong vài mili-giây.
// ============================================================================================

public sealed class FakeFrame(long id, int width = 640, int height = 480) : IVideoFrame
{
    public long Id => id;
    public int Width => width;
    public int Height => height;
    public bool Disposed { get; private set; }
    public void Dispose() => Disposed = true;
}

/// <summary>Mỗi lần hỏi trả một frame mới (Empty = true: chưa có hình).</summary>
public sealed class FakeVideoSource : IVideoSource
{
    private long _id;
    public bool Empty { get; set; }

    public IVideoFrame? TryGetLatest(long afterFrameId) => Empty ? null : new FakeFrame(++_id);
    public CameraStatus Status => new(!Empty, false, "0", 640, 480, 30, null);
}

/// <summary>Analyzer giả: trả về danh sách mặt do test quy định.</summary>
public sealed class FakeAnalyzer : IFaceAnalyzer
{
    public Func<IVideoFrame, RecognitionGate, IReadOnlyList<FaceObservation>> Faces { get; set; } = (_, _) => [];
    public string ModelName => "sface";
    public IReadOnlyList<FaceObservation> Analyze(IVideoFrame frame, RecognitionGate gate) => Faces(frame, gate);
    public float[]? EmbedLargestFace(IVideoFrame frame) => Vectors.Unit(0);
    public void Dispose() { }
}

public sealed class FakeAnalyzerFactory(FakeAnalyzer analyzer) : IFaceAnalyzerFactory
{
    public int Created { get; private set; }
    public IFaceAnalyzer Create() { Created++; return analyzer; }
}

public sealed class FakeImaging : IFrameImaging
{
    public IReadOnlyList<FaceLabel> LastLabels { get; private set; } = [];
    public byte[] EncodeAnnotated(IVideoFrame frame, IReadOnlyList<FaceLabel> labels) { LastLabels = labels; return [1, 2, 3]; }
    public byte[] EncodeCrop(IVideoFrame frame, FaceBox box, double scale) => [0xFF, 0xD8];

    /// <summary>"Ảnh" hợp lệ = bắt đầu bằng 0xFF.</summary>
    public IVideoFrame? Decode(byte[] data) => data is [0xFF, ..] ? new FakeFrame(0) : null;
}

public sealed class FakeFileStore : IFileStore
{
    private int _next;
    public Dictionary<string, byte[]> Files { get; } = [];

    public Task<string> SaveAsync(string folder, byte[] data, CancellationToken ct = default)
    {
        var path = $"{folder}/{++_next}.jpg";
        Files[path] = data;
        return Task.FromResult(path);
    }
    public Task<byte[]?> ReadAsync(string relativePath, CancellationToken ct = default) =>
        Task.FromResult(Files.GetValueOrDefault(relativePath));
    public void Delete(string? relativePath) { if (relativePath is not null) Files.Remove(relativePath); }
    public string? ResolveFullPath(string relativePath) => null;
}

/// <summary>"Database" trong RAM: cài cả repository, unit of work và query. SaveChanges gán Id như EF Core.</summary>
public sealed class InMemoryDb : IEmployeeRepository, IFaceSampleRepository, IAttendanceRepository, IUnitOfWork, IReportQueries
{
    public List<Employee> Employees { get; } = [];
    public List<FaceSample> Samples { get; } = [];
    public List<AttendanceEvent> Attendance { get; } = [];
    public List<RejectedEvent> Rejected { get; } = [];
    public int Saves { get; private set; }
    private int _nextId = 1;

    public Task<int> SaveChangesAsync(CancellationToken ct = default)
    {
        foreach (var entity in Employees.Cast<object>().Concat(Samples).Concat(Attendance).Concat(Rejected))
        {
            var id = entity.GetType().GetProperty("Id")!;
            if ((int)id.GetValue(entity)! == 0) id.SetValue(entity, _nextId++); // setter private → dùng reflection như EF
        }
        Saves++;
        return Task.FromResult(1);
    }

    // ----- IEmployeeRepository -----
    public Task<Employee?> GetActiveAsync(int id, CancellationToken ct = default) =>
        Task.FromResult(Employees.FirstOrDefault(e => e.Id == id && e.Active));
    public Task<Employee?> FindByCodeAsync(string code, CancellationToken ct = default) =>
        Task.FromResult(Employees.FirstOrDefault(e => e.Code == code));
    public Task<Dictionary<int, string>> ActiveNamesAsync(CancellationToken ct = default) =>
        Task.FromResult(Employees.Where(e => e.Active).ToDictionary(e => e.Id, e => e.Name));
    public Task<int> ActiveCountAsync(CancellationToken ct = default) => Task.FromResult(Employees.Count(e => e.Active));
    public void Add(Employee employee) => Employees.Add(employee);

    // ----- IFaceSampleRepository -----
    public Task<FaceSample?> GetAsync(int id, CancellationToken ct = default) => Task.FromResult(Samples.FirstOrDefault(s => s.Id == id));
    public Task<List<FaceSample>> ListByEmployeeAsync(int employeeId, CancellationToken ct = default) =>
        Task.FromResult(Samples.Where(s => s.EmployeeId == employeeId).ToList());
    public Task<List<(int EmployeeId, float[] Vector)>> GalleryAsync(string model, CancellationToken ct = default) =>
        Task.FromResult((
            from s in Samples
            join e in Employees on s.EmployeeId equals e.Id
            where e.Active
            from m in s.Embeddings
            where m.Model == model
            select (s.EmployeeId, m.Vector)).ToList());
    public Task<List<FaceSample>> MissingEmbeddingAsync(string model, CancellationToken ct = default) =>
        Task.FromResult(Samples.Where(s => s.Embeddings.All(m => m.Model != model)).ToList());
    public void Add(FaceSample sample) => Samples.Add(sample);
    public void Remove(FaceSample sample) => Samples.Remove(sample);

    // ----- IAttendanceRepository -----
    public void Add(AttendanceEvent attendance) => Attendance.Add(attendance);
    public void Add(RejectedEvent rejected) => Rejected.Add(rejected);
    public void Remove(RejectedEvent rejected) => Rejected.Remove(rejected);
    public Task<Dictionary<int, DateTimeOffset>> LastTimesAsync(string cameraId, DateTimeOffset since, CancellationToken ct = default) =>
        Task.FromResult(Attendance.Where(a => a.CameraId == cameraId && a.Timestamp >= since)
            .GroupBy(a => a.EmployeeId).ToDictionary(g => g.Key, g => g.Max(a => a.Timestamp)));
    public Task<List<AttendanceEvent>> GetManyAsync(IReadOnlyCollection<int> ids, CancellationToken ct = default) =>
        Task.FromResult(Attendance.Where(a => ids.Contains(a.Id)).ToList());
    public Task<List<AttendanceEvent>> WithSnapshotBeforeAsync(DateTimeOffset cutoff, CancellationToken ct = default) =>
        Task.FromResult(Attendance.Where(a => a.Timestamp < cutoff && a.SnapshotPath != null).ToList());
    public Task<List<RejectedEvent>> RejectedBeforeAsync(DateTimeOffset cutoff, CancellationToken ct = default) =>
        Task.FromResult(Rejected.Where(r => r.Timestamp < cutoff).ToList());

    // ----- IReportQueries (chỉ phần test cần) -----
    public Task<List<EmployeeDto>> EmployeesAsync(CancellationToken ct = default) =>
        Task.FromResult(Employees.Where(e => e.Active)
            .Select(e => new EmployeeDto(e.Id, e.Code, e.Name, e.CreatedAt, Samples.Count(s => s.EmployeeId == e.Id))).ToList());
    public Task<List<SampleDto>> SamplesAsync(int employeeId, CancellationToken ct = default) =>
        Task.FromResult(Samples.Where(s => s.EmployeeId == employeeId)
            .Select(s => new SampleDto(s.Id, s.ImagePath, s.Source.ToString().ToLowerInvariant(), s.CreatedAt)).ToList());
    public Task<List<AttendanceRow>> AttendanceAsync(DateTimeOffset from, DateTimeOffset to, CancellationToken ct = default) =>
        Task.FromResult(Attendance
            .Where(a => a.Timestamp >= from && a.Timestamp < to) // "from" là từ khoá trong cú pháp query → dùng method syntax
            .Join(Employees, a => a.EmployeeId, e => e.Id, (a, e) => new AttendanceRow(a.Id, e.Id, e.Code, e.Name,
                a.CameraId, a.Timestamp, a.Score, a.LivenessScore, a.SnapshotPath, a.SyncedAt))
            .OrderBy(r => r.Ts).ToList());
    public Task<List<DailySummaryRow>> DailySummaryAsync(DateTimeOffset from, DateTimeOffset to, CancellationToken ct = default) =>
        throw new NotSupportedException();
    public Task<List<RejectedRow>> RejectedAsync(DateTimeOffset from, DateTimeOffset to, CancellationToken ct = default) =>
        throw new NotSupportedException();
    public Task<List<PendingAttendance>> PendingSyncAsync(int limit, CancellationToken ct = default) =>
        throw new NotSupportedException();
}

public static class Vectors
{
    /// <summary>Vector 128 chiều chỉ có 1 ở vị trí <paramref name="axis"/> — mỗi "người" một trục, cosine giữa 2 người = 0.</summary>
    public static float[] Unit(int axis)
    {
        var v = new float[128];
        v[axis] = 1;
        return v;
    }

    public static float[] Random(Random rng) => Enumerable.Range(0, 128).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
}
