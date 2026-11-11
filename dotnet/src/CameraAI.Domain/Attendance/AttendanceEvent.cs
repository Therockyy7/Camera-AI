namespace CameraAI.Domain.Attendance;

/// <summary>Một lần chấm công thành công.</summary>
public sealed class AttendanceEvent
{
    public int Id { get; private set; }
    public int EmployeeId { get; private set; }
    public string CameraId { get; private set; } = "";
    public DateTimeOffset Timestamp { get; private set; }
    public double Score { get; private set; }
    public double? LivenessScore { get; private set; }   // null = chưa kiểm được giả mạo → hậu kiểm bằng ảnh
    public string? SnapshotPath { get; private set; }
    public DateTimeOffset? SyncedAt { get; private set; } // null = chưa gửi sang hệ thống ngoài (Fastdo)

    private AttendanceEvent() { }

    public static AttendanceEvent Record(
        int employeeId, string cameraId, DateTimeOffset at, double score, double? liveness, string? snapshotPath) =>
        new()
        {
            EmployeeId = employeeId, CameraId = cameraId, Timestamp = at,
            Score = score, LivenessScore = liveness, SnapshotPath = snapshotPath,
        };

    public void MarkSynced(DateTimeOffset at) => SyncedAt = at;

    /// <summary>Hết hạn lưu ảnh: bỏ ảnh nhưng GIỮ lần chấm công.</summary>
    public void ClearSnapshot() => SnapshotPath = null;
}

/// <summary>Một lần bị từ chối (giả mạo / người lạ) — chụp ảnh lưu lại để hậu kiểm.</summary>
public sealed class RejectedEvent
{
    public int Id { get; private set; }
    public string CameraId { get; private set; } = "";
    public DateTimeOffset Timestamp { get; private set; }
    public DecisionKind Reason { get; private set; }
    public int? BestEmployeeId { get; private set; }
    public double? Score { get; private set; }
    public double? LivenessScore { get; private set; }
    public string? SnapshotPath { get; private set; }

    private RejectedEvent() { }

    public static RejectedEvent Record(
        string cameraId, DateTimeOffset at, DecisionKind reason, int? bestEmployeeId,
        double? score, double? liveness, string? snapshotPath)
    {
        if (reason == DecisionKind.Attendance)
            throw new ArgumentException("Chấm công thành công không phải sự kiện bị từ chối", nameof(reason));
        return new()
        {
            CameraId = cameraId, Timestamp = at, Reason = reason, BestEmployeeId = bestEmployeeId,
            Score = score, LivenessScore = liveness, SnapshotPath = snapshotPath,
        };
    }
}
