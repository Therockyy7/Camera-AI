using CameraAI.Domain.Faces;

namespace CameraAI.Domain.Attendance;

public enum DecisionKind { Attendance, Spoof, Unknown }

/// <summary>Quyết định cho một vệt mặt: chấm công / giả mạo / người lạ.</summary>
/// <param name="Observation">Quan sát đại diện — dùng để cắt ảnh lưu lại.</param>
/// <param name="EmployeeId">Người khớp (Attendance) hoặc người gần nhất (Spoof/Unknown).</param>
public sealed record Decision(
    DecisionKind Kind, int TrackId, FaceObservation Observation, int? EmployeeId, double Score, double? Liveness);

/// <summary>Các con số nghiệp vụ của việc chấm công (đọc từ config ở tầng ngoài, truyền vào đây).</summary>
public sealed record AttendanceRules(
    int WindowFrames = 5,       // số quan sát gần nhất xét cho mỗi vệt
    int ConfirmFrames = 3,      // cần ít nhất chừng này phiếu cho cùng một người
    int CooldownMinutes = 5,    // không ghi lại cùng người trong khoảng này
    double SpoofThreshold = 0.3 // ghi "giả mạo" khi điểm mặt thật TRUNG BÌNH cả cửa sổ thấp hơn
);

/// <summary>
/// Quyết định khi nào ghi chấm công / từ chối, dựa trên lịch sử của một vệt.
/// Phụ thuộc vào <see cref="ILivenessPolicy"/> (abstraction) chứ không phải class cụ thể → DIP.
///
/// Không thread-safe: chỉ luồng xử lý camera gọi vào.
/// </summary>
public sealed class AttendanceDecider(AttendanceRules rules, ILivenessPolicy policy)
{
    private readonly Dictionary<int, DateTimeOffset> _lastRecorded = [];

    public ILivenessPolicy Policy => policy;

    /// <summary>Nạp lần chấm gần nhất (từ DB) để cooldown đúng cả sau khi khởi động lại.</summary>
    public void Seed(IReadOnlyDictionary<int, DateTimeOffset> lastRecorded)
    {
        foreach (var (id, at) in lastRecorded) _lastRecorded[id] = at;
    }

    public DateTimeOffset? LastRecorded(int employeeId) =>
        _lastRecorded.TryGetValue(employeeId, out var at) ? at : null;

    /// <summary>
    /// Cửa sổ HIỆN TẠI có đủ frame ĐÃ KIỂM và điểm trung bình dưới ngưỡng. Dùng trung bình vì mặt thật
    /// đôi khi tụt điểm vài frame (ánh sáng, cử động), còn ảnh/màn hình thì thấp liên tục.
    /// </summary>
    public bool LooksSpoof(Track track)
    {
        if (!policy.CanFlagSpoof) return false;
        var scores = track.History.Where(o => o.Liveness is not null).Select(o => o.Liveness!.Value).ToList();
        return scores.Count >= rules.WindowFrames && scores.Average() < rules.SpoofThreshold;
    }

    public Decision? Evaluate(Track track, DateTimeOffset now)
    {
        var history = track.History.ToList();
        if (history.Count == 0) return null;

        // 1) Giả mạo
        if (!track.Done.Contains(DecisionKind.Spoof) && LooksSpoof(track))
        {
            track.Done.Add(DecisionKind.Spoof);
            var rep = history.MinBy(o => o.Liveness ?? 1.0)!;
            return new Decision(DecisionKind.Spoof, track.Id, rep, rep.Match?.BestEmployeeId, ScoreOf(rep), rep.Liveness);
        }

        var voters = history.Where(policy.CountsTowardVote).ToList();

        // 2) Chấm công: bỏ phiếu đa số trong cửa sổ
        if (!track.Done.Contains(DecisionKind.Attendance))
        {
            var winner = voters
                .Where(o => o.Match?.EmployeeId is not null)
                .GroupBy(o => o.Match!.EmployeeId!.Value)
                .MaxBy(g => g.Count());
            if (winner is not null && winner.Count() >= rules.ConfirmFrames)
            {
                int employeeId = winner.Key;
                track.Done.Add(DecisionKind.Attendance);
                track.EmployeeId = employeeId;

                if (LastRecorded(employeeId) is { } last && now - last < TimeSpan.FromMinutes(rules.CooldownMinutes))
                    return null; // đã chấm gần đây → nhận ra nhưng không ghi lại

                _lastRecorded[employeeId] = now;
                var best = winner.MaxBy(ScoreOf)!;
                var checkedScores = winner.Where(o => o.Liveness is not null).Select(o => o.Liveness!.Value).ToList();
                double? liveness = checkedScores.Count > 0 ? checkedScores.Average() : null;
                return new Decision(DecisionKind.Attendance, track.Id, best, employeeId, ScoreOf(best), liveness);
            }
        }

        // 3) Người lạ: cả cửa sổ đều hợp lệ nhưng không khớp ai
        if (!track.Done.Contains(DecisionKind.Unknown)
            && track.EmployeeId is null
            && history.Count >= rules.WindowFrames
            && voters.Count == history.Count
            && history.All(o => o.Match is { EmployeeId: null }))
        {
            track.Done.Add(DecisionKind.Unknown);
            var rep = history.MaxBy(o => o.DetScore)!;
            return new Decision(DecisionKind.Unknown, track.Id, rep, rep.Match?.BestEmployeeId, ScoreOf(rep), rep.Liveness);
        }
        return null;
    }

    private static double ScoreOf(FaceObservation o) => o.Match?.Score ?? 0.0;
}
