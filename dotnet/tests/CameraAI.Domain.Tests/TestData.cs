using CameraAI.Domain.Attendance;
using CameraAI.Domain.Faces;

namespace CameraAI.Domain.Tests;

/// <summary>Dữ liệu giả dùng chung — tương đương tests/helpers.py bên Python.</summary>
internal static class TestData
{
    private static readonly TimeSpan Vn = TimeSpan.FromHours(7);

    public static DateTimeOffset At(int hour, int minute = 0, int second = 0) =>
        new(2026, 10, 26, hour, minute, second, Vn);

    public static FaceObservation Obs(
        int? employeeId = 1, double score = 0.7, bool? real = true, FaceBox? box = null,
        double detScore = 0.95, RejectReason? reject = null, int? nearest = null, double? liveness = null)
    {
        if (liveness is not null) real = liveness >= 0.5;
        else if (real is not null) liveness = real.Value ? 0.9 : 0.1;
        return new FaceObservation
        {
            Box = box ?? new FaceBox(100, 100, 120, 120),
            DetScore = detScore,
            Sharpness = 100,
            RejectReason = reject,
            Liveness = liveness,
            IsReal = real,
            Embedding = Enumerable.Repeat(1f, 128).ToArray(),
            Match = new FaceMatch(employeeId, score, employeeId ?? nearest),
        };
    }

    /// <summary>Cho từng frame qua tracker + decider. Trả về (chỉ số frame, quyết định).</summary>
    public static List<(int Frame, Decision Decision)> Feed(
        AttendanceDecider decider, IEnumerable<FaceObservation[]> frames, DateTimeOffset start,
        FaceTracker? tracker = null, double stepSeconds = 0.125)
    {
        tracker ??= new FaceTracker(window: 5);
        var output = new List<(int, Decision)>();
        int index = 0;
        foreach (var observations in frames)
        {
            var now = start.AddSeconds(index * stepSeconds);
            foreach (var (track, _) in tracker.Update(observations, now))
                if (decider.Evaluate(track, now) is { } decision)
                    output.Add((index, decision));
            index++;
        }
        return output;
    }

    public static List<(int, DecisionKind, int?)> Summary(List<(int Frame, Decision Decision)> results) =>
        results.Select(r => (r.Frame, r.Decision.Kind, r.Decision.EmployeeId)).ToList();

    public static IEnumerable<FaceObservation[]> Repeat(Func<FaceObservation> make, int count) =>
        Enumerable.Range(0, count).Select(_ => new[] { make() });
}
