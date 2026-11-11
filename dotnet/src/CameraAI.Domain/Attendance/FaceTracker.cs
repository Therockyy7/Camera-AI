using CameraAI.Domain.Faces;

namespace CameraAI.Domain.Attendance;

/// <summary>
/// "Vệt" của một khuôn mặt qua nhiều frame liên tiếp. Không quyết định trên 1 frame đơn lẻ (dễ nhầm),
/// mà nhìn cửa sổ N quan sát gần nhất của cùng một vệt.
/// </summary>
public sealed class Track
{
    private readonly Queue<FaceObservation> _history = new();
    private readonly int _window;

    internal Track(int id, FaceBox box, DateTimeOffset now, int window)
    {
        Id = id;
        Box = box;
        LastSeen = now;
        _window = window;
    }

    public int Id { get; }
    public FaceBox Box { get; internal set; }
    public DateTimeOffset LastSeen { get; internal set; }

    /// <summary>N quan sát DÙNG ĐƯỢC gần nhất (cũ → mới).</summary>
    public IReadOnlyCollection<FaceObservation> History => _history;

    /// <summary>Các loại quyết định đã đưa ra cho vệt này (mỗi loại chỉ một lần).</summary>
    public HashSet<DecisionKind> Done { get; } = [];

    /// <summary>Nhân viên đã xác nhận trên vệt này (để vẽ nhãn xanh).</summary>
    public int? EmployeeId { get; internal set; }

    internal void Push(FaceObservation obs)
    {
        _history.Enqueue(obs);
        while (_history.Count > _window) _history.Dequeue();
    }
}

/// <summary>
/// Gán mặt ở frame mới vào vệt cũ bằng IoU (khung chồng lên nhau nhiều = cùng người), tạo vệt mới nếu cần,
/// xoá vệt quá lâu không thấy. Logic thuần — test bằng dữ liệu giả, không cần camera.
/// </summary>
public sealed class FaceTracker(int window = 5, double maxAgeSeconds = 1.5, double iouThreshold = 0.3)
{
    private readonly Dictionary<int, Track> _tracks = [];
    private int _nextId = 1;

    public IReadOnlyCollection<Track> Tracks => _tracks.Values;

    public IReadOnlyList<(Track Track, FaceObservation Observation)> Update(
        IReadOnlyList<FaceObservation> observations, DateTimeOffset now)
    {
        // Ghép tham lam: cặp (vệt, mặt) có IoU cao nhất được ghép trước.
        var candidates =
            from t in _tracks.Values
            from i in Enumerable.Range(0, observations.Count)
            let overlap = t.Box.Iou(observations[i].Box)
            where overlap >= iouThreshold
            orderby overlap descending, t.Id descending, i descending
            select (t, i);

        var assigned = new Dictionary<int, Track>();
        var usedTracks = new HashSet<int>();
        foreach (var (track, index) in candidates)
        {
            if (usedTracks.Contains(track.Id) || assigned.ContainsKey(index)) continue;
            usedTracks.Add(track.Id);
            assigned[index] = track;
        }

        var result = new List<(Track, FaceObservation)>(observations.Count);
        for (int i = 0; i < observations.Count; i++)
        {
            var obs = observations[i];
            if (!assigned.TryGetValue(i, out var track))
            {
                track = new Track(_nextId++, obs.Box, now, window);
                _tracks[track.Id] = track;
            }
            track.Box = obs.Box;
            track.LastSeen = now;
            if (obs.Usable) track.Push(obs);
            result.Add((track, obs));
        }

        foreach (var expired in _tracks.Values.Where(t => (now - t.LastSeen).TotalSeconds > maxAgeSeconds).ToList())
            _tracks.Remove(expired.Id);
        return result;
    }
}
