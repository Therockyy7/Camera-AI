namespace CameraAI.Domain.Faces;

/// <summary>
/// Danh sách embedding đã đăng ký (trong RAM) và so khớp cosine.
///
/// Thread-safety không dùng lock: mỗi lần Load tạo một <see cref="Snapshot"/> MỚI, bất biến, rồi tráo tham chiếu.
/// Luồng xử lý camera đang Match trên snapshot cũ vẫn chạy đúng; lần Match sau sẽ thấy snapshot mới.
/// Sẽ được đăng ký Singleton trong DI (một gallery dùng chung cho cả ứng dụng).
/// </summary>
public sealed class FaceGallery(double threshold)
{
    private sealed record Snapshot(float[][] Vectors, int[] Owners)
    {
        public static readonly Snapshot Empty = new([], []);
    }

    private volatile Snapshot _snapshot = Snapshot.Empty;

    public double Threshold { get; } = threshold;
    public int Size => _snapshot.Owners.Length;

    public void Load(IEnumerable<(int EmployeeId, float[] Embedding)> samples)
    {
        var items = samples.ToArray();
        _snapshot = new Snapshot(
            items.Select(s => VectorMath.Normalize(s.Embedding)).ToArray(),
            items.Select(s => s.EmployeeId).ToArray());
    }

    public FaceMatch Match(float[] embedding)
    {
        var snap = _snapshot; // đọc tham chiếu MỘT lần → cả hàm dùng cùng một snapshot
        if (snap.Owners.Length == 0) return FaceMatch.None;

        var query = VectorMath.Normalize(embedding);
        int best = 0;
        double bestScore = double.NegativeInfinity;
        for (int i = 0; i < snap.Vectors.Length; i++)
        {
            double score = VectorMath.Dot(snap.Vectors[i], query);
            if (score > bestScore) (best, bestScore) = (i, score);
        }
        int owner = snap.Owners[best];
        return new FaceMatch(bestScore >= Threshold ? owner : null, bestScore, owner);
    }
}

public static class VectorMath
{
    /// <summary>Chia cho độ dài để ‖v‖ = 1 → tích vô hướng chính là cosine. Vector 0 giữ nguyên (không NaN).</summary>
    public static float[] Normalize(float[] v)
    {
        double norm = Math.Sqrt(Dot(v, v));
        if (norm == 0) return (float[])v.Clone();
        var result = new float[v.Length];
        for (int i = 0; i < v.Length; i++) result[i] = (float)(v[i] / norm);
        return result;
    }

    public static double Dot(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Embedding khác số chiều ({a.Length} vs {b.Length}) — lẫn model?");
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += (double)a[i] * b[i];
        return sum;
    }
}
