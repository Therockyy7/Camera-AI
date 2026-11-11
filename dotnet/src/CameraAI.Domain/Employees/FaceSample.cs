namespace CameraAI.Domain.Employees;

public enum SampleSource { Camera, Upload }

/// <summary>
/// Một mẫu khuôn mặt đã đăng ký. Mỗi mẫu có thể có nhiều embedding — một cho mỗi model nhận diện
/// (SFace 128 chiều, ArcFace 512 chiều) — nên đổi model không phải đăng ký lại, chỉ cần tính lại từ ảnh.
/// </summary>
public sealed class FaceSample
{
    private readonly List<SampleEmbedding> _embeddings = [];

    public int Id { get; private set; }
    public int EmployeeId { get; private set; }
    public string? ImagePath { get; private set; }
    public SampleSource Source { get; private set; }
    public DateTimeOffset CreatedAt { get; private set; }
    public IReadOnlyCollection<SampleEmbedding> Embeddings => _embeddings;

    private FaceSample() { }

    public static FaceSample Create(
        int employeeId, string? imagePath, SampleSource source, DateTimeOffset now, string model, float[] embedding)
    {
        var sample = new FaceSample { EmployeeId = employeeId, ImagePath = imagePath, Source = source, CreatedAt = now };
        sample.SetEmbedding(model, embedding);
        return sample;
    }

    /// <summary>Thêm hoặc thay embedding của một model (dùng khi đổi model và tính lại).</summary>
    public void SetEmbedding(string model, float[] embedding)
    {
        _embeddings.RemoveAll(e => e.Model == model);
        _embeddings.Add(new SampleEmbedding(model, embedding));
    }
}

/// <summary>Value object: không có Id riêng, chỉ có ý nghĩa khi gắn với một FaceSample.</summary>
public sealed record SampleEmbedding(string Model, float[] Vector);
