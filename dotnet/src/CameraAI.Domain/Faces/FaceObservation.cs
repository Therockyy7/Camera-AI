namespace CameraAI.Domain.Faces;

/// <summary>Lý do một khuôn mặt trên frame không dùng được để nhận diện.</summary>
public enum RejectReason
{
    Small,   // mặt quá nhỏ → "Lại gần hơn"
    Edge,    // chạm mép khung hình
    Turned,  // quay nghiêng quá
    Blurry,  // ảnh mờ
}

/// <summary>Kết quả so khớp một embedding với gallery.</summary>
/// <param name="EmployeeId">Người khớp (điểm ≥ ngưỡng), null nếu không ai đạt.</param>
/// <param name="Score">Cosine cao nhất, kể cả khi dưới ngưỡng.</param>
/// <param name="BestEmployeeId">Người gần nhất dù dưới ngưỡng (để ghi log "người lạ giống ai").</param>
public sealed record FaceMatch(int? EmployeeId, double Score, int? BestEmployeeId)
{
    public static readonly FaceMatch None = new(null, 0.0, null);
}

/// <summary>
/// Kết quả phân tích MỘT khuôn mặt trên MỘT frame. Tầng Infrastructure (OpenCV/ONNX) tạo ra nó,
/// tầng Domain (tracker, decider) chỉ đọc nó → Domain không cần biết model AI nào đã chạy.
/// </summary>
public sealed record FaceObservation
{
    public required FaceBox Box { get; init; }
    public required double DetScore { get; init; }
    public double Sharpness { get; init; }
    public double Turn { get; init; }               // 0 = nhìn thẳng, ≥ 0.5 = quay hẳn sang bên
    public RejectReason? RejectReason { get; init; }
    public double? Liveness { get; init; }          // xác suất mặt thật; null = chưa kiểm
    public bool? IsReal { get; init; }              // null = chưa kiểm (tắt, hoặc mặt nhỏ/nghiêng)
    public float[]? Embedding { get; init; }
    public FaceMatch? Match { get; init; }

    public bool Usable => RejectReason is null;
}
