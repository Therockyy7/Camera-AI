namespace CameraAI.Domain.Faces;

/// <summary>
/// Cổng lọc NHẬN DIỆN (nới): mặt phải đủ to, không chạm mép, không quá nghiêng, không mờ.
/// Là record nên tạo biến thể dễ dàng: đăng ký mẫu dùng <c>gate with { MinFacePx = 80 }</c>.
/// </summary>
public sealed record RecognitionGate(int MinFacePx = 40, double MaxTurn = 0.45, double MinSharpness = 12)
{
    /// <returns>null = dùng được; ngược lại là lý do loại.</returns>
    public RejectReason? Evaluate(FaceBox box, double turn, double sharpness, int frameWidth, int frameHeight)
    {
        if (box.MinSide < MinFacePx) return RejectReason.Small;
        if (box.TouchesEdge(frameWidth, frameHeight)) return RejectReason.Edge;
        if (turn > MaxTurn) return RejectReason.Turned;
        if (sharpness < MinSharpness) return RejectReason.Blurry;
        return null;
    }

    /// <summary>
    /// Độ quay ngang từ 3 điểm mốc: mũi nằm giữa hai mắt → 0; lệch hẳn về một mắt → 0.5.
    /// (Mắt phải của người nằm bên TRÁI ảnh.)
    /// </summary>
    public static double TurnOffset(float rightEyeX, float leftEyeX, float noseX)
    {
        float span = leftEyeX - rightEyeX;
        if (span <= 0) return 1.0;
        return Math.Abs((noseX - rightEyeX) / span - 0.5);
    }
}

/// <summary>
/// Cổng lọc CHỐNG GIẢ (chặt hơn): model MiniFASNet chỉ tin cậy khi mặt ≥ 64 px và gần thẳng.
/// Mặt qua cổng nhận diện nhưng không qua cổng này → Liveness = null ("chưa kiểm").
/// </summary>
public sealed record LivenessGate(int MinFacePx = 64, double MaxTurn = 0.25, double RealThreshold = 0.5)
{
    public bool CanCheck(FaceBox box, double turn) => box.MinSide >= MinFacePx && turn <= MaxTurn;

    public bool IsReal(double probability) => probability >= RealThreshold;
}
