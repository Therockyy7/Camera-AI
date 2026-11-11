namespace CameraAI.Domain.Faces;

/// <summary>
/// Khung mặt (x, y, rộng, cao) theo toạ độ frame gốc.
/// <c>readonly record struct</c> = value object: so sánh theo giá trị, bất biến, nằm trên stack (không cấp phát heap).
/// </summary>
public readonly record struct FaceBox(int X, int Y, int Width, int Height)
{
    public int MinSide => Math.Min(Width, Height);
    public int Right => X + Width;
    public int Bottom => Y + Height;

    /// <summary>Intersection over Union: 1 = trùng khít, 0 = không chạm nhau. Dùng để bám mặt qua các frame.</summary>
    public double Iou(FaceBox other)
    {
        int ix = Math.Max(0, Math.Min(Right, other.Right) - Math.Max(X, other.X));
        int iy = Math.Max(0, Math.Min(Bottom, other.Bottom) - Math.Max(Y, other.Y));
        double inter = (double)ix * iy;
        double union = (double)Width * Height + (double)other.Width * other.Height - inter;
        return union > 0 ? inter / union : 0.0;
    }

    /// <summary>Mặt chạm/tràn mép khung hình → điểm mốc lệch, crop chống giả sai → bỏ qua frame này.</summary>
    public bool TouchesEdge(int frameWidth, int frameHeight, int margin = 5) =>
        Math.Min(Math.Min(X, Y), Math.Min(frameWidth - Right, frameHeight - Bottom)) < margin;
}
