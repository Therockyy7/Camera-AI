namespace CameraAI.Application.Common;

/// <summary>Không tìm thấy tài nguyên → tầng Web dịch thành HTTP 404.</summary>
public sealed class NotFoundException(string message) : Exception(message);
