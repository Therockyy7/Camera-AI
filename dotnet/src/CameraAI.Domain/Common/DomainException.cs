namespace CameraAI.Domain.Common;

/// <summary>
/// Lỗi vi phạm quy tắc nghiệp vụ (vd: thiếu mã nhân viên). Tầng Web sẽ dịch nó thành HTTP 400.
/// Domain chỉ "ném" lỗi nghiệp vụ, không biết HTTP là gì.
/// </summary>
public class DomainException(string message) : Exception(message);

/// <summary>Mã nhân viên đã tồn tại → tầng Web dịch thành HTTP 409.</summary>
public sealed class DuplicateEmployeeException(string code)
    : DomainException($"Mã nhân viên {code} đã tồn tại");
