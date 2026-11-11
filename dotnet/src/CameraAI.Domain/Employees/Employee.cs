using CameraAI.Domain.Common;

namespace CameraAI.Domain.Employees;

/// <summary>
/// Entity Nhân viên. Đây là "rich domain model": setter để private, muốn đổi trạng thái phải gọi method
/// có tên nghiệp vụ (Create / Reactivate / Deactivate) → quy tắc nằm một chỗ, không ai gán bậy Active = false.
/// </summary>
public sealed class Employee
{
    public int Id { get; private set; }
    public string Code { get; private set; } = "";
    public string Name { get; private set; } = "";
    public bool Active { get; private set; }
    public DateTimeOffset CreatedAt { get; private set; }

    private Employee() { } // cho EF Core dựng object khi đọc DB

    public static Employee Create(string code, string name, DateTimeOffset now)
    {
        var (c, n) = Validate(code, name);
        return new Employee { Code = c, Name = n, Active = true, CreatedAt = now };
    }

    /// <summary>Mã từng bị xoá được dùng lại: kích hoạt lại, giữ lịch sử chấm công cũ.</summary>
    public void Reactivate(string name)
    {
        if (Active) throw new DuplicateEmployeeException(Code);
        Name = Validate(Code, name).Name;
        Active = true;
    }

    /// <summary>Ngừng chấm công. Trả về false nếu đã ngừng từ trước.</summary>
    public bool Deactivate()
    {
        if (!Active) return false;
        Active = false;
        return true;
    }

    private static (string Code, string Name) Validate(string code, string name)
    {
        code = code?.Trim() ?? "";
        name = name?.Trim() ?? "";
        if (code.Length == 0 || name.Length == 0)
            throw new DomainException("Cần nhập mã và tên nhân viên");
        return (code, name);
    }
}
