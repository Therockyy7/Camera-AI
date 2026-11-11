using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Faces;
using CameraAI.Application.Messaging;
using CameraAI.Domain.Employees;

namespace CameraAI.Application.Features.Employees;

// Mỗi use case = 1 cặp (Request, Handler) đặt cạnh nhau — "vertical slice": muốn hiểu "thêm nhân viên"
// làm gì thì đọc đúng một chỗ, không phải lần qua Controller → Service → Manager...

// ---------- QUERY: danh sách nhân viên ----------
public sealed record ListEmployeesQuery : IRequest<List<EmployeeDto>>;

public sealed class ListEmployeesHandler(IReportQueries queries) : IRequestHandler<ListEmployeesQuery, List<EmployeeDto>>
{
    public Task<List<EmployeeDto>> Handle(ListEmployeesQuery request, CancellationToken ct) => queries.EmployeesAsync(ct);
}

// ---------- COMMAND: thêm nhân viên ----------
public sealed record CreateEmployeeCommand(string Code, string Name) : IRequest<EmployeeDto>;

public sealed class CreateEmployeeHandler(IEmployeeRepository employees, IUnitOfWork uow, LocalClock clock)
    : IRequestHandler<CreateEmployeeCommand, EmployeeDto>
{
    public async Task<EmployeeDto> Handle(CreateEmployeeCommand request, CancellationToken ct)
    {
        var existing = await employees.FindByCodeAsync(request.Code?.Trim() ?? "", ct);
        Employee employee;
        if (existing is null)
        {
            employee = Employee.Create(request.Code ?? "", request.Name ?? "", clock.Now.TrimToSeconds()); // validate trong Domain
            employees.Add(employee);
        }
        else
        {
            existing.Reactivate(request.Name ?? ""); // đang hoạt động → Domain ném DuplicateEmployeeException
            employee = existing;
        }
        await uow.SaveChangesAsync(ct); // EF gán Id ở đây
        return new EmployeeDto(employee.Id, employee.Code, employee.Name, clock.ToLocal(employee.CreatedAt), SampleCount: 0);
    }
}

// ---------- COMMAND: xoá (ngừng) nhân viên ----------
public sealed record DeleteEmployeeCommand(int Id) : IRequest<bool>;

public sealed class DeleteEmployeeHandler(
    IEmployeeRepository employees, IFaceSampleRepository samples, IUnitOfWork uow, IFileStore files, GalleryLoader gallery)
    : IRequestHandler<DeleteEmployeeCommand, bool>
{
    /// <summary>Giữ lịch sử chấm công, xoá VĨNH VIỄN mẫu khuôn mặt (dữ liệu sinh trắc — NĐ 13/2023).</summary>
    public async Task<bool> Handle(DeleteEmployeeCommand request, CancellationToken ct)
    {
        var employee = await employees.GetActiveAsync(request.Id, ct) ?? throw new NotFoundException("Không tìm thấy nhân viên");
        employee.Deactivate();
        var faces = await samples.ListByEmployeeAsync(employee.Id, ct);
        foreach (var sample in faces) samples.Remove(sample);
        await uow.SaveChangesAsync(ct); // nhân viên + mẫu: MỘT giao dịch

        foreach (var sample in faces) files.Delete(sample.ImagePath);
        await gallery.ReloadAsync(ct); // camera thôi nhận người này ngay
        return true;
    }
}
