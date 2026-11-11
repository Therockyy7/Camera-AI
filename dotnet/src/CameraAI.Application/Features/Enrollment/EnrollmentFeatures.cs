using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Enrollment;
using CameraAI.Application.Faces;
using CameraAI.Application.Messaging;

namespace CameraAI.Application.Features.Enrollment;

// ---------- QUERY: các mẫu của một nhân viên ----------
public sealed record ListSamplesQuery(int EmployeeId) : IRequest<List<SampleDto>>;

public sealed class ListSamplesHandler(IEmployeeRepository employees, IReportQueries queries)
    : IRequestHandler<ListSamplesQuery, List<SampleDto>>
{
    public async Task<List<SampleDto>> Handle(ListSamplesQuery request, CancellationToken ct)
    {
        _ = await employees.GetActiveAsync(request.EmployeeId, ct) ?? throw new NotFoundException("Không tìm thấy nhân viên");
        return await queries.SamplesAsync(request.EmployeeId, ct);
    }
}

// ---------- COMMAND: xoá một mẫu ----------
public sealed record DeleteSampleCommand(int SampleId) : IRequest<bool>;

public sealed class DeleteSampleHandler(IFaceSampleRepository samples, IUnitOfWork uow, IFileStore files, GalleryLoader gallery)
    : IRequestHandler<DeleteSampleCommand, bool>
{
    public async Task<bool> Handle(DeleteSampleCommand request, CancellationToken ct)
    {
        var sample = await samples.GetAsync(request.SampleId, ct) ?? throw new NotFoundException("Không tìm thấy mẫu");
        samples.Remove(sample);
        await uow.SaveChangesAsync(ct);
        files.Delete(sample.ImagePath);
        await gallery.ReloadAsync(ct);
        return true;
    }
}

// ---------- COMMAND: chụp mẫu từ camera ----------
public sealed record CaptureSamplesCommand(int EmployeeId) : IRequest<CaptureResult>;

public sealed class CaptureSamplesHandler(FaceEnrollmentService enrollment, GalleryLoader gallery)
    : IRequestHandler<CaptureSamplesCommand, CaptureResult>
{
    public async Task<CaptureResult> Handle(CaptureSamplesCommand request, CancellationToken ct)
    {
        var result = await enrollment.CaptureAsync(request.EmployeeId, ct);
        if (result.Saved > 0) await gallery.ReloadAsync(ct);
        return result;
    }
}

// ---------- COMMAND: tải ảnh mẫu lên ----------
public sealed record UploadedImage(string FileName, byte[] Data);

public sealed record UploadResult(string Filename, bool Ok, string Message);

public sealed record UploadSamplesCommand(int EmployeeId, IReadOnlyList<UploadedImage> Files) : IRequest<List<UploadResult>>;

public sealed class UploadSamplesHandler(FaceEnrollmentService enrollment, GalleryLoader gallery)
    : IRequestHandler<UploadSamplesCommand, List<UploadResult>>
{
    public async Task<List<UploadResult>> Handle(UploadSamplesCommand request, CancellationToken ct)
    {
        var results = new List<UploadResult>();
        foreach (var file in request.Files)
        {
            var issue = await enrollment.AddUploadAsync(request.EmployeeId, file.Data, ct);
            results.Add(new UploadResult(file.FileName, issue is null, issue?.Text() ?? "Đã lưu mẫu"));
        }
        if (results.Any(r => r.Ok)) await gallery.ReloadAsync(ct); // nạp lại MỘT lần cho cả lô
        return results;
    }
}
