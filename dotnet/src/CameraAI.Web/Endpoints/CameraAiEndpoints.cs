using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Features.Employees;
using CameraAI.Application.Features.Enrollment;
using CameraAI.Application.Features.Reports;
using CameraAI.Application.Messaging;
using CameraAI.Application.Pipeline;

namespace CameraAI.Web.Endpoints;

public sealed record CreateEmployeeRequest(string Code, string Name);

/// <summary>
/// MINIMAL API: mỗi endpoint chỉ làm 3 việc — đọc tham số HTTP → tạo Command/Query → gửi qua ISender (Mediator).
/// Không có logic nghiệp vụ ở đây; tầng Web là "lớp vỏ HTTP" mỏng. Đường dẫn + JSON giữ y hệt bản Python
/// nên giao diện cũ (wwwroot) chạy không sửa dòng nào.
///
/// Tham số được ASP.NET Core tự BIND: {id:int} từ route, DateOnly? date từ query string, CreateEmployeeRequest từ
/// body JSON, ISender / dịch vụ khác từ DI, CancellationToken = huỷ khi trình duyệt đóng kết nối.
/// </summary>
public static class CameraAiEndpoints
{
    public static void MapCameraAi(this IEndpointRouteBuilder app)
    {
        app.MapGet("/", (HttpContext ctx, IWebHostEnvironment env) =>
        {
            ctx.Response.Headers.CacheControl = "no-store";
            return Results.File(env.WebRootFileProvider.GetFileInfo("index.html").PhysicalPath!, "text/html; charset=utf-8");
        });
        app.MapGet("/stream.mjpg", MjpegStream.WriteAsync);
        app.MapGet("/files/{**path}", (string path, IFileStore files) =>
            files.ResolveFullPath(path) is { } full ? Results.File(full, "image/jpeg") : throw new NotFoundException("Không tìm thấy file"));

        var api = app.MapGroup("/api");

        api.MapGet("/status", (ISender s, CancellationToken ct) => s.Send(new GetStatusQuery(), ct));

        // ---------- nhân viên ----------
        api.MapGet("/employees", (ISender s, CancellationToken ct) => s.Send(new ListEmployeesQuery(), ct));
        api.MapPost("/employees", async (CreateEmployeeRequest body, ISender s, CancellationToken ct) =>
        {
            var created = await s.Send(new CreateEmployeeCommand(body.Code, body.Name), ct);
            return Results.Created($"/api/employees/{created.Id}", created);
        });
        api.MapDelete("/employees/{id:int}", async (int id, ISender s, CancellationToken ct) =>
            Results.Ok(new { Ok = await s.Send(new DeleteEmployeeCommand(id), ct) }));

        // ---------- mẫu khuôn mặt ----------
        api.MapGet("/employees/{id:int}/samples", (int id, ISender s, CancellationToken ct) => s.Send(new ListSamplesQuery(id), ct));
        api.MapPost("/employees/{id:int}/capture", (int id, ISender s, CancellationToken ct) => s.Send(new CaptureSamplesCommand(id), ct));
        api.MapPost("/employees/{id:int}/upload", async (int id, IFormFileCollection files, ISender s, CancellationToken ct) =>
        {
            var images = new List<UploadedImage>();
            foreach (var file in files)
            {
                using var buffer = new MemoryStream();
                await file.CopyToAsync(buffer, ct);
                images.Add(new UploadedImage(file.FileName, buffer.ToArray()));
            }
            return await s.Send(new UploadSamplesCommand(id, images), ct);
        }).DisableAntiforgery(); // API dùng Basic Auth, không có form HTML do server render → không cần anti-forgery token
        api.MapDelete("/samples/{id:int}", async (int id, ISender s, CancellationToken ct) =>
            Results.Ok(new { Ok = await s.Send(new DeleteSampleCommand(id), ct) }));

        // ---------- lịch sử ----------
        api.MapGet("/attendance", (DateOnly? date, ISender s, CancellationToken ct) => s.Send(new GetAttendanceDayQuery(date), ct));
        api.MapGet("/attendance.csv", async (DateOnly? start, DateOnly? end, ISender s, CancellationToken ct) =>
        {
            var csv = await s.Send(new ExportAttendanceCsvQuery(start, end), ct);
            return Results.File(csv.Content, "text/csv; charset=utf-8", csv.FileName);
        });
        api.MapGet("/rejected", (DateOnly? date, ISender s, CancellationToken ct) => s.Send(new GetRejectedDayQuery(date), ct));
    }
}

/// <summary>
/// Luồng MJPEG: MỘT response HTTP không bao giờ kết thúc, chứa liên tiếp các ảnh JPEG ngăn cách bởi "--frame".
/// Thẻ &lt;img src="/stream.mjpg"&gt; tự hiển thị thành video — không cần JavaScript hay WebSocket.
/// </summary>
internal static class MjpegStream
{
    public static async Task WriteAsync(HttpContext ctx, AttendancePipeline pipeline, IHostApplicationLifetime lifetime)
    {
        ctx.Response.ContentType = "multipart/x-mixed-replace; boundary=frame";
        ctx.Response.Headers.CacheControl = "no-store";
        // Dừng khi trình duyệt đóng tab HOẶC app tắt. Thiếu ApplicationStopping thì Ctrl+C phải đợi request
        // "vô tận" này hết hạn (mặc định 30 giây) mới thoát được.
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted, lifetime.ApplicationStopping);
        var ct = stop.Token;
        long lastId = -1;
        try
        {
            while (!ct.IsCancellationRequested)
            {
                var (id, jpeg) = pipeline.Latest;
                if (jpeg is null || id == lastId)
                {
                    await Task.Delay(30, ct);
                    continue;
                }
                lastId = id;
                var header = System.Text.Encoding.ASCII.GetBytes(
                    $"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {jpeg.Length}\r\n\r\n");
                await ctx.Response.Body.WriteAsync(header, ct);
                await ctx.Response.Body.WriteAsync(jpeg, ct);
                await ctx.Response.Body.WriteAsync("\r\n"u8.ToArray(), ct);
                await ctx.Response.Body.FlushAsync(ct);
            }
        }
        catch (OperationCanceledException)
        {
            // người xem đóng tab / app tắt — kết thúc bình thường
        }
    }
}
