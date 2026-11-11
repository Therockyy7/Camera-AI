using CameraAI.Application.Common;
using CameraAI.Domain.Common;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.Mvc;

namespace CameraAI.Web.Http;

/// <summary>
/// DỊCH lỗi nghiệp vụ → mã HTTP, tại MỘT chỗ. Domain/Application chỉ ném exception mang nghĩa nghiệp vụ
/// (trùng mã, không tìm thấy…) mà không biết HTTP là gì; chỉ tầng Web biết "trùng mã = 409".
///
/// Trả về ProblemDetails (RFC 9457): { "status": 409, "title": "...", "detail": "Mã nhân viên DEV01 đã tồn tại" }.
/// Giao diện web cũ đọc đúng trường "detail" để hiện thông báo (FastAPI cũng trả "detail").
/// </summary>
internal sealed class ApiExceptionHandler(IProblemDetailsService problems, ILogger<ApiExceptionHandler> log) : IExceptionHandler
{
    public async ValueTask<bool> TryHandleAsync(HttpContext context, Exception exception, CancellationToken ct)
    {
        (int Status, string Title)? mapped = exception switch
        {
            DuplicateEmployeeException => (StatusCodes.Status409Conflict, "Trùng dữ liệu"),
            DomainException => (StatusCodes.Status400BadRequest, "Dữ liệu không hợp lệ"),
            InvalidRequestException => (StatusCodes.Status400BadRequest, "Tham số không hợp lệ"),
            NotFoundException => (StatusCodes.Status404NotFound, "Không tìm thấy"),
            BadHttpRequestException bad => (bad.StatusCode, "Request không hợp lệ"), // vd ?date=abc
            _ => null,
        };
        if (mapped is not { } m)
        {
            log.LogError(exception, "Lỗi không mong đợi khi xử lý {Path}", context.Request.Path);
            return false; // để mặc định trả 500, không lộ chi tiết lỗi ra ngoài
        }

        context.Response.StatusCode = m.Status;
        return await problems.TryWriteAsync(new ProblemDetailsContext
        {
            HttpContext = context,
            Exception = exception,
            ProblemDetails = new ProblemDetails { Status = m.Status, Title = m.Title, Detail = exception.Message },
        });
    }
}
