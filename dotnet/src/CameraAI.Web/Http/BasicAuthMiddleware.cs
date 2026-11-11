using System.Security.Cryptography;
using System.Text;
using CameraAI.Application.Common;
using Microsoft.Extensions.Options;

namespace CameraAI.Web.Http;

/// <summary>
/// MIDDLEWARE: một mắt xích trong chuỗi xử lý request. Mỗi request đi qua theo thứ tự app.Use...() trong Program.cs;
/// middleware được chọn: cho đi tiếp (<c>next(context)</c>) hay chặn lại trả lời luôn (401).
///
/// HTTP Basic cho MỌI đường dẫn khi có ADMIN_PASSWORD (tên đăng nhập tuỳ ý) — giống bản Python.
/// </summary>
internal sealed class BasicAuthMiddleware(RequestDelegate next, IOptions<CameraAiOptions> options)
{
    public Task InvokeAsync(HttpContext context)
    {
        var password = options.Value.Web.AdminPassword;
        if (string.IsNullOrEmpty(password) || IsAuthorized(context, password)) return next(context);

        context.Response.StatusCode = StatusCodes.Status401Unauthorized;
        context.Response.Headers.WWWAuthenticate = "Basic realm=\"CameraAI\", charset=\"UTF-8\"";
        return context.Response.WriteAsync("Cần đăng nhập");
    }

    private static bool IsAuthorized(HttpContext context, string password)
    {
        string header = context.Request.Headers.Authorization.ToString();
        if (!header.StartsWith("Basic ", StringComparison.OrdinalIgnoreCase)) return false;
        string decoded;
        try
        {
            decoded = Encoding.UTF8.GetString(Convert.FromBase64String(header[6..].Trim()));
        }
        catch (FormatException)
        {
            return false;
        }
        int colon = decoded.IndexOf(':');
        if (colon < 0) return false;
        var given = decoded[(colon + 1)..]; // "tên:mật khẩu" → lấy phần sau dấu ':'
        // So sánh THỜI GIAN CỐ ĐỊNH: không dừng sớm ở ký tự sai đầu tiên → không đoán được mật khẩu qua thời gian phản hồi.
        return CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(given), Encoding.UTF8.GetBytes(password));
    }
}
