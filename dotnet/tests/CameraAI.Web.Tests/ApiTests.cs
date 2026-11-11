using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.Options;
using OpenCvSharp;

namespace CameraAI.Web.Tests;

public class ApiTests
{
    private static async Task<JsonElement> Json(HttpResponseMessage response) =>
        JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement;

    [Fact]
    public async Task Home_page_is_the_old_ui_and_not_cached()
    {
        await using var app = new CameraAiApp();
        var client = app.CreateClient();

        var home = await client.GetAsync("/");
        Assert.Equal("no-store", home.Headers.CacheControl?.ToString());
        Assert.Contains("/static/app.js", await home.Content.ReadAsStringAsync());
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/static/app.js")).StatusCode);
    }

    [Fact]
    public async Task Create_employee_returns_201_with_snake_case_json()
    {
        await using var app = new CameraAiApp();
        var client = app.CreateClient();

        var response = await client.PostAsJsonAsync("/api/employees", new { code = "DEV01", name = "An" });

        Assert.Equal(HttpStatusCode.Created, response.StatusCode);
        var body = await Json(response);
        Assert.Equal(0, body.GetProperty("sample_count").GetInt32());              // snake_case như bản Python
        Assert.EndsWith("+07:00", body.GetProperty("created_at").GetString());     // giờ VN
        var list = await Json(await client.GetAsync("/api/employees"));
        Assert.Equal("DEV01", list[0].GetProperty("code").GetString());
    }

    [Theory]
    [InlineData("DEV01", "Bình", HttpStatusCode.Conflict, "Mã nhân viên DEV01 đã tồn tại")]
    [InlineData(" ", "Bình", HttpStatusCode.BadRequest, "Cần nhập mã và tên nhân viên")]
    public async Task Business_errors_become_problem_details(string code, string name, HttpStatusCode status, string detail)
    {
        await using var app = new CameraAiApp();
        var client = app.CreateClient();
        await client.PostAsJsonAsync("/api/employees", new { code = "DEV01", name = "An" });

        var response = await client.PostAsJsonAsync("/api/employees", new { code, name });

        Assert.Equal(status, response.StatusCode);
        Assert.Equal(detail, (await Json(response)).GetProperty("detail").GetString()); // app.js hiện đúng trường này
    }

    [Fact]
    public async Task Not_found_and_bad_date_map_to_404_and_400()
    {
        await using var app = new CameraAiApp();
        var client = app.CreateClient();

        Assert.Equal(HttpStatusCode.NotFound, (await client.DeleteAsync("/api/employees/999")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync("/files/../cameraai.db")).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await client.GetAsync("/api/attendance?date=26-10-2026")).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await client.GetAsync("/api/attendance.csv?start=2026-10-26&end=2026-10-25")).StatusCode);
    }

    [Fact]
    public async Task Status_reports_camera_and_settings()
    {
        await using var app = new CameraAiApp();
        var client = app.CreateClient();
        await client.PostAsJsonAsync("/api/employees", new { code = "DEV01", name = "An" });

        var status = await Json(await client.GetAsync("/api/status"));

        Assert.False(status.GetProperty("camera").GetProperty("connected").GetBoolean());
        Assert.Equal("opportunistic", status.GetProperty("liveness_mode").GetString());
        Assert.Equal("sface", status.GetProperty("recognition_model").GetString());
        Assert.Equal(1, status.GetProperty("employees").GetInt32());
        Assert.Equal(0, status.GetProperty("recent").GetArrayLength());
    }

    [Fact]
    public async Task History_and_csv_for_a_day()
    {
        await using var app = new CameraAiApp();
        var client = app.CreateClient();

        var day = await Json(await client.GetAsync("/api/attendance?date=2026-10-26"));
        Assert.Equal("2026-10-26", day.GetProperty("date").GetString());
        Assert.Equal(0, day.GetProperty("events").GetArrayLength());

        var csv = await client.GetAsync("/api/attendance.csv?start=2026-10-01&end=2026-10-31");
        Assert.Equal("cham-cong_2026-10-01_2026-10-31.csv", csv.Content.Headers.ContentDisposition?.FileNameStar
            ?? csv.Content.Headers.ContentDisposition?.FileName);
        var bytes = await csv.Content.ReadAsByteArrayAsync();
        Assert.Equal([0xEF, 0xBB, 0xBF], bytes[..3]);
    }

    [Fact]
    public async Task Upload_reports_result_per_file()
    {
        await using var app = new CameraAiApp();
        var client = app.CreateClient();
        var id = (await Json(await client.PostAsJsonAsync("/api/employees", new { code = "DEV01", name = "An" })))
            .GetProperty("id").GetInt32();

        Cv2.ImEncode(".jpg", new Mat(240, 320, MatType.CV_8UC3, Scalar.All(127)), out var jpeg); // ảnh thật, không có mặt
        using var form = new MultipartFormDataContent
        {
            { new ByteArrayContent(Encoding.UTF8.GetBytes("không phải ảnh")), "files", "a.txt" },
            { new ByteArrayContent(jpeg), "files", "b.jpg" },
        };
        var results = await Json(await client.PostAsync($"/api/employees/{id}/upload", form));

        Assert.Equal("Không đọc được file ảnh", results[0].GetProperty("message").GetString());
        Assert.Equal("Không thấy khuôn mặt", results[1].GetProperty("message").GetString());
        Assert.False(results[1].GetProperty("ok").GetBoolean());
    }

    [Fact]
    public async Task Admin_password_protects_everything()
    {
        await using var app = new CameraAiApp(new() { ["CameraAI:Web:AdminPassword"] = "bí-mật" });
        var client = app.CreateClient();

        var denied = await client.GetAsync("/api/status");
        Assert.Equal(HttpStatusCode.Unauthorized, denied.StatusCode);
        Assert.Equal("Basic", denied.Headers.WwwAuthenticate.Single().Scheme);

        client.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Basic", Convert.ToBase64String(Encoding.UTF8.GetBytes("ai-cung-duoc:bí-mật")));
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/api/status")).StatusCode);
    }

    [Fact]
    public async Task Listening_on_lan_without_password_refuses_to_start()
    {
        await using var app = new CameraAiApp(new() { ["CameraAI:Web:Host"] = "0.0.0.0" });
        var ex = Assert.ThrowsAny<OptionsValidationException>(() => app.CreateClient());
        Assert.Contains("ADMIN_PASSWORD", ex.Message);
    }
}
