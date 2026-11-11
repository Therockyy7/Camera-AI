using System.Text;
using CameraAI.Application.Common;
using CameraAI.Application.Features.Employees;
using CameraAI.Application.Features.Enrollment;
using CameraAI.Application.Features.Reports;
using CameraAI.Application.Messaging;
using CameraAI.Application.Pipeline;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Common;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using static CameraAI.Application.Tests.TestHost;

namespace CameraAI.Application.Tests;

public class MediatorTests
{
    private sealed record Ping(string Text) : IRequest<string>;

    private sealed class PingHandler(List<string> log) : IRequestHandler<Ping, string>
    {
        public Task<string> Handle(Ping request, CancellationToken ct) { log.Add("handler"); return Task.FromResult($"pong {request.Text}"); }
    }

    private sealed class Tag(List<string> log, string name) : IPipelineBehavior<Ping, string>
    {
        public async Task<string> Handle(Ping request, Func<Task<string>> next, CancellationToken ct)
        {
            log.Add($"{name}>");
            var result = await next();
            log.Add($"<{name}");
            return result;
        }
    }

    [Fact]
    public async Task Send_routes_to_handler_through_behaviors_like_an_onion()
    {
        var log = new List<string>();
        var services = new ServiceCollection();
        services.AddScoped<ISender>(sp => (ISender)Activator.CreateInstance(
            typeof(IRequest<>).Assembly.GetType("CameraAI.Application.Messaging.Sender")!, sp)!);
        services.AddScoped<IRequestHandler<Ping, string>>(_ => new PingHandler(log));
        services.AddScoped<IPipelineBehavior<Ping, string>>(_ => new Tag(log, "A")); // đăng ký trước → ngoài cùng
        services.AddScoped<IPipelineBehavior<Ping, string>>(_ => new Tag(log, "B"));
        using var sp = services.BuildServiceProvider();

        var result = await sp.CreateScope().ServiceProvider.GetRequiredService<ISender>().Send(new Ping("hi"));

        Assert.Equal("pong hi", result);
        Assert.Equal(["A>", "B>", "handler", "<B", "<A"], log);
    }
}

public class EmployeeFeatureTests
{
    [Fact]
    public async Task Create_then_duplicate_then_reactivate_after_delete()
    {
        using var host = new TestHost();
        var created = await host.Send(new CreateEmployeeCommand(" DEV01 ", "An"));
        Assert.Equal("DEV01", created.Code);
        Assert.Equal(TimeSpan.FromHours(7), created.CreatedAt.Offset); // DTO luôn là giờ VN

        await Assert.ThrowsAsync<DuplicateEmployeeException>(() => host.Send(new CreateEmployeeCommand("DEV01", "Bình")));

        await host.Send(new DeleteEmployeeCommand(created.Id));
        var again = await host.Send(new CreateEmployeeCommand("DEV01", "An mới"));
        Assert.Equal(created.Id, again.Id); // cùng nhân viên → giữ lịch sử
        Assert.Equal("An mới", again.Name);
    }

    [Fact]
    public async Task Create_with_blank_code_is_rejected_by_domain()
    {
        using var host = new TestHost();
        await Assert.ThrowsAsync<DomainException>(() => host.Send(new CreateEmployeeCommand(" ", "An")));
    }

    [Fact]
    public async Task Delete_removes_face_samples_and_files_and_reloads_gallery()
    {
        using var host = new TestHost();
        var employee = host.SeedEmployee("DEV01", "An", axis: 0);
        await host.Get<AttendancePipeline>().PrimeAsync();
        Assert.Equal(1, host.Get<FaceGallery>().Size);

        await host.Send(new DeleteEmployeeCommand(employee.Id));

        Assert.Empty(host.Db.Samples);
        Assert.Equal(0, host.Get<FaceGallery>().Size);
        await Assert.ThrowsAsync<NotFoundException>(() => host.Send(new DeleteEmployeeCommand(employee.Id)));
    }
}

public class PipelineTests
{
    [Fact]
    public async Task Records_attendance_once_after_three_matching_frames()
    {
        using var host = new TestHost();
        var an = host.SeedEmployee("DEV01", "An", axis: 0);
        host.Analyzer.Faces = (_, _) => [Face(Vectors.Unit(0))];
        var pipeline = host.Get<AttendancePipeline>();
        await pipeline.PrimeAsync();

        var decisions = new List<Decision>();
        for (int i = 0; i < 5; i++)
        {
            using var frame = new FakeFrame(i + 1);
            decisions.AddRange(await pipeline.ProcessFrameAsync(frame, host.Get<LocalClock>().Now.AddSeconds(i * 0.125)));
        }

        var decision = Assert.Single(decisions);
        Assert.Equal((DecisionKind.Attendance, an.Id), (decision.Kind, decision.EmployeeId));
        var saved = Assert.Single(host.Db.Attendance);
        Assert.True(host.Files.Files.ContainsKey(saved.SnapshotPath!)); // ảnh chụp đã lưu
        Assert.StartsWith("snapshots/2026-10-26/", saved.SnapshotPath);

        var recent = Assert.Single(pipeline.RecentEvents);
        Assert.Equal(("attendance", "An"), (recent.Kind, recent.Name));
        Assert.NotNull(pipeline.Latest.Jpeg);

        var label = Assert.Single(host.Imaging.LastLabels);
        Assert.Equal(Abstractions.LabelColor.Green, label.Color);
        Assert.Equal("An - đã chấm 08:00", label.Text);
    }

    [Fact]
    public async Task Prime_restores_cooldown_from_database()
    {
        using var host = new TestHost();
        var an = host.SeedEmployee("DEV01", "An", axis: 0);
        var now = host.Get<LocalClock>().Now;
        host.Db.Add(AttendanceEvent.Record(an.Id, "laptop", now.AddMinutes(-2), 0.8, 0.9, null)); // chấm 2 phút trước
        host.Analyzer.Faces = (_, _) => [Face(Vectors.Unit(0))];
        var pipeline = host.Get<AttendancePipeline>();
        await pipeline.PrimeAsync();

        for (int i = 0; i < 5; i++)
            await pipeline.ProcessFrameAsync(new FakeFrame(i + 1), now.AddSeconds(i * 0.125));

        Assert.Single(host.Db.Attendance); // không ghi thêm
    }

    [Fact]
    public async Task Stranger_is_recorded_as_rejected_unknown()
    {
        using var host = new TestHost();
        host.SeedEmployee("DEV01", "An", axis: 0);
        host.Analyzer.Faces = (_, _) => [Face(Vectors.Unit(5))]; // không giống ai
        var pipeline = host.Get<AttendancePipeline>();
        await pipeline.PrimeAsync();
        var now = host.Get<LocalClock>().Now;

        for (int i = 0; i < 6; i++)
            await pipeline.ProcessFrameAsync(new FakeFrame(i + 1), now.AddSeconds(i * 0.125));

        Assert.Equal(DecisionKind.Unknown, Assert.Single(host.Db.Rejected).Reason);
        Assert.Equal("Người lạ", Assert.Single(host.Imaging.LastLabels).Text);
    }

    [Fact]
    public async Task Retention_clears_old_snapshots_but_keeps_attendance()
    {
        using var host = new TestHost();
        var an = host.SeedEmployee("DEV01", "An", axis: 0);
        var now = host.Get<LocalClock>().Now;
        var oldPath = await host.Files.SaveAsync("snapshots/old", [1]);
        var newPath = await host.Files.SaveAsync("snapshots/new", [1]);
        host.Db.Add(AttendanceEvent.Record(an.Id, "laptop", now.AddDays(-40), 0.8, null, oldPath));
        host.Db.Add(AttendanceEvent.Record(an.Id, "laptop", now.AddDays(-1), 0.8, null, newPath));
        host.Db.Add(RejectedEvent.Record("laptop", now.AddDays(-40), DecisionKind.Spoof, null, 0.1, 0.1, null));

        await using var scope = host.Services.CreateAsyncScope();
        var removed = await scope.ServiceProvider.GetRequiredService<SnapshotRetention>().CleanupAsync();

        Assert.Equal(1, removed);
        Assert.Equal(2, host.Db.Attendance.Count);                // lần chấm vẫn còn
        Assert.Null(host.Db.Attendance[0].SnapshotPath);          // chỉ mất ảnh
        Assert.Empty(host.Db.Rejected);
        Assert.Equal([newPath], host.Files.Files.Keys);
    }
}

public class EnrollmentTests
{
    [Fact]
    public async Task Capture_saves_distinct_samples_and_reloads_gallery()
    {
        using var host = new TestHost();
        var e = await host.Send(new CreateEmployeeCommand("DEV01", "An"));
        var rng = new Random(0);
        host.Analyzer.Faces = (_, _) => [Face(Vectors.Random(rng))]; // mỗi frame một góc mặt khác

        var result = await host.Send(new CaptureSamplesCommand(e.Id));

        Assert.Equal(3, result.Saved);
        Assert.Equal(3, host.Db.Samples.Count);
        Assert.Equal(3, host.Files.Files.Count);
        Assert.Equal(3, host.Get<FaceGallery>().Size);
    }

    [Fact]
    public async Task Capture_skips_samples_too_similar_to_existing()
    {
        using var host = new TestHost(o => o.Enrollment.CaptureTimeoutSeconds = 5);
        host.Time.AutoAdvanceAmount = TimeSpan.FromMilliseconds(100); // mỗi lần đọc đồng hồ trôi 100 ms
        var e = await host.Send(new CreateEmployeeCommand("DEV01", "An"));
        host.Analyzer.Faces = (_, _) => [Face(Vectors.Unit(0))]; // đứng im → mặt y hệt

        var result = await host.Send(new CaptureSamplesCommand(e.Id));

        Assert.Equal(1, result.Saved);
        Assert.Contains(result.Reasons, r => r.Code == "similar" && r.Count > 0);
    }

    [Fact]
    public async Task Capture_without_camera_reports_no_camera()
    {
        using var host = new TestHost();
        host.Time.AutoAdvanceAmount = TimeSpan.FromSeconds(1);
        host.Source.Empty = true;
        var e = await host.Send(new CreateEmployeeCommand("DEV01", "An"));

        var result = await host.Send(new CaptureSamplesCommand(e.Id));

        Assert.Equal((0, 0), (result.Saved, result.Attempts));
        Assert.Equal("no_camera", Assert.Single(result.Reasons).Code);
    }

    [Fact]
    public async Task Upload_reports_reason_per_file()
    {
        using var host = new TestHost();
        var e = await host.Send(new CreateEmployeeCommand("DEV01", "An"));
        host.Analyzer.Faces = (_, _) => [Face(Vectors.Unit(0)), Face(Vectors.Unit(1), new FaceBox(300, 100, 120, 120))];

        var results = await host.Send(new UploadSamplesCommand(e.Id, [
            new("a.txt", Encoding.UTF8.GetBytes("không phải ảnh")),
            new("b.jpg", [0xFF, 0xD8]),
        ]));

        Assert.Equal([
            new UploadResult("a.txt", false, "Không đọc được file ảnh"),
            new UploadResult("b.jpg", false, "Có nhiều hơn một khuôn mặt"),
        ], results);
    }

    [Fact]
    public async Task Upload_ignores_small_background_faces()
    {
        using var host = new TestHost();
        var e = await host.Send(new CreateEmployeeCommand("DEV01", "An"));
        host.Analyzer.Faces = (_, _) => [Face(Vectors.Unit(0)), Face(Vectors.Unit(1), reject: RejectReason.Small)];

        var results = await host.Send(new UploadSamplesCommand(e.Id, [new("b.jpg", [0xFF, 0xD8])]));

        Assert.True(Assert.Single(results).Ok);
        Assert.Equal(Domain.Employees.SampleSource.Upload, Assert.Single(host.Db.Samples).Source);
    }

    [Fact]
    public async Task Unknown_employee_is_not_found()
    {
        using var host = new TestHost();
        await Assert.ThrowsAsync<NotFoundException>(() => host.Send(new CaptureSamplesCommand(99)));
    }
}

public class ReportTests
{
    [Fact]
    public async Task Csv_has_bom_header_and_rows()
    {
        using var host = new TestHost();
        var an = host.SeedEmployee("DEV01", "An, Nguyễn", axis: 0);
        var now = host.Get<LocalClock>().Now.TrimToSeconds();
        host.Db.Add(AttendanceEvent.Record(an.Id, "laptop", now, 0.8, null, null));

        var file = await host.Send(new ExportAttendanceCsvQuery(null, null));

        Assert.Equal("cham-cong_2026-10-26_2026-10-26.csv", file.FileName);
        Assert.Equal([0xEF, 0xBB, 0xBF], file.Content[..3]);
        var lines = Encoding.UTF8.GetString(file.Content[3..]).Split("\r\n", StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal("Mã NV,Tên,Thời điểm,Camera,Điểm khớp,Điểm mặt thật", lines[0]);
        Assert.Equal("DEV01,\"An, Nguyễn\",2026-10-26T08:00:00+07:00,laptop,0.800,", lines[1]);
    }

    [Fact]
    public async Task Csv_rejects_reversed_range()
    {
        using var host = new TestHost();
        await Assert.ThrowsAsync<InvalidRequestException>(() =>
            host.Send(new ExportAttendanceCsvQuery(new DateOnly(2026, 10, 26), new DateOnly(2026, 10, 25))));
    }

    [Fact]
    public void Day_range_uses_configured_time_zone()
    {
        var clock = new LocalClock(TimeProvider.System, Options.Create(new CameraAiOptions()));
        var (from, to) = clock.DayRange(new DateOnly(2026, 10, 26), new DateOnly(2026, 10, 26));
        Assert.Equal(new DateTimeOffset(2026, 10, 26, 0, 0, 0, TimeSpan.FromHours(7)), from);
        Assert.Equal(TimeSpan.FromDays(1), to - from);
    }
}
