using CameraAI.Application;
using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Features.Employees;
using CameraAI.Application.Messaging;
using CameraAI.Domain.Attendance;
using CameraAI.Domain.Common;
using CameraAI.Domain.Employees;
using CameraAI.Infrastructure.Files;
using CameraAI.Infrastructure.Persistence;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace CameraAI.Infrastructure.Tests;

/// <summary>SQLite THẬT trong thư mục tạm — test cả cấu hình EF (bảng, cột, converter) chứ không chỉ logic.</summary>
public sealed class DbFixture : IAsyncDisposable
{
    public string DataDir { get; } = Path.Combine(Path.GetTempPath(), "cameraai-test-" + Guid.NewGuid().ToString("N")[..8]);
    public ServiceProvider Services { get; }
    public static readonly TimeSpan Vn = TimeSpan.FromHours(7);

    public DbFixture()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.Configure<CameraAiOptions>(o => o.Storage.DataDir = DataDir);
        services.AddApplication();
        services.AddPersistence();
        Services = services.BuildServiceProvider(new ServiceProviderOptions { ValidateScopes = true });
        Services.InitializeDatabaseAsync().GetAwaiter().GetResult();
    }

    /// <summary>Mỗi lần gọi = một scope mới = một DbContext mới, như mỗi HTTP request.</summary>
    public async Task<T> In<T>(Func<IServiceProvider, Task<T>> work)
    {
        await using var scope = Services.CreateAsyncScope();
        return await work(scope.ServiceProvider);
    }

    public Task In(Func<IServiceProvider, Task> work) => In(async sp => { await work(sp); return 0; });

    public static DateTimeOffset At(int hour, int minute = 0, int day = 26) => new(2026, 10, day, hour, minute, 0, Vn);

    public async ValueTask DisposeAsync()
    {
        await Services.DisposeAsync();
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools(); // nhả khoá file DB để xoá được thư mục
        Directory.Delete(DataDir, recursive: true);
    }
}

public class PersistenceTests
{
    private static async Task<int> AddEmployee(DbFixture f, string code, string name) => await f.In(async sp =>
    {
        var e = Employee.Create(code, name, DbFixture.At(8));
        sp.GetRequiredService<IEmployeeRepository>().Add(e);
        await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        return e.Id;
    });

    [Fact]
    public async Task Repository_and_unit_of_work_share_one_DbContext_per_scope()
    {
        await using var f = new DbFixture();
        await f.In(sp =>
        {
            Assert.Same(sp.GetRequiredService<CameraAiDbContext>(), sp.GetRequiredService<IUnitOfWork>());
            return Task.CompletedTask;
        });
        var id = await AddEmployee(f, "DEV01", "An");
        Assert.True(id > 0); // nếu UoW là DbContext khác thì không có gì được lưu, Id vẫn = 0
    }

    [Fact]
    public async Task Sample_embeddings_round_trip_per_model()
    {
        await using var f = new DbFixture();
        var id = await AddEmployee(f, "DEV01", "An");
        var vector = Enumerable.Range(0, 128).Select(i => i / 128f).ToArray();
        await f.In(async sp =>
        {
            sp.GetRequiredService<IFaceSampleRepository>().Add(
                FaceSample.Create(id, "faces/1/a.jpg", SampleSource.Camera, DbFixture.At(8), "sface", vector));
            await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        });

        // Đổi model: mẫu thiếu embedding arcface → tính rồi lưu thêm, embedding sface giữ nguyên.
        await f.In(async sp =>
        {
            var repo = sp.GetRequiredService<IFaceSampleRepository>();
            var missing = Assert.Single(await repo.MissingEmbeddingAsync("arcface"));
            missing.SetEmbedding("arcface", new float[512]);
            await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        });

        await f.In(async sp =>
        {
            var repo = sp.GetRequiredService<IFaceSampleRepository>();
            Assert.Empty(await repo.MissingEmbeddingAsync("arcface"));
            var (owner, loaded) = Assert.Single(await repo.GalleryAsync("sface"));
            Assert.Equal(id, owner);
            Assert.Equal(vector, loaded);
            Assert.Equal(512, Assert.Single(await repo.GalleryAsync("arcface")).Vector.Length);
            var sample = Assert.Single(await repo.ListByEmployeeAsync(id));
            Assert.Equal(SampleSource.Camera, sample.Source);
            Assert.Equal(2, sample.Embeddings.Count);
        });
    }

    [Fact]
    public async Task Removing_sample_cascades_embeddings()
    {
        await using var f = new DbFixture();
        var id = await AddEmployee(f, "DEV01", "An");
        await f.In(async sp =>
        {
            sp.GetRequiredService<IFaceSampleRepository>().Add(
                FaceSample.Create(id, null, SampleSource.Upload, DbFixture.At(8), "sface", new float[128]));
            await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        });
        await f.In(async sp =>
        {
            var repo = sp.GetRequiredService<IFaceSampleRepository>();
            foreach (var s in await repo.ListByEmployeeAsync(id)) repo.Remove(s);
            await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        });
        var left = await f.In(sp => sp.GetRequiredService<CameraAiDbContext>().Database
            .SqlQueryRaw<int>("SELECT COUNT(*) AS Value FROM sample_embeddings").SingleAsync());
        Assert.Equal(0, left);
    }

    [Fact]
    public async Task Reads_rows_written_in_python_format()
    {
        await using var f = new DbFixture();
        // Ghi bằng SQL thô đúng như cameraai/storage.py ghi (giờ ISO +07:00, reason chữ thường).
        await f.In(sp => sp.GetRequiredService<CameraAiDbContext>().Database.ExecuteSqlRawAsync("""
            INSERT INTO employees(code, name, active, created_at) VALUES ('PY01', 'Từ Python', 1, '2026-10-20T09:00:00+07:00');
            INSERT INTO attendance_events(employee_id, camera_id, ts, score, liveness_score, snapshot_path)
                VALUES (1, 'laptop', '2026-10-26T07:59:59+07:00', 0.71, NULL, NULL),
                       (1, 'laptop', '2026-10-26T17:30:00+07:00', 0.75, 0.92, 'snapshots/2026-10-26/x.jpg'),
                       (1, 'laptop', '2026-10-27T00:00:00+07:00', 0.80, NULL, NULL);
            INSERT INTO rejected_events(camera_id, ts, reason, best_employee_id, score) VALUES
                ('laptop', '2026-10-26T10:00:00+07:00', 'spoof', 1, 0.6),
                ('laptop', '2026-10-26T11:00:00+07:00', 'unknown', NULL, 0.2);
            """));

        var clock = f.Services.GetRequiredService<LocalClock>();
        var (from, to) = clock.DayRange(new DateOnly(2026, 10, 26), new DateOnly(2026, 10, 26));
        await f.In(async sp =>
        {
            var q = sp.GetRequiredService<IReportQueries>();
            var events = await q.AttendanceAsync(from, to);
            Assert.Equal(2, events.Count);                                  // 27/10 00:00 thuộc ngày sau
            Assert.Equal(DbFixture.At(7, 59).AddSeconds(59), events[0].Ts);
            Assert.Equal(DbFixture.Vn, events[0].Ts.Offset);                // đọc ra vẫn là giờ VN

            var summary = Assert.Single(await q.DailySummaryAsync(from, to));
            Assert.Equal(("PY01", 2, DbFixture.At(17, 30)), (summary.Code, summary.Count, summary.LastTs));

            var rejected = await q.RejectedAsync(from, to);
            Assert.Equal(["unknown", "spoof"], rejected.Select(r => r.Reason));  // mới nhất trước
            Assert.Equal("Từ Python", rejected[1].BestName);                     // LEFT JOIN có người
            Assert.Null(rejected[0].BestName);                                   // LEFT JOIN không có người

            var employee = Assert.Single(await q.EmployeesAsync());
            Assert.Equal(new DateTimeOffset(2026, 10, 20, 9, 0, 0, DbFixture.Vn), employee.CreatedAt);
        });
    }

    [Fact]
    public async Task Writes_times_in_python_format_whatever_the_input_offset()
    {
        await using var f = new DbFixture();
        var id = await AddEmployee(f, "DEV01", "An");
        await f.In(async sp =>
        {
            var utc = new DateTimeOffset(2026, 10, 26, 1, 0, 0, TimeSpan.Zero); // = 08:00 giờ VN
            sp.GetRequiredService<IAttendanceRepository>().Add(AttendanceEvent.Record(id, "laptop", utc, 0.8, null, null));
            await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        });
        var raw = await f.In(sp => sp.GetRequiredService<CameraAiDbContext>().Database
            .SqlQueryRaw<string>("SELECT ts AS Value FROM attendance_events").SingleAsync());
        Assert.Equal("2026-10-26T08:00:00+07:00", raw);

        var last = await f.In(sp => sp.GetRequiredService<IAttendanceRepository>()
            .LastTimesAsync("laptop", DbFixture.At(7, 58)));
        Assert.Equal(DbFixture.At(8), last[id]);
    }

    [Fact]
    public async Task Pending_sync_then_mark_synced()
    {
        await using var f = new DbFixture();
        var id = await AddEmployee(f, "DEV01", "An");
        await f.In(async sp =>
        {
            var repo = sp.GetRequiredService<IAttendanceRepository>();
            repo.Add(AttendanceEvent.Record(id, "laptop", DbFixture.At(8), 0.8, null, null));
            repo.Add(AttendanceEvent.Record(id, "laptop", DbFixture.At(17), 0.8, null, null));
            await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        });
        var pending = await f.In(sp => sp.GetRequiredService<IReportQueries>().PendingSyncAsync(10));
        Assert.Equal(2, pending.Count);
        Assert.Equal("DEV01", pending[0].EmployeeCode);

        await f.In(async sp =>
        {
            foreach (var e in await sp.GetRequiredService<IAttendanceRepository>().GetManyAsync([pending[0].Id]))
                e.MarkSynced(DbFixture.At(18));
            await sp.GetRequiredService<IUnitOfWork>().SaveChangesAsync();
        });
        Assert.Single(await f.In(sp => sp.GetRequiredService<IReportQueries>().PendingSyncAsync(10)));
    }

    [Fact]
    public async Task Use_cases_run_end_to_end_on_real_database()
    {
        await using var f = new DbFixture();
        Task<T> Send<T>(IRequest<T> request) => f.In(sp => sp.GetRequiredService<ISender>().Send(request));

        var created = await Send(new CreateEmployeeCommand("DEV01", "An"));
        await Assert.ThrowsAsync<DuplicateEmployeeException>(() => Send(new CreateEmployeeCommand("DEV01", "Bình")));
        var list = await Send(new ListEmployeesQuery());
        Assert.Equal(created.Id, Assert.Single(list).Id);
    }
}

public class LocalFileStoreTests
{
    [Fact]
    public async Task Save_read_delete_and_block_path_traversal()
    {
        var root = Path.Combine(Path.GetTempPath(), "cameraai-files-" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            var store = new LocalFileStore(root);
            var path = await store.SaveAsync("faces/7", [1, 2, 3]);

            Assert.Matches(@"^faces/7/[0-9a-f]{12}\.jpg$", path);
            Assert.Equal([1, 2, 3], await store.ReadAsync(path));
            Assert.Null(store.ResolveFullPath("faces/../../secret.txt"));
            Assert.Null(store.ResolveFullPath("../outside.jpg"));
            Assert.Null(store.ResolveFullPath("cameraai.db"));           // chỉ cho phép faces/ và snapshots/
            await Assert.ThrowsAsync<ArgumentException>(() => store.SaveAsync("../x", [1]));

            store.Delete(path);
            Assert.Null(await store.ReadAsync(path));
            Assert.False(Directory.Exists(Path.Combine(root, "faces", "7")));  // thư mục rỗng bị dọn
            Assert.True(Directory.Exists(Path.Combine(root, "faces")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
