using System.Globalization;
using System.Text;
using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Application.Messaging;
using CameraAI.Application.Pipeline;
using Microsoft.Extensions.Options;

namespace CameraAI.Application.Features.Reports;

// ---------- QUERY: chấm công trong một ngày ----------
public sealed record AttendanceDayDto(DateOnly Date, List<DailySummaryRow> Summary, List<AttendanceRow> Events);

/// <param name="Date">null = hôm nay (theo múi giờ cấu hình, không phải múi giờ máy chủ).</param>
public sealed record GetAttendanceDayQuery(DateOnly? Date) : IRequest<AttendanceDayDto>;

public sealed class GetAttendanceDayHandler(IReportQueries queries, LocalClock clock)
    : IRequestHandler<GetAttendanceDayQuery, AttendanceDayDto>
{
    public async Task<AttendanceDayDto> Handle(GetAttendanceDayQuery request, CancellationToken ct)
    {
        var day = request.Date ?? clock.Today;
        var (from, to) = clock.DayRange(day, day);
        return new AttendanceDayDto(day,
            await queries.DailySummaryAsync(from, to, ct),
            await queries.AttendanceAsync(from, to, ct));
    }
}

// ---------- QUERY: các lần bị từ chối trong ngày ----------
public sealed record RejectedDayDto(DateOnly Date, List<RejectedRow> Events);

public sealed record GetRejectedDayQuery(DateOnly? Date) : IRequest<RejectedDayDto>;

public sealed class GetRejectedDayHandler(IReportQueries queries, LocalClock clock)
    : IRequestHandler<GetRejectedDayQuery, RejectedDayDto>
{
    public async Task<RejectedDayDto> Handle(GetRejectedDayQuery request, CancellationToken ct)
    {
        var day = request.Date ?? clock.Today;
        var (from, to) = clock.DayRange(day, day);
        return new RejectedDayDto(day, await queries.RejectedAsync(from, to, ct));
    }
}

// ---------- QUERY: xuất CSV ----------
public sealed record CsvFile(string FileName, byte[] Content);

public sealed record ExportAttendanceCsvQuery(DateOnly? Start, DateOnly? End) : IRequest<CsvFile>;

public sealed class ExportAttendanceCsvHandler(IReportQueries queries, LocalClock clock)
    : IRequestHandler<ExportAttendanceCsvQuery, CsvFile>
{
    public async Task<CsvFile> Handle(ExportAttendanceCsvQuery request, CancellationToken ct)
    {
        var start = request.Start ?? clock.Today;
        var end = request.End ?? start;
        if (end < start) throw new InvalidRequestException("Ngày kết thúc phải sau ngày bắt đầu");

        var (from, to) = clock.DayRange(start, end);
        var csv = new StringBuilder();
        AppendRow(csv, "Mã NV", "Tên", "Thời điểm", "Camera", "Điểm khớp", "Điểm mặt thật");
        foreach (var e in await queries.AttendanceAsync(from, to, ct))
        {
            AppendRow(csv, e.Code, e.Name, e.Ts.ToString("yyyy-MM-ddTHH:mm:sszzz", CultureInfo.InvariantCulture), e.CameraId,
                e.Score.ToString("F3", CultureInfo.InvariantCulture),
                e.LivenessScore?.ToString("F3", CultureInfo.InvariantCulture) ?? "");
        }
        // UTF-8 có BOM để Excel đọc đúng tiếng Việt
        var bytes = Encoding.UTF8.GetPreamble().Concat(Encoding.UTF8.GetBytes(csv.ToString())).ToArray();
        return new CsvFile($"cham-cong_{start:yyyy-MM-dd}_{end:yyyy-MM-dd}.csv", bytes);
    }

    private static void AppendRow(StringBuilder csv, params string[] cells) =>
        csv.Append(string.Join(',', cells.Select(Escape))).Append("\r\n");

    private static string Escape(string cell) =>
        cell.IndexOfAny([',', '"', '\r', '\n']) >= 0 ? $"\"{cell.Replace("\"", "\"\"")}\"" : cell;
}

// ---------- QUERY: trạng thái (tab Trực tiếp hỏi mỗi giây) ----------
public sealed record StatusDto(
    CameraStatus Camera, string CameraId, double ProcessingFps, string RecognitionModel, string LivenessMode,
    int Employees, IReadOnlyList<RecentEvent> Recent, DateTimeOffset Now);

public sealed record GetStatusQuery : IRequest<StatusDto>;

public sealed class GetStatusHandler(
    IVideoSource source, AttendancePipeline pipeline, IEmployeeRepository employees, LocalClock clock,
    IOptions<CameraAiOptions> options) : IRequestHandler<GetStatusQuery, StatusDto>
{
    public async Task<StatusDto> Handle(GetStatusQuery request, CancellationToken ct)
    {
        var o = options.Value;
        return new StatusDto(source.Status, o.Camera.Id, pipeline.ProcessedFps, o.Recognition.Model,
            o.Liveness.Mode.ToString().ToLowerInvariant(), await employees.ActiveCountAsync(ct),
            pipeline.RecentEvents, clock.Now.TrimToSeconds());
    }
}
