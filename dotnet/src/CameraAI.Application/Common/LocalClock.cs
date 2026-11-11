using Microsoft.Extensions.Options;

namespace CameraAI.Application.Common;

/// <summary>
/// Giờ địa phương theo múi giờ cấu hình (Asia/Ho_Chi_Minh), dựa trên <see cref="TimeProvider"/> của .NET.
///
/// Vì sao không gọi thẳng <c>DateTimeOffset.Now</c>? Đó là static, mỗi lần gọi ra một giá trị khác →
/// không test được "cooldown 5 phút" hay "dọn ảnh cũ 30 ngày". Tiêm TimeProvider vào thì khi test ta dùng
/// FakeTimeProvider và tua giờ tuỳ ý.
/// </summary>
public sealed class LocalClock
{
    private readonly TimeProvider _time;
    private readonly TimeZoneInfo _zone;

    public LocalClock(TimeProvider time, IOptions<CameraAiOptions> options)
    {
        _time = time;
        _zone = TimeZoneInfo.FindSystemTimeZoneById(options.Value.TimeZone);
    }

    public DateTimeOffset Now => ToLocal(_time.GetUtcNow());
    public DateOnly Today => DateOnly.FromDateTime(Now.DateTime);

    public DateTimeOffset ToLocal(DateTimeOffset at) => TimeZoneInfo.ConvertTime(at, _zone);

    /// <summary>Khoảng [đầu ngày first, đầu ngày sau last) theo giờ địa phương.</summary>
    public (DateTimeOffset From, DateTimeOffset To) DayRange(DateOnly first, DateOnly last) =>
        (StartOf(first), StartOf(last.AddDays(1)));

    private DateTimeOffset StartOf(DateOnly day)
    {
        var midnight = day.ToDateTime(TimeOnly.MinValue);
        return new DateTimeOffset(midnight, _zone.GetUtcOffset(midnight));
    }
}

public static class TimeExtensions
{
    /// <summary>Bỏ phần lẻ của giây — giờ lưu/hiển thị chỉ cần tới giây (như bản Python).</summary>
    public static DateTimeOffset TrimToSeconds(this DateTimeOffset at) =>
        at.AddTicks(-(at.Ticks % TimeSpan.TicksPerSecond));
}
