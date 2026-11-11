using System.Diagnostics;
using Microsoft.Extensions.Logging;

namespace CameraAI.Application.Messaging;

/// <summary>
/// DECORATOR áp cho mọi use case: ghi log tên request + thời gian chạy, cảnh báo khi chậm.
/// 11 handler, không handler nào phải tự viết dòng log đo thời gian.
/// </summary>
public sealed class LoggingBehavior<TRequest, TResponse>(ILogger<LoggingBehavior<TRequest, TResponse>> log)
    : IPipelineBehavior<TRequest, TResponse> where TRequest : IRequest<TResponse>
{
    private static readonly TimeSpan Slow = TimeSpan.FromSeconds(1);

    public async Task<TResponse> Handle(TRequest request, Func<Task<TResponse>> next, CancellationToken ct)
    {
        var name = typeof(TRequest).Name;
        var started = Stopwatch.GetTimestamp();
        try
        {
            return await next();
        }
        finally
        {
            var elapsed = Stopwatch.GetElapsedTime(started);
            if (elapsed > Slow) log.LogWarning("{Request} chạy chậm: {Elapsed:F0} ms", name, elapsed.TotalMilliseconds);
            else log.LogDebug("{Request}: {Elapsed:F0} ms", name, elapsed.TotalMilliseconds);
        }
    }
}
