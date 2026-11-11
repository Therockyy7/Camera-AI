using System.Collections.Concurrent;
using Microsoft.Extensions.DependencyInjection;

namespace CameraAI.Application.Messaging;

// ============================================================================================
//  MEDIATOR PATTERN — bản tự viết, cùng ý tưởng (và cùng tên interface) với thư viện MediatR.
//  (MediatR từ v13 đổi sang giấy phép thương mại; tự viết ~70 dòng vừa đủ dùng, vừa hiểu nó chạy thế nào.)
//
//  Controller/endpoint KHÔNG gọi thẳng service. Nó tạo một "thông điệp" (Command/Query) rồi gửi cho Mediator:
//      var dto = await sender.Send(new CreateEmployeeCommand("DEV01", "An"));
//  Mediator tìm đúng Handler trong DI và gọi. Endpoint chỉ phụ thuộc ISender → thêm use case mới
//  không phải sửa constructor của ai.
// ============================================================================================

/// <summary>Thông điệp có kết quả kiểu <typeparamref name="TResponse"/>. Command (ghi) hay Query (đọc) đều là IRequest.</summary>
public interface IRequest<TResponse>;

/// <summary>Xử lý đúng MỘT loại request. Mỗi use case = 1 handler (Single Responsibility).</summary>
public interface IRequestHandler<in TRequest, TResponse> where TRequest : IRequest<TResponse>
{
    Task<TResponse> Handle(TRequest request, CancellationToken ct);
}

/// <summary>
/// Lớp bọc quanh MỌI handler (logging, đo thời gian, validate, transaction...).
/// Đây là DECORATOR PATTERN: thêm hành vi mà không sửa handler nào. <paramref name="next"/> = lớp bên trong.
/// </summary>
public interface IPipelineBehavior<in TRequest, TResponse> where TRequest : IRequest<TResponse>
{
    Task<TResponse> Handle(TRequest request, Func<Task<TResponse>> next, CancellationToken ct);
}

public interface ISender
{
    Task<TResponse> Send<TResponse>(IRequest<TResponse> request, CancellationToken ct = default);
}

/// <summary>Đăng ký Scoped: dùng IServiceProvider của request hiện tại → handler nhận đúng DbContext của request đó.</summary>
internal sealed class Sender(IServiceProvider services) : ISender
{
    // Cache "người chuyển phát" theo kiểu request: reflection (MakeGenericType) chỉ chạy lần đầu.
    private static readonly ConcurrentDictionary<Type, object> Dispatchers = new();

    public Task<TResponse> Send<TResponse>(IRequest<TResponse> request, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        // Lúc compile chỉ biết IRequest<TResponse>; kiểu thật (vd CreateEmployeeCommand) chỉ biết lúc chạy.
        // → dựng Dispatcher<CreateEmployeeCommand, EmployeeDto> bằng reflection để lấy lại kiểu cụ thể.
        var dispatcher = (Dispatcher<TResponse>)Dispatchers.GetOrAdd(request.GetType(), static type =>
            Activator.CreateInstance(typeof(Dispatcher<,>).MakeGenericType(type, typeof(TResponse)))!);
        return dispatcher.Dispatch(request, services, ct);
    }
}

internal abstract class Dispatcher<TResponse>
{
    public abstract Task<TResponse> Dispatch(IRequest<TResponse> request, IServiceProvider services, CancellationToken ct);
}

internal sealed class Dispatcher<TRequest, TResponse> : Dispatcher<TResponse> where TRequest : IRequest<TResponse>
{
    public override Task<TResponse> Dispatch(IRequest<TResponse> request, IServiceProvider services, CancellationToken ct)
    {
        var typed = (TRequest)request;
        var handler = services.GetRequiredService<IRequestHandler<TRequest, TResponse>>();

        // Xếp các lớp như củ hành: behavior đăng ký TRƯỚC nằm NGOÀI cùng.
        //   Logging( Validation( handler ) )
        Func<Task<TResponse>> next = () => handler.Handle(typed, ct);
        foreach (var behavior in services.GetServices<IPipelineBehavior<TRequest, TResponse>>().Reverse())
        {
            var inner = next;
            next = () => behavior.Handle(typed, inner, ct);
        }
        return next();
    }
}
