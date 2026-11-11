using CameraAI.Application.Abstractions;
using CameraAI.Application.Common;
using CameraAI.Domain.Faces;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CameraAI.Application.Faces;

/// <summary>
/// Bảng id → tên nhân viên cho luồng camera vẽ nhãn. SINGLETON, cùng kỹ thuật "tráo snapshot bất biến"
/// như FaceGallery: luồng web nạp lại, luồng camera đọc, không cần lock.
/// </summary>
public sealed class EmployeeNames
{
    private volatile IReadOnlyDictionary<int, string> _names = new Dictionary<int, string>();

    public int Count => _names.Count;
    public void Replace(IReadOnlyDictionary<int, string> names) => _names = names;
    public string NameOf(int? employeeId) =>
        employeeId is { } id && _names.TryGetValue(id, out var name) ? name : "";
}

/// <summary>
/// Nạp lại gallery + tên từ DB sau mỗi thay đổi (thêm/xoá mẫu, xoá nhân viên) và lúc khởi động.
///
/// SCOPED, vì cần repository (Scoped, dùng DbContext). Nó phụ thuộc vào FaceGallery/EmployeeNames (Singleton)
/// — chiều Scoped → Singleton là HỢP LỆ. Chiều ngược lại (Singleton giữ Scoped) mới là lỗi "captive dependency".
/// </summary>
public sealed class GalleryLoader(
    IFaceSampleRepository samples,
    IEmployeeRepository employees,
    FaceGallery gallery,
    EmployeeNames names,
    IOptions<CameraAiOptions> options,
    ILogger<GalleryLoader> log)
{
    public async Task ReloadAsync(CancellationToken ct = default)
    {
        var model = options.Value.Recognition.Model;
        gallery.Load(await samples.GalleryAsync(model, ct));
        names.Replace(await employees.ActiveNamesAsync(ct));
        log.LogInformation("Đã nạp {Samples} mẫu khuôn mặt ({Model}) của {Employees} nhân viên",
            gallery.Size, model, names.Count);
    }
}
