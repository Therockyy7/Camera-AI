using CameraAI.Application.Abstractions;

namespace CameraAI.Infrastructure.Files;

/// <summary>
/// Ảnh lưu trong thư mục data/ (data/faces/&lt;nhân viên&gt;/..., data/snapshots/&lt;ngày&gt;/...).
/// DB chỉ giữ đường dẫn TƯƠNG ĐỐI → chép cả thư mục data sang máy khác vẫn chạy.
/// </summary>
internal sealed class LocalFileStore : IFileStore
{
    private static readonly string[] AllowedRoots = ["snapshots", "faces"];
    private readonly string _root;

    public LocalFileStore(string dataDir)
    {
        _root = Path.GetFullPath(dataDir);
        foreach (var sub in AllowedRoots) Directory.CreateDirectory(Path.Combine(_root, sub));
    }

    public async Task<string> SaveAsync(string folder, byte[] data, CancellationToken ct = default)
    {
        var name = Guid.NewGuid().ToString("N")[..12] + ".jpg"; // 12 ký tự hex như bản Python
        var relative = $"{folder.Trim('/')}/{name}";
        var full = ToSafeFullPath(relative) ?? throw new ArgumentException($"Thư mục không hợp lệ: {folder}", nameof(folder));
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        await File.WriteAllBytesAsync(full, data, ct);
        return relative;
    }

    public async Task<byte[]?> ReadAsync(string relativePath, CancellationToken ct = default) =>
        ResolveFullPath(relativePath) is { } full ? await File.ReadAllBytesAsync(full, ct) : null;

    public void Delete(string? relativePath)
    {
        if (relativePath is null || ResolveFullPath(relativePath) is not { } full) return;
        File.Delete(full);
        var folder = Path.GetDirectoryName(full)!;
        // Xoá thư mục ngày / thư mục nhân viên khi đã rỗng (không đụng tới data/faces, data/snapshots).
        if (Path.GetDirectoryName(folder) != _root && !Directory.EnumerateFileSystemEntries(folder).Any())
            Directory.Delete(folder);
    }

    public string? ResolveFullPath(string relativePath) =>
        ToSafeFullPath(relativePath) is { } full && File.Exists(full) ? full : null;

    /// <summary>Chặn path traversal: "faces/../../Windows/win.ini" → null.</summary>
    private string? ToSafeFullPath(string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath)) return null;
        var full = Path.GetFullPath(Path.Combine(_root, relativePath));
        if (!full.StartsWith(_root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) return null;
        var top = Path.GetRelativePath(_root, full).Split(Path.DirectorySeparatorChar)[0];
        return AllowedRoots.Contains(top, StringComparer.OrdinalIgnoreCase) ? full : null;
    }
}
