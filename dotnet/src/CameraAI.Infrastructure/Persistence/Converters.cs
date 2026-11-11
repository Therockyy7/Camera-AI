using System.Globalization;
using System.Runtime.InteropServices;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

namespace CameraAI.Infrastructure.Persistence;

/// <summary>
/// DateTimeOffset ↔ chuỗi ISO 8601 theo giờ ĐỊA PHƯƠNG, đúng định dạng bản Python ghi: "2026-10-26T08:00:00+07:00".
///
/// Vì sao không để EF lưu kiểu mặc định? (1) Tương thích file DB của bản Python. (2) SQLite không có kiểu ngày giờ;
/// EF Core không dịch được phép so sánh/sắp xếp DateTimeOffset sang SQL. Khi mọi giá trị cùng một offset (+07:00,
/// Việt Nam không đổi giờ mùa hè) thì so sánh CHUỖI = so sánh THỜI GIAN → EF dịch được <c>ts &gt;= @from</c>
/// (nó tự đổi tham số @from qua converter này rồi so chuỗi).
/// </summary>
internal sealed class LocalIsoTimeConverter(TimeZoneInfo zone) : ValueConverter<DateTimeOffset, string>(
    v => TimeZoneInfo.ConvertTime(v, zone).ToString("yyyy-MM-ddTHH:mm:sszzz", CultureInfo.InvariantCulture),
    s => DateTimeOffset.Parse(s, CultureInfo.InvariantCulture));

/// <summary>float[] ↔ BLOB float32 liền nhau (giống <c>numpy.tobytes()</c> bên Python).</summary>
internal sealed class EmbeddingConverter() : ValueConverter<float[], byte[]>(
    v => EmbeddingBytes.ToBytes(v),
    b => EmbeddingBytes.FromBytes(b));

internal static class EmbeddingBytes
{
    public static byte[] ToBytes(float[] v) => MemoryMarshal.AsBytes(v.AsSpan()).ToArray();
    public static float[] FromBytes(byte[] b) => MemoryMarshal.Cast<byte, float>(b).ToArray();

    // Lambda trong ValueConverter/ValueComparer là EXPRESSION TREE (EF đọc được cấu trúc của nó), mà expression
    // tree không chứa được Span → gói phần dùng Span vào hàm thường rồi gọi hàm đó.
    public static bool SameValues(float[]? a, float[]? b) => a is not null && b is not null && a.AsSpan().SequenceEqual(b);
}

/// <summary>
/// Mảng là kiểu tham chiếu: mặc định EF so "cùng object?" để biết có thay đổi không. Comparer này so theo
/// NỘI DUNG → EF phát hiện đúng khi embedding đổi giá trị.
/// </summary>
internal sealed class EmbeddingComparer() : ValueComparer<float[]>(
    (a, b) => EmbeddingBytes.SameValues(a, b),
    v => v.Length,
    v => v.ToArray());
