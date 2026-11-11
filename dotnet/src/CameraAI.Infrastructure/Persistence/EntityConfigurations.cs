using CameraAI.Domain.Attendance;
using CameraAI.Domain.Employees;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace CameraAI.Infrastructure.Persistence;

// ============================================================================================
//  Fluent API: mô tả "entity Domain nằm trong bảng nào, cột nào" Ở TẦNG INFRASTRUCTURE.
//  Domain không biết nó được lưu vào SQLite — đổi sang SQL Server chỉ sửa các file ở đây.
//  Tên bảng/cột trùng schema của bản Python (cameraai/storage.py) để dùng chung file data/cameraai.db.
// ============================================================================================

internal sealed class EmployeeConfiguration : IEntityTypeConfiguration<Employee>
{
    public void Configure(EntityTypeBuilder<Employee> b)
    {
        b.ToTable("employees");
        b.HasKey(e => e.Id);
        b.HasIndex(e => e.Code).IsUnique();
    }
}

internal sealed class FaceSampleConfiguration : IEntityTypeConfiguration<FaceSample>
{
    public void Configure(EntityTypeBuilder<FaceSample> b)
    {
        b.ToTable("face_samples");
        b.HasKey(s => s.Id);
        b.HasOne<Employee>().WithMany().HasForeignKey(s => s.EmployeeId).OnDelete(DeleteBehavior.Restrict);
        b.Property(s => s.Source).HasConversion(
            v => v.ToString().ToLowerInvariant(),                 // "camera" / "upload"
            v => Enum.Parse<SampleSource>(v, ignoreCase: true));

        // SampleEmbedding là VALUE OBJECT (không có Id riêng) → "owned type": sống trong bảng riêng
        // sample_embeddings, khoá = (sample_id, model), xoá mẫu thì xoá theo.
        b.OwnsMany(s => s.Embeddings, e =>
        {
            e.ToTable("sample_embeddings");
            e.WithOwner().HasForeignKey("SampleId");
            e.Property<int>("SampleId").HasColumnName("sample_id");
            e.HasKey("SampleId", nameof(SampleEmbedding.Model));
            e.Property(x => x.Vector).HasColumnName("embedding")
                .HasConversion(new EmbeddingConverter(), new EmbeddingComparer());
        });
        // Embeddings là IReadOnlyCollection → EF ghi thẳng vào field _embeddings (Domain không phải mở setter).
        b.Navigation(s => s.Embeddings).UsePropertyAccessMode(PropertyAccessMode.Field);
    }
}

internal sealed class AttendanceEventConfiguration : IEntityTypeConfiguration<AttendanceEvent>
{
    public void Configure(EntityTypeBuilder<AttendanceEvent> b)
    {
        b.ToTable("attendance_events");
        b.HasKey(a => a.Id);
        b.HasOne<Employee>().WithMany().HasForeignKey(a => a.EmployeeId).OnDelete(DeleteBehavior.Restrict);
        b.Property(a => a.Timestamp).HasColumnName("ts");
        b.HasIndex(a => a.Timestamp).HasDatabaseName("ix_attendance_ts");
    }
}

internal sealed class RejectedEventConfiguration : IEntityTypeConfiguration<RejectedEvent>
{
    public void Configure(EntityTypeBuilder<RejectedEvent> b)
    {
        b.ToTable("rejected_events");
        b.HasKey(r => r.Id);
        b.Property(r => r.Timestamp).HasColumnName("ts");
        b.Property(r => r.Reason).HasConversion(
            v => v.ToString().ToLowerInvariant(),                 // "spoof" / "unknown"
            v => Enum.Parse<DecisionKind>(v, ignoreCase: true));
        b.HasIndex(r => r.Timestamp).HasDatabaseName("ix_rejected_ts");
    }
}
