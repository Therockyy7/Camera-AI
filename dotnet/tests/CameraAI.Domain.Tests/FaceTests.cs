using CameraAI.Domain.Common;
using CameraAI.Domain.Employees;
using CameraAI.Domain.Faces;

namespace CameraAI.Domain.Tests;

public class FaceGalleryTests
{
    private static float[] Vec(params float[] head)
    {
        var v = new float[128];
        head.CopyTo(v, 0);
        return v;
    }

    [Fact]
    public void Empty_gallery_matches_nobody()
    {
        var m = new FaceGallery(0.363).Match(Vec(1));
        Assert.Equal(FaceMatch.None, m);
    }

    [Fact]
    public void Best_match_above_threshold()
    {
        var g = new FaceGallery(0.363);
        g.Load([(1, Vec(1, 0)), (1, Vec(0.9f, 0.1f)), (2, Vec(0, 1))]);
        var m = g.Match(Vec(2, 0.1f));
        Assert.Equal(1, m.EmployeeId);
        Assert.True(m.Score > 0.99);
        Assert.Equal(3, g.Size);
    }

    [Fact]
    public void Below_threshold_keeps_nearest()
    {
        var g = new FaceGallery(0.9);
        g.Load([(1, Vec(1, 0)), (2, Vec(0, 1))]);
        var m = g.Match(Vec(1, 1));
        Assert.Null(m.EmployeeId);
        Assert.Contains(m.BestEmployeeId, new int?[] { 1, 2 });
        Assert.Equal(0.7071, m.Score, precision: 3);
    }

    [Fact]
    public void Reload_replaces_samples()
    {
        var g = new FaceGallery(0.363);
        g.Load([(1, Vec(1))]);
        g.Load([(2, Vec(1))]);
        Assert.Equal(2, g.Match(Vec(1)).EmployeeId);
    }

    [Fact]
    public void Normalize_zero_vector_is_safe()
    {
        Assert.DoesNotContain(VectorMath.Normalize(new float[4]), float.IsNaN);
    }
}

public class QualityGateTests
{
    private readonly RecognitionGate _gate = new(MinFacePx: 40, MaxTurn: 0.45, MinSharpness: 12);

    [Theory]
    [InlineData(100, 100, 30, 30, 0.0, 50, RejectReason.Small)]
    [InlineData(2, 100, 80, 80, 0.0, 50, RejectReason.Edge)]
    [InlineData(100, 100, 80, 80, 0.5, 50, RejectReason.Turned)]
    [InlineData(100, 100, 80, 80, 0.0, 5, RejectReason.Blurry)]
    [InlineData(100, 100, 80, 80, 0.1, 50, null)]
    public void Recognition_gate(int x, int y, int w, int h, double turn, double sharp, RejectReason? expected)
    {
        Assert.Equal(expected, _gate.Evaluate(new FaceBox(x, y, w, h), turn, sharp, 640, 480));
    }

    [Fact]
    public void Enrollment_uses_stricter_variant_via_with()
    {
        var enroll = _gate with { MinFacePx = 80 };
        Assert.Equal(RejectReason.Small, enroll.Evaluate(new(100, 100, 60, 60), 0, 50, 640, 480));
        Assert.Null(_gate.Evaluate(new(100, 100, 60, 60), 0, 50, 640, 480));
    }

    [Fact]
    public void Turn_offset_zero_when_nose_centered()
    {
        Assert.Equal(0.0, RecognitionGate.TurnOffset(100, 140, 120), precision: 6);
        Assert.Equal(0.5, RecognitionGate.TurnOffset(100, 140, 140), precision: 6);
        Assert.Equal(1.0, RecognitionGate.TurnOffset(140, 100, 120));
    }

    [Fact]
    public void Liveness_gate_needs_big_frontal_face()
    {
        var gate = new LivenessGate();
        Assert.True(gate.CanCheck(new(0, 0, 64, 64), 0.2));
        Assert.False(gate.CanCheck(new(0, 0, 50, 50), 0.0));
        Assert.False(gate.CanCheck(new(0, 0, 100, 100), 0.3));
    }
}

public class EmployeeTests
{
    [Fact]
    public void Create_trims_and_validates()
    {
        var e = Employee.Create("  DEV01 ", " An ", TestData.At(8));
        Assert.Equal(("DEV01", "An", true), (e.Code, e.Name, e.Active));
        Assert.Throws<DomainException>(() => Employee.Create(" ", "An", TestData.At(8)));
    }

    [Fact]
    public void Deactivate_then_reactivate_keeps_identity()
    {
        var e = Employee.Create("DEV01", "An", TestData.At(8));
        Assert.True(e.Deactivate());
        Assert.False(e.Deactivate());
        e.Reactivate("An Nguyễn");
        Assert.True(e.Active);
        Assert.Equal("An Nguyễn", e.Name);
        Assert.Throws<DuplicateEmployeeException>(() => e.Reactivate("X"));
    }

    [Fact]
    public void Sample_replaces_embedding_of_same_model()
    {
        var s = FaceSample.Create(1, "faces/1/a.jpg", SampleSource.Camera, TestData.At(8), "sface", new float[128]);
        s.SetEmbedding("arcface", new float[512]);
        s.SetEmbedding("sface", new float[128]);
        Assert.Equal(2, s.Embeddings.Count);
    }
}
