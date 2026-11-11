using CameraAI.Domain.Attendance;
using CameraAI.Domain.Faces;
using static CameraAI.Domain.Tests.TestData;

namespace CameraAI.Domain.Tests;

public class FaceTrackerTests
{
    [Fact]
    public void Iou_of_identical_and_disjoint_boxes()
    {
        Assert.Equal(1.0, new FaceBox(0, 0, 10, 10).Iou(new FaceBox(0, 0, 10, 10)));
        Assert.Equal(0.0, new FaceBox(0, 0, 10, 10).Iou(new FaceBox(20, 20, 10, 10)));
    }

    [Fact]
    public void Follows_moving_face_and_splits_far_faces()
    {
        var t = new FaceTracker();
        var a = t.Update([Obs(box: new(100, 100, 100, 100))], At(8)).Single().Track;
        var b = t.Update([Obs(box: new(110, 105, 100, 100))], At(8).AddSeconds(0.1)).Single().Track;
        Assert.Equal(a.Id, b.Id);

        var pairs = t.Update([Obs(box: new(112, 105, 100, 100)), Obs(box: new(400, 100, 100, 100))], At(8).AddSeconds(0.2));
        Assert.Equal(a.Id, pairs[0].Track.Id);
        Assert.NotEqual(a.Id, pairs[1].Track.Id);
    }

    [Fact]
    public void Expires_old_tracks()
    {
        var t = new FaceTracker(maxAgeSeconds: 1.5);
        t.Update([Obs()], At(8));
        t.Update([], At(8, 0, 2));
        Assert.Empty(t.Tracks);
    }

    [Fact]
    public void Unusable_observation_not_in_history()
    {
        var track = new FaceTracker().Update([Obs(reject: RejectReason.Blurry)], At(8)).Single().Track;
        Assert.Empty(track.History);
    }
}

public class AttendanceDeciderTests
{
    private static readonly AttendanceRules Rules = new(WindowFrames: 5, ConfirmFrames: 3, CooldownMinutes: 5);

    private static AttendanceDecider Decider(LivenessMode mode = LivenessMode.Required, double spoof = 0.3) =>
        new(Rules with { SpoofThreshold = spoof }, LivenessPolicyFactory.Create(mode));

    [Fact]
    public void Attendance_after_confirm_frames_once_per_track()
    {
        var results = Feed(Decider(), Repeat(() => Obs(7), 6), At(8));
        Assert.Equal([(2, DecisionKind.Attendance, 7)], Summary(results));
    }

    [Fact]
    public void Majority_vote_picks_most_seen_employee()
    {
        var frames = new[] { 7, 8, 7, 8, 7 }.Select(e => new[] { Obs(e) });
        Assert.Equal([(4, DecisionKind.Attendance, 7)], Summary(Feed(Decider(), frames, At(8))));
    }

    [Fact]
    public void Cooldown_blocks_new_track_then_allows_later()
    {
        var d = Decider();
        Assert.Single(Feed(d, Repeat(() => Obs(7), 3), At(8, 0)));
        Assert.Empty(Feed(d, Repeat(() => Obs(7), 3), At(8, 3)));
        Assert.Single(Feed(d, Repeat(() => Obs(7), 3), At(8, 6)));
    }

    [Fact]
    public void Required_mode_needs_real_faces()
    {
        var frames = new bool?[] { true, false, true, false }.Select(r => new[] { Obs(7, real: r) });
        Assert.Empty(Feed(Decider(), frames, At(8)));
    }

    [Fact]
    public void Spoof_recorded_once_after_full_window()
    {
        var results = Feed(Decider(), Repeat(() => Obs(7, real: false), 8), At(8));
        Assert.Equal([(4, DecisionKind.Spoof, 7)], Summary(results));
    }

    [Fact]
    public void Spoof_uses_window_average()
    {
        var frames = new[] { 0.2, 0.1, 0.4, 0.1, 0.2 }.Select(p => new[] { Obs(7, liveness: p) });
        Assert.Equal([(4, DecisionKind.Spoof, 7)], Summary(Feed(Decider(), frames, At(8))));
    }

    [Fact]
    public void Real_face_with_occasional_low_scores_is_not_spoof()
    {
        var frames = new[] { 0.9, 0.1, 0.95, 0.05, 0.98, 0.2, 0.97 }.Select(p => new[] { Obs(7, liveness: p) });
        Assert.Equal([(4, DecisionKind.Attendance, 7)], Summary(Feed(Decider(), frames, At(8))));
    }

    [Fact]
    public void Looks_spoof_follows_current_window()
    {
        var d = Decider();
        var tracker = new FaceTracker(window: 5);
        var results = Feed(d, Repeat(() => Obs(7, liveness: 0.05), 5), At(8), tracker);
        var track = tracker.Tracks.Single();
        Assert.True(d.LooksSpoof(track));

        results.AddRange(Feed(d, Repeat(() => Obs(7, liveness: 0.95), 5), At(8, 0, 1), tracker));
        Assert.Equal([(4, DecisionKind.Spoof, 7), (2, DecisionKind.Attendance, 7)], Summary(results));
        Assert.False(d.LooksSpoof(track));
        Assert.Equal(7, track.EmployeeId);
    }

    [Fact]
    public void Off_mode_ignores_liveness()
    {
        var results = Feed(Decider(LivenessMode.Off), Repeat(() => Obs(7, real: null), 3), At(8));
        Assert.Equal([(2, DecisionKind.Attendance, 7)], Summary(results));
    }

    [Fact]
    public void Unknown_after_full_window()
    {
        var results = Feed(Decider(), Repeat(() => Obs(null, score: 0.2, nearest: 3), 7), At(8));
        Assert.Equal([(4, DecisionKind.Unknown, 3)], Summary(results));
    }

    [Fact]
    public void Opportunistic_counts_unchecked_frames()
    {
        var results = Feed(Decider(LivenessMode.Opportunistic), Repeat(() => Obs(7, real: null), 3), At(8));
        Assert.Equal([(2, DecisionKind.Attendance, 7)], Summary(results));
        Assert.Null(results[0].Decision.Liveness);
    }

    [Fact]
    public void Required_ignores_unchecked_frames()
    {
        Assert.Empty(Feed(Decider(), Repeat(() => Obs(7, real: null), 5), At(8)));
    }

    [Fact]
    public void Opportunistic_rejects_checked_spoof_frames()
    {
        var frames = new bool?[] { null, false, null, false }.Select(r => new[] { Obs(7, real: r) });
        Assert.Empty(Feed(Decider(LivenessMode.Opportunistic), frames, At(8)));
    }

    [Fact]
    public void Spoof_needs_full_window_of_checked_frames()
    {
        var frames = new bool?[] { false, null, false, null, false, null, false }
            .Select(r => new[] { Obs(null, score: 0.2, real: r) });
        var results = Feed(Decider(LivenessMode.Opportunistic), frames, At(8));
        Assert.DoesNotContain(results, r => r.Decision.Kind == DecisionKind.Spoof);
    }

    [Fact]
    public void Attendance_reports_mean_of_checked_liveness()
    {
        FaceObservation[][] frames = [[Obs(7, liveness: 0.8)], [Obs(7, real: null)], [Obs(7, liveness: 1.0)]];
        var decision = Feed(Decider(LivenessMode.Opportunistic), frames, At(8)).Single().Decision;
        Assert.Equal(0.9, decision.Liveness!.Value, precision: 6);
    }

    [Fact]
    public void Seed_restores_cooldown()
    {
        var d = Decider();
        d.Seed(new Dictionary<int, DateTimeOffset> { [7] = At(7, 58) });
        Assert.Empty(Feed(d, Repeat(() => Obs(7), 3), At(8)));
        Assert.Equal(At(7, 58), d.LastRecorded(7));
    }
}
