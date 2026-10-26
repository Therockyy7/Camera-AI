from datetime import timedelta

import pytest
from helpers import at, make_obs

from cameraai.attendance import AttendanceDecider, FaceTracker, iou
from cameraai.config import AttendanceConfig

CFG = AttendanceConfig(window_frames=5, confirm_frames=3, cooldown_minutes=5)


def feed(decider, frames, start, tracker=None, step_s=0.125):
    """frames: list[list[obs]]. Trả về [(chỉ số frame, Decision)]."""
    tracker = tracker or FaceTracker(window=CFG.window_frames)
    out = []
    for index, observations in enumerate(frames):
        now = start + timedelta(seconds=index * step_s)
        for track, _ in tracker.update(observations, now.timestamp()):
            decision = decider.evaluate(track, now)
            if decision:
                out.append((index, decision))
    return out


def summary(results):
    return [(index, d.kind, d.employee_id) for index, d in results]


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_tracker_follows_moving_face_and_splits_far_faces():
    t = FaceTracker()
    ((a, _),) = t.update([make_obs(box=(100, 100, 100, 100))], 0.0)
    ((b, _),) = t.update([make_obs(box=(110, 105, 100, 100))], 0.1)
    assert a.id == b.id
    pairs = t.update([make_obs(box=(112, 105, 100, 100)), make_obs(box=(400, 100, 100, 100))], 0.2)
    assert pairs[0][0].id == a.id and pairs[1][0].id != a.id


def test_tracker_expires_old_tracks():
    t = FaceTracker(max_age_s=1.5)
    t.update([make_obs()], 0.0)
    t.update([], 2.0)
    assert t.tracks == []


def test_unusable_observation_not_in_history():
    ((track, _),) = FaceTracker().update([make_obs(reject_reason="blurry")], 0.0)
    assert len(track.history) == 0


def test_attendance_after_confirm_frames_once_per_track():
    d = AttendanceDecider(CFG, liveness_mode="required")
    assert summary(feed(d, [[make_obs(7)]] * 6, at(8))) == [(2, "attendance", 7)]


def test_majority_vote_picks_most_seen_employee():
    d = AttendanceDecider(CFG, liveness_mode="required")
    frames = [[make_obs(e)] for e in (7, 8, 7, 8, 7)]
    assert summary(feed(d, frames, at(8))) == [(4, "attendance", 7)]


def test_cooldown_blocks_new_track_then_allows_later():
    d = AttendanceDecider(CFG, liveness_mode="required")
    first = feed(d, [[make_obs(7)]] * 3, at(8, 0))
    again = feed(d, [[make_obs(7)]] * 3, at(8, 3))
    later = feed(d, [[make_obs(7)]] * 3, at(8, 6))
    assert len(first) == 1 and again == [] and len(later) == 1


def test_requires_real_faces_when_liveness_on():
    d = AttendanceDecider(CFG, liveness_mode="required")
    frames = [[make_obs(7, real=r)] for r in (True, False, True, False)]
    assert feed(d, frames, at(8)) == []


def test_spoof_recorded_once_after_full_window():
    d = AttendanceDecider(CFG, liveness_mode="required")
    assert summary(feed(d, [[make_obs(7, real=False)]] * 8, at(8))) == [(4, "spoof", 7)]


def test_spoof_uses_window_average():
    d = AttendanceDecider(CFG, liveness_mode="required", spoof_threshold=0.3)
    frames = [[make_obs(7, liveness=p)] for p in (0.2, 0.1, 0.4, 0.1, 0.2)]
    assert summary(feed(d, frames, at(8))) == [(4, "spoof", 7)]


def test_real_face_with_occasional_low_scores_not_spoof():
    d = AttendanceDecider(CFG, liveness_mode="required", spoof_threshold=0.3)
    frames = [[make_obs(7, liveness=p)] for p in (0.9, 0.1, 0.95, 0.05, 0.98, 0.2, 0.97)]
    assert summary(feed(d, frames, at(8))) == [(4, "attendance", 7)]


def test_looks_spoof_follows_current_window():
    d = AttendanceDecider(CFG, liveness_mode="required", spoof_threshold=0.3)
    tracker = FaceTracker(window=CFG.window_frames)
    frames = [[make_obs(7, liveness=0.05)]] * 5 + [[make_obs(7, liveness=0.95)]] * 5
    results = feed(d, frames[:5], at(8), tracker=tracker)
    (track,) = tracker.tracks
    assert d.looks_spoof(track) is True
    results += feed(d, frames[5:], at(8, 0, 1), tracker=tracker)
    assert summary(results) == [(4, "spoof", 7), (2, "attendance", 7)]
    assert d.looks_spoof(track) is False and track.employee_id == 7


def test_liveness_disabled_ignores_flag():
    d = AttendanceDecider(CFG, liveness_mode="off")
    assert summary(feed(d, [[make_obs(7, real=None)]] * 3, at(8))) == [(2, "attendance", 7)]


def test_unknown_after_full_window():
    d = AttendanceDecider(CFG, liveness_mode="required")
    frames = [[make_obs(None, score=0.2, nearest=3)]] * 7
    assert summary(feed(d, frames, at(8))) == [(4, "unknown", 3)]


def test_opportunistic_counts_unchecked_frames():
    # Camera trần: mặt nhỏ/xiên không kiểm được giả mạo (liveness None) nhưng vẫn được chấm.
    d = AttendanceDecider(CFG, liveness_mode="opportunistic")
    results = feed(d, [[make_obs(7, real=None)]] * 3, at(8))
    assert summary(results) == [(2, "attendance", 7)] and results[0][1].liveness is None


def test_required_ignores_unchecked_frames():
    d = AttendanceDecider(CFG, liveness_mode="required")
    assert feed(d, [[make_obs(7, real=None)]] * 5, at(8)) == []


def test_opportunistic_rejects_checked_spoof_frames():
    d = AttendanceDecider(CFG, liveness_mode="opportunistic")
    frames = [[make_obs(7, real=r)] for r in (None, False, None, False)]
    assert feed(d, frames, at(8)) == []  # chỉ 2 frame không bị kiểm là giả


def test_spoof_needs_full_window_of_checked_frames():
    d = AttendanceDecider(CFG, liveness_mode="opportunistic", spoof_threshold=0.3)
    frames = [[make_obs(None, score=0.2, real=r)] for r in (False, None, False, None, False, None, False)]
    assert all(kind != "spoof" for _, kind, _ in summary(feed(d, frames, at(8))))


def test_attendance_reports_mean_of_checked_liveness():
    d = AttendanceDecider(CFG, liveness_mode="opportunistic")
    frames = [[make_obs(7, liveness=0.8)], [make_obs(7, real=None)], [make_obs(7, liveness=1.0)]]
    ((_, decision),) = feed(d, frames, at(8))
    assert decision.liveness == pytest.approx(0.9)


def test_seed_restores_cooldown():
    d = AttendanceDecider(CFG, liveness_mode="required")
    d.seed({7: at(7, 58)})
    assert feed(d, [[make_obs(7)]] * 3, at(8)) == []
    assert d.last_recorded(7) == at(7, 58)
