"""Possession as a hidden Markov model: frame evidence from the tracking, decoded with Viterbi.

The baseline smooths a nearest-player vote with a fixed window. Smoothing has no idea what a
possession is, so it either flickers (narrow window) or swallows short possessions (wide one).
An HMM states the structure directly -- three states, H, A and D, each sticky -- and Viterbi
finds the single most likely sequence. The stickiness is a property of football rather than a
number fitted to the public window:

    possessions last ~12 s, play runs ~30 s between stoppages, a stoppage lasts ~20 s

Evidence, all on by default:

  - which team is nearest the ball, fading when the ball is in flight;
  - touches: where the ball's velocity changes sharply with a player close, that player's side
    gets 4x the evidence; a fast ball with nobody near gets 0.3x;
  - for D: the ball out of the pitch or standing still, and players slowed to a walk (median
    speed from positions -- the ATD's own speed field over-called D on open play);
  - short ball gaps are interpolated.

After decoding, each possession change is snapped to the nearest supporting event within 1 s: a
touch by the new side, or the ball leaving the pitch or the tracking. This moves boundaries only;
the number of possessions cannot change.

Two settings were chosen by full-match plausibility, as reported by validate_submission.py, and
not by the public window: WALK_MS sets the dead-ball share (target ~40-45% of graded frames, the
usual LaLiga ball-out-of-play time) and TEMPER_FRAMES sets possession length (target ~14 s).
They barely interact. 1.1 m/s and 5 frames give 44.7% and 13.0 s.

Options tested and left off (solution/ROADMAP.md has the numbers):

    --dead _gap / _walkline   "ball missing > 3 s" or "lost near a line" as D: over-call D
    --tags setpieces          Smart Tagging restarts: harmless, within noise
    --tags tags               plus phase-team priors: neutral
    --shape                   ball-free team-shape model: merges short away possessions
    --learned replace/blend   local model trained between same-side touches: worse
    --chain k                 Erlang possession lengths: identical once length is matched

Run it blind (it reads the match data), or through the experiment harness:

    .venv/Scripts/python.exe tools/run_blind.py solution/hmm_possession.py
    .venv/Scripts/python.exe tools/experiment.py <name> -- [options]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import baseline  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FPS = GAMES[GAME]["fps"]
HOME, AWAY, DEAD = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# Football priors, per frame. Not fitted to any ground truth.
MEAN_POSSESSION_S = 12.0
MEAN_PLAY_BETWEEN_STOPPAGES_S = 30.0
MEAN_STOPPAGE_S = 20.0

# Nearest-player evidence. A player within CONTROL_RADIUS_M of the ball can be on it; beyond
# that the ball is in flight and the evidence fades with DECAY_M.
CONTROL_SCALE_M = 1.5
CONTROL_RADIUS_M = 1.5
DECAY_M = 3.0
DEAD_BASE = 0.1
# A ball that barely moves for a second while visible is being placed for a restart.
STATIONARY_HALF_WIDTH = FPS // 2
STATIONARY_M = 0.5
STATIONARY_DEAD = 0.6
# Consecutive frames are not independent observations: one touch lasts several of them. Dividing
# the log evidence by this many frames counts each touch roughly once. Set by full-match mean
# possession length, see the module docstring.
TEMPER_FRAMES = 5

# Dead-ball evidence beyond the ball itself. A ball missing for several seconds is usually out of
# play; players whose median speed drops to a walk are waiting for a restart. GAP_DEAD_S is a
# football fact; WALK_MS is set by the full-match dead-ball share, see the module docstring.
GAP_DEAD_S = 3.0
GAP_SOFTNESS_S = 1.0
GAP_DEAD = 0.6
MOTION_HALF_WIDTH = 5
MAX_PLAYER_SPEED_MS = 11.0
MIN_TRACKED_PLAYERS = 10
WALK_MS = 1.1
WALK_SOFTNESS_MS = 0.3
WALK_DEAD = 0.6
LINE_M = 2.0
LINE_GAP_S = 1.0
LINE_DEAD = 0.5
STICKY_MAX_S = 10.0
STICKY_DEAD = 0.6
# Touches. Tracking jitter alone moves the smoothed ball velocity by ~2-3 m/s, a pass or a
# tackle changes it by well over 5 m/s. A ball faster than FLIGHT_MS with nobody near is in the air
# or rolling between players.
VELOCITY_HALF_WIDTH = 2
TOUCH_HALF_WIDTH = 3
TOUCH_DV_MS = 5.0
TOUCH_NEAR_M = 2.5
TOUCH_GAIN = 4.0
FLIGHT_MS = 8.0
FLIGHT_GAIN = 0.3
# After decoding, each possession change moves to the nearest supporting event within +/- 1 s.
SNAP_FRAMES = FPS
# Learned local model: trained on frames between same-side touches at most 3 s apart.
BETWEEN_TOUCH_MAX_GAP = 3 * FPS
LOCAL_CAP_M = 25.0
LEARNED_CLIP = 0.03

# Team shape, learned per half from frames where the tracking alone is unambiguous: one player
# within WEAK_NEAR_M of the ball and the nearest opponent WEAK_MARGIN_M further away, in open play.
WEAK_NEAR_M = 1.0
WEAK_MARGIN_M = 2.0
MIN_SHAPE_TRAINING = 500
SHAPE_CLIP = 0.1
SHAPE_WEIGHT = 1.0
OUT_OF_PITCH_MARGIN_M = 0.5
MAX_INTERP_GAP = FPS // 2

# Smart Tagging priors, in nats per frame. Weak by design: a tag is evidence, not an answer.
TAG_TEAM_NATS = 0.15
SET_PIECE_DEAD_NATS = 0.3
SET_PIECE_DEAD_SHARE = 0.4
MIN_CALIBRATION_FRAMES = 200
FLOOR = 1e-6


def parse_atd(raw_path, n_frames, n_slots):
    """(ball_xy, player_xy, player_speed) on the video index, NaN where unobserved."""
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(baseline.FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            if parts[2][0] != baseline.MISSING:
                values = parts[2].split(",")
                ball[frame] = float(values[0]), float(values[1])
            for slot, group in enumerate(parts[1].split(baseline.GROUP_SEP)[:n_slots]):
                if group[0] == baseline.MISSING:
                    continue
                values = group.split(",")
                xy[frame, slot] = float(values[0]), float(values[1])
                if len(values) > 2 and values[2][0] != baseline.MISSING:
                    speed[frame, slot] = float(values[2])
    return ball, xy, speed


def interpolate_gaps(ball, blocks, max_gap):
    """Fill ball gaps of at most max_gap frames inside each half by straight lines."""
    out = ball.copy()
    for _, first, last in blocks:
        seen = ~np.isnan(ball[first:last + 1, 0])
        index = np.flatnonzero(seen)
        for left, right in zip(index[:-1], index[1:]):
            gap = right - left - 1
            if 0 < gap <= max_gap:
                step = np.linspace(0.0, 1.0, gap + 2)[1:-1, None]
                out[first + left + 1:first + right] = (
                    ball[first + left] + step * (ball[first + right] - ball[first + left]))
    return out


def nearest_distances(ball, xy, team_of_track, pitch):
    """Metres from the ball to the nearest home and nearest away player. inf where unknown."""
    length, width = pitch
    dx = (xy[:, :, 0] - ball[:, None, 0]) * length
    dy = (xy[:, :, 1] - ball[:, None, 1]) * width
    distance = np.sqrt(dx * dx + dy * dy)
    distance = np.where(np.isnan(distance), np.inf, distance)
    home = distance[:, team_of_track == baseline.HOME].min(axis=1)
    away = distance[:, team_of_track == baseline.AWAY].min(axis=1)
    return home, away


def gap_seconds(ball, blocks):
    """Length in seconds of the missing-ball run each frame sits in; 0 where the ball is seen."""
    out = np.zeros(len(ball))
    missing = np.isnan(ball[:, 0])
    for _, first, last in blocks:
        block = missing[first:last + 1].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], block, [0]))))
        for start, end in zip(edges[::2], edges[1::2]):
            out[first + start:first + end] = (end - start) / float(FPS)
    return out


def line_exit(ball_raw, blocks, pitch):
    """True over missing-ball runs longer than LINE_GAP_S whose last sighting was within LINE_M
    of a touchline or goal line: the ball most likely went out of play there."""
    length, width = pitch
    out = np.zeros(len(ball_raw), dtype=bool)
    missing = np.isnan(ball_raw[:, 0])
    for _, first, last in blocks:
        block = missing[first:last + 1].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], block, [0]))))
        for start, end in zip(edges[::2], edges[1::2]):
            if start == 0 or end - start < LINE_GAP_S * FPS:
                continue
            x, y = ball_raw[first + start - 1] * (length, width)
            if min(x, length - x, y, width - y) < LINE_M:
                out[first + start:first + end] = True
    return out


def team_motion(xy, pitch):
    """Median player speed in m/s from positions, over +/- MOTION_HALF_WIDTH frames, smoothed.

    Median across tracked players, so a slot that swaps identity and jumps across the pitch
    does not count; per-slot speeds above MAX_PLAYER_SPEED_MS are such jumps and are dropped.
    """
    length, width = pitch
    h = MOTION_HALF_WIDTH
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    delta = xy[2 * h:] - xy[:-2 * h]
    speed[h:-h] = np.hypot(delta[..., 0] * length, delta[..., 1] * width) * FPS / (2 * h)
    speed[speed > MAX_PLAYER_SPEED_MS] = np.nan
    tracked = np.isfinite(speed).sum(axis=1)
    median = np.full(len(xy), np.nan)
    enough = tracked >= MIN_TRACKED_PLAYERS
    median[enough] = np.nanmedian(speed[enough], axis=1)
    return rolling_mean(median, FPS // 2)


def shape_features(xy, team_of_track, pitch):
    """Ball-free team-shape features per frame, in metres, smoothed over 1 s. NaN if unknown.

    Centroid and spread of each side, as home-minus-away differences plus each side's own
    depth. The attacking direction flips at halftime, which is why the model is fitted per half.
    """
    length, width = pitch
    x, y = xy[..., 0] * length, xy[..., 1] * width
    columns = []
    stats = {}
    for side in (baseline.HOME, baseline.AWAY):
        mask = team_of_track == side
        sx, sy = x[:, mask], y[:, mask]
        count = np.isfinite(sx).sum(axis=1)
        ok = count >= 5
        mean_x = np.where(ok, np.nansum(sx, axis=1) / np.maximum(count, 1), np.nan)
        mean_y = np.where(ok, np.nansum(sy, axis=1) / np.maximum(count, 1), np.nan)
        spread_x = np.sqrt(np.where(ok, np.nansum((sx - mean_x[:, None]) ** 2, axis=1)
                                    / np.maximum(count, 1), np.nan))
        spread_y = np.sqrt(np.where(ok, np.nansum((sy - mean_y[:, None]) ** 2, axis=1)
                                    / np.maximum(count, 1), np.nan))
        stats[side] = (mean_x, mean_y, spread_x, spread_y)
    home, away = stats[baseline.HOME], stats[baseline.AWAY]
    columns = [home[0] - away[0], home[2] - away[2], home[3] - away[3], home[0], away[0],
               home[2] * home[3] - away[2] * away[3]]
    return np.column_stack([rolling_mean(column, FPS // 2) for column in columns])


def fit_logistic(features, labels, l2=1.0, iterations=25):
    """Logistic regression by Newton's method on standardised features. Deterministic."""
    mean, scale = features.mean(axis=0), features.std(axis=0) + 1e-9
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    weights = np.zeros(design.shape[1])
    penalty = l2 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (p - labels) + penalty @ weights
        hessian = (design * (p * (1 - p))[:, None]).T @ design + penalty
        weights -= np.linalg.solve(hessian, gradient)
    return mean, scale, weights


def shape_home_probability(features, weak_home, weak_mask, blocks):
    """P(home in possession) from team shape, fitted per half on confident tracking frames."""
    p = np.full(len(features), 0.5)
    for _, first, last in blocks:
        block = slice(first, last + 1)
        finite = np.isfinite(features[block]).all(axis=1)
        train = finite & weak_mask[block]
        if train.sum() < MIN_SHAPE_TRAINING:
            continue
        mean, scale, weights = fit_logistic(features[block][train],
                                            weak_home[block][train].astype(float))
        design = np.column_stack([np.ones(int(finite.sum())),
                                  (features[block][finite] - mean) / scale])
        out = np.full(last - first + 1, 0.5)
        out[finite] = 1.0 / (1.0 + np.exp(-design @ weights))
        p[block] = out
    return np.clip(p, SHAPE_CLIP, 1 - SHAPE_CLIP)


def rolling_mean(values, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN."""
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def stationary_ball(ball, pitch):
    """True where the ball is seen at both ends of a 1 s window and moved under STATIONARY_M.

    Computed from positions rather than the ATD's third field, which did not behave like a
    speed in m/s (an ablation showed it labelling most of open play as stoppage).
    """
    length, width = pitch
    h = STATIONARY_HALF_WIDTH
    ahead = np.full_like(ball, np.nan)
    behind = np.full_like(ball, np.nan)
    ahead[:-h], behind[h:] = ball[h:], ball[:-h]
    span = np.hypot((ahead[:, 0] - behind[:, 0]) * length, (ahead[:, 1] - behind[:, 1]) * width)
    return np.isfinite(span) & (span < STATIONARY_M) & ~np.isnan(ball[:, 0])


def ball_velocity(ball, pitch):
    """Ball velocity in m/s, (n, 2), from positions smoothed over +/- VELOCITY_HALF_WIDTH."""
    length, width = pitch
    h = VELOCITY_HALF_WIDTH
    metres = np.column_stack([rolling_mean(ball[:, 0].astype(float) * length, 1),
                              rolling_mean(ball[:, 1].astype(float) * width, 1)])
    metres[np.isnan(ball[:, 0])] = np.nan
    velocity = np.full_like(metres, np.nan)
    velocity[h:-h] = (metres[2 * h:] - metres[:-2 * h]) * FPS / (2.0 * h)
    return velocity


def touch_gain(ball, d_home, d_away, pitch):
    """Per-frame multiplier on the nearest-player log-odds.

    A touch -- the ball's velocity changing sharply while a player is close -- is the moment the
    nearest player really is playing the ball, so its evidence is boosted over a short window.
    A ball travelling fast with nobody near is in flight between touches, and the nearest player
    then says little about who has it.
    """
    velocity = ball_velocity(ball, pitch)
    h = TOUCH_HALF_WIDTH
    change = np.full(len(ball), np.nan)
    change[h:-h] = np.hypot(*(velocity[2 * h:] - velocity[:-2 * h]).T)
    change = np.nan_to_num(change, nan=0.0)
    near = np.minimum(d_home, d_away)
    touch = (change > TOUCH_DV_MS) & (near < TOUCH_NEAR_M)
    spread = np.convolve(touch.astype(float), np.ones(2 * h + 1), mode="same") > 0
    speed = np.nan_to_num(np.hypot(velocity[:, 0], velocity[:, 1]), nan=0.0)
    flight = (speed > FLIGHT_MS) & (near > TOUCH_NEAR_M) & ~spread
    touch_team = np.where(touch, np.where(d_home < d_away, HOME, AWAY), -1).astype(np.int8)
    return np.where(spread, TOUCH_GAIN, np.where(flight, FLIGHT_GAIN, 1.0)), touch_team


def between_touch_labels(touch_team, blocks, max_gap):
    """Self-supervised labels: frames strictly between two consecutive touches by the same side,
    at most max_gap frames apart, belong to that side. -1 elsewhere.

    A pass or dribble between two touches by one team is that team's possession throughout, even
    on the frames where an opponent happens to be the nearest player -- which is exactly the
    situation the hand-written nearest-player rule gets wrong.
    """
    labels = np.full(len(touch_team), -1, dtype=np.int8)
    for _, first, last in blocks:
        block = touch_team[first:last + 1]
        frames = np.flatnonzero(block >= 0)
        if not len(frames):
            continue
        starts = frames[np.concatenate(([True], np.diff(frames) > 1))]
        teams = block[starts]
        for a, b, team_a, team_b in zip(starts[:-1], starts[1:], teams[:-1], teams[1:]):
            if team_a == team_b and b - a <= max_gap:
                labels[first + a + 1:first + b] = team_a
    return labels


def local_features(ball, xy, team_of_track, pitch):
    """Ball-local features, (n, 7): two nearest per side (capped), counts within 5 and 10 m, speed.

    Nothing about territory or attacking direction, so one model serves both halves and cannot
    learn which side usually has the ball.
    """
    length, width = pitch
    dx = (xy[:, :, 0] - ball[:, None, 0]) * length
    dy = (xy[:, :, 1] - ball[:, None, 1]) * width
    distance = np.sqrt(dx * dx + dy * dy)
    distance = np.where(np.isnan(distance), np.inf, distance)
    columns = []
    for side in (baseline.HOME, baseline.AWAY):
        side_distance = np.sort(distance[:, team_of_track == side], axis=1)
        columns += [np.minimum(side_distance[:, 0], LOCAL_CAP_M),
                    np.minimum(side_distance[:, 1], LOCAL_CAP_M),
                    (side_distance < 5.0).sum(axis=1).astype(float),
                    (side_distance < 10.0).sum(axis=1).astype(float)]
    speed = np.hypot(*ball_velocity(ball, pitch).T)
    features = np.column_stack(columns[:4] + columns[4:] + [np.minimum(speed, 30.0)])
    features[np.isnan(ball[:, 0])] = np.nan
    return features


def learned_home_probability(features, labels):
    """P(home) from a logistic regression on between-touch labels, balanced by side."""
    finite = np.isfinite(features).all(axis=1)
    home = np.flatnonzero(finite & (labels == HOME))
    away = np.flatnonzero(finite & (labels == AWAY))
    p = np.full(len(features), np.nan)
    if min(len(home), len(away)) < MIN_SHAPE_TRAINING:
        return p
    n = min(len(home), len(away))
    # Deterministic balancing: an evenly spaced subsample of the larger side.
    home = home[np.linspace(0, len(home) - 1, n).astype(int)]
    away = away[np.linspace(0, len(away) - 1, n).astype(int)]
    train = np.concatenate((home, away))
    mean, scale, weights = fit_logistic(features[train], (labels[train] == HOME).astype(float))
    design = np.column_stack([np.ones(int(finite.sum())), (features[finite] - mean) / scale])
    p[finite] = 1.0 / (1.0 + np.exp(-design @ weights))
    return p


def dead_onsets(ball_raw, pitch):
    """Frames where the ball leaves the pitch or the tracking loses it: where a stoppage begins."""
    length, width = pitch
    seen = ~np.isnan(ball_raw[:, 0])
    margin_x, margin_y = OUT_OF_PITCH_MARGIN_M / length, OUT_OF_PITCH_MARGIN_M / width
    inside = seen & (ball_raw[:, 0] >= -margin_x) & (ball_raw[:, 0] <= 1 + margin_x) \
        & (ball_raw[:, 1] >= -margin_y) & (ball_raw[:, 1] <= 1 + margin_y)
    onset = np.zeros(len(ball_raw), dtype=bool)
    onset[1:] = inside[:-1] & ~inside[1:]
    return onset


def snap_boundaries(states, blocks, touch_team, onset, window):
    """Move each label change to the nearest supporting event within +/- window frames.

    Into H or A: a touch by that side. Into D: the ball leaving the pitch or the tracking. A
    change never moves past the midpoint to its neighbours, so changes cannot cross or merge and
    the number of possessions is unchanged; with no event in range it stays where it is.
    """
    out = states.copy()
    for _, first, last in blocks:
        block = states[first:last + 1]
        changes = (np.flatnonzero(block[1:] != block[:-1]) + 1).tolist()
        for k, change in enumerate(changes):
            before, after = block[change - 1], block[change]
            low = (changes[k - 1] + change) // 2 + 1 if k else 1
            high = (change + changes[k + 1]) // 2 if k + 1 < len(changes) else len(block) - 1
            low, high = max(low, change - window), min(high, change + window)
            if after == DEAD:
                candidates = np.flatnonzero(onset[first + low:first + high + 1]) + low
            else:
                candidates = np.flatnonzero(
                    touch_team[first + low:first + high + 1] == after) + low
            if not len(candidates):
                continue
            target = int(candidates[np.argmin(np.abs(candidates - change))])
            if target < change:
                out[first + target:first + change] = after
            elif target > change:
                out[first + change:first + target] = before
    return out


def out_until_touch(ball_raw, touch_team, pitch):
    """True from each moment the ball leaves the pitch until the next touch (the restart),
    at most STICKY_MAX_S: players keep moving after the ball goes out, so the walking-pace
    signal alone notices the stoppage late."""
    length, width = pitch
    margin_x, margin_y = OUT_OF_PITCH_MARGIN_M / length, OUT_OF_PITCH_MARGIN_M / width
    out = np.isfinite(ball_raw[:, 0]) & (
        (ball_raw[:, 0] < -margin_x) | (ball_raw[:, 0] > 1 + margin_x)
        | (ball_raw[:, 1] < -margin_y) | (ball_raw[:, 1] > 1 + margin_y))
    touches = np.flatnonzero(touch_team >= 0)
    sticky = np.zeros(len(ball_raw), dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], out.astype(np.int8), [0]))))
    for start, end in zip(edges[::2], edges[1::2]):
        following = touches[np.searchsorted(touches, end):]
        stop = following[0] if len(following) else len(ball_raw)
        sticky[start:min(stop, start + int(STICKY_MAX_S * FPS))] = True
    return sticky


def tracking_emissions(ball, d_home, d_away, pitch, dead_extra=0.0, gain=1.0, p_learned=None,
                       learned_mode="replace", decay_m=DECAY_M):
    """log P(observation | state) up to a per-frame constant, shape (n, 3)."""
    length, width = pitch
    seen = ~np.isnan(ball[:, 0])
    margin_x, margin_y = OUT_OF_PITCH_MARGIN_M / length, OUT_OF_PITCH_MARGIN_M / width
    out = seen & ((ball[:, 0] < -margin_x) | (ball[:, 0] > 1 + margin_x)
                  | (ball[:, 1] < -margin_y) | (ball[:, 1] > 1 + margin_y))

    dead = DEAD_BASE + np.where(seen, 0.0, 0.05) + np.where(
        stationary_ball(ball, pitch), STATIONARY_DEAD, 0.0) + dead_extra
    dead = np.where(out, 0.95, np.clip(dead, 0.05, 0.95))

    near = np.minimum(d_home, d_away)
    known = seen & np.isfinite(near)
    margin = np.where(known, np.clip(d_away - d_home, -20, 20), 0.0)
    home_share = 1.0 / (1.0 + np.exp(-margin / CONTROL_SCALE_M))
    confidence = np.where(known, np.exp(-np.maximum(0.0, near - CONTROL_RADIUS_M) / decay_m), 0.0)
    p_home = np.clip(0.5 + confidence * (home_share - 0.5), 0.02, 0.98)
    if p_learned is not None:
        use = known & np.isfinite(p_learned)
        learned = np.clip(np.nan_to_num(p_learned, nan=0.5), LEARNED_CLIP, 1 - LEARNED_CLIP)
        if learned_mode == "blend":
            # Average of the two log-odds: each rule gets half the say.
            mixed = 0.5 * (np.log(p_home / (1 - p_home)) + np.log(learned / (1 - learned)))
            learned = 1.0 / (1.0 + np.exp(-mixed))
        p_home = np.where(use, learned, p_home)
    if not np.isscalar(gain) or gain != 1.0:
        log_odds = np.log(p_home / (1 - p_home)) * gain
        p_home = np.clip(1.0 / (1.0 + np.exp(-log_odds)), 0.01, 0.99)

    emit = np.empty((len(ball), 3))
    emit[:, HOME] = np.log((1 - dead) * p_home + FLOOR)
    emit[:, AWAY] = np.log((1 - dead) * (1 - p_home) + FLOOR)
    emit[:, DEAD] = np.log(dead + FLOOR)
    return emit, known, d_home < d_away


def side_of_tag(value, names):
    """H, A or None for a Smart Tagging Team value, matched against the ATD club names."""
    value = (value or "").strip().lower()
    if not value or value == "n/a":
        return None
    home, away = (name.strip().lower() for name in names)
    if value in home or home in value:
        return HOME
    if value in away or away in value:
        return AWAY
    return {"barcelona": HOME, "atletic club": AWAY}.get(value)


def tag_emissions(events, names, n_frames, known, home_nearer, use_phases=True):
    """Additive log priors from Smart Tagging, shape (n, 3). Tempered with the tracking evidence.

    use_phases=False keeps only the set pieces: a dead-ball prior on the run-up to each restart
    and the restarting side after it.

    Whether a phase's Team is the side on the ball or the side defending depends on the code
    ("DEFENDING IN ..." presumably names the defenders). Rather than assume, each code is
    oriented by how often its Team agrees with the nearest player across the whole match, and
    weighted by how strongly it agrees -- a code that is a coin flip contributes nothing.
    """
    prior = np.zeros((n_frames, 3))
    phases = {}
    for event in events:
        code = (event["code"] or "").strip().upper()
        start, end = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        if end <= start:
            continue
        side = side_of_tag(event["tags"].get("Team"), names)
        if code == "SET PIECES":
            split = start + int(SET_PIECE_DEAD_SHARE * (end - start))
            prior[start:split, DEAD] += SET_PIECE_DEAD_NATS
            if side is not None:
                prior[split:end, side] += TAG_TEAM_NATS
        elif side is not None and use_phases:
            phases.setdefault(code, []).append((start, end, side))

    for code, spans in phases.items():
        agree = total = 0
        for start, end, side in spans:
            mask = known[start:end]
            nearer = np.where(home_nearer[start:end][mask], HOME, AWAY)
            agree += int(np.sum(nearer == side))
            total += int(mask.sum())
        if total < MIN_CALIBRATION_FRAMES:
            continue
        share = agree / float(total)
        weight = min(1.0, abs(share - 0.5) / 0.2)
        for start, end, side in spans:
            team = side if share >= 0.5 else 1 - side
            prior[start:end, team] += weight * TAG_TEAM_NATS
    return prior


def second_half_kickoff(events):
    """First frame of the tagged second-half kickoff window, or None."""
    for event in events:
        if ((event["code"] or "").strip().lower() == load_smart_tagging.KICKOFF_CODE
                and (event["tags"].get("Type") or "").strip().lower()
                == load_smart_tagging.KICKOFF_TYPE
                and event["tags"].get("Half") == "2nd Half"):
            return event["start_frame"]
    return None


def transitions(chain=1):
    """log transition matrix over 3 * chain sub-states, and the label of each sub-state.

    Each label is a chain of `chain` sub-states sharing its evidence, which turns the duration of
    a stay from geometric -- where one frame is the single most likely length -- into an Erlang
    with the same mean, so very short possessions become unlikely. chain=1 is the plain HMM.
    """
    to_dead = 1.0 / (MEAN_PLAY_BETWEEN_STOPPAGES_S * FPS)
    switch = max(1.0 / (MEAN_POSSESSION_S * FPS) - to_dead, 1e-5)
    revive = 1.0 / (MEAN_STOPPAGE_S * FPS)
    # Exit probability of each label, and where an exit goes.
    leave = {HOME: switch + to_dead, AWAY: switch + to_dead, DEAD: revive}
    exits = {HOME: {AWAY: switch, DEAD: to_dead}, AWAY: {HOME: switch, DEAD: to_dead},
             DEAD: {HOME: revive / 2, AWAY: revive / 2}}
    n = 3 * chain
    matrix = np.zeros((n, n))
    for label in (HOME, AWAY, DEAD):
        step = min(chain * leave[label], 0.5)  # per sub-state, so the mean stay is unchanged
        for k in range(chain):
            state = label * chain + k
            matrix[state, state] = 1 - step
            if k + 1 < chain:
                matrix[state, state + 1] = step
            else:
                for target, rate in exits[label].items():
                    matrix[state, target * chain] = step * rate / leave[label]
    with np.errstate(divide="ignore"):
        return np.log(matrix), np.repeat(np.arange(3), chain)


def viterbi(emit, log_transition, label_of_state):
    """Most likely label sequence for one contiguous block."""
    n, n_states = len(emit), len(label_of_state)
    emit = emit[:, label_of_state]
    back = np.zeros((n, n_states), dtype=np.int16)
    score = emit[0] + np.log(1.0 / n_states)
    for t in range(1, n):
        candidates = score[:, None] + log_transition
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + emit[t]
    path = np.empty(n, dtype=np.int16)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return label_of_state[path].astype(np.int8)


def decode(emit, blocks, n_frames, chain=1):
    states = np.full(n_frames, DEAD, dtype=np.int8)
    log_transition, label_of_state = transitions(chain)
    for _, first, last in blocks:
        states[first:last + 1] = viterbi(emit[first:last + 1], log_transition, label_of_state)
    return states


def build_parser():
    """The command-line options, also used by solution/pipeline.py for its defaults."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--prefix", default="hmm", help="file name prefix for the variants")
    parser.add_argument("--temper", type=float, nargs="+", default=[TEMPER_FRAMES],
                        help="frames per independent observation; one variant per value")
    parser.add_argument("--tags", nargs="*", default=[], choices=["tags", "setpieces"],
                        help="also write Smart Tagging variants: all tags, or set pieces only")
    parser.add_argument("--walk-ms", type=float, default=WALK_MS,
                        help="median player speed below which play looks stopped")
    parser.add_argument("--snap", type=int, nargs="+", default=[SNAP_FRAMES],
                        help="snap boundaries to events within +/- this many frames, one variant "
                             "per value (0 = off)")
    parser.add_argument("--chain", type=int, default=1,
                        help="sub-states per label: Erlang possession lengths (1 = plain HMM)")
    parser.add_argument("--learned", default="off", choices=["off", "replace", "blend"],
                        help="learned local model for P(home): replace the hand-written rule "
                             "where the ball is seen, or blend the two")
    parser.add_argument("--no-touch", dest="touch", action="store_false",
                        help="turn off touch detection (nearest-player evidence boosted at "
                             "touches, damped in flight)")
    parser.add_argument("--decay-m", type=float, default=DECAY_M,
                        help="metres over which nearest-player evidence fades beyond control range")
    parser.add_argument("--sticky-out", action="store_true",
                        help="keep D evidence from the ball leaving the pitch until the next touch")
    parser.add_argument("--broadcast", action="store_true",
                        help="fill tactical ball gaps from the synced broadcast tracking")
    parser.add_argument("--shape", action="store_true",
                        help="add self-supervised team-shape evidence (works without the ball)")
    parser.add_argument("--dead", nargs="+", default=["_walk"],
                        choices=["", "_gap", "_walk", "_gapwalk", "_walkline"],
                        help="extra dead-ball evidence sets, one variant each ('' = none)")
    return parser


def predict(args):
    """{output name: states} for every variant the options ask for."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    team_of_track = baseline.team_of_each_track(meta_path)
    names = baseline.team_names(meta_path)
    blocks = half_video_blocks(GAME)
    events = load_smart_tagging.load(GAME)

    ball_raw, xy, _ = parse_atd(raw_path, n_frames, len(team_of_track))
    if args.broadcast:
        import broadcast_sync
        ball_raw = broadcast_sync.broadcast_filled_ball(
            ball_raw, xy, parse_atd, baseline.team_of_each_track,
            os.path.join(args.out, "broadcast_diagnostics.txt"))

    kickoff_2nd = second_half_kickoff(events)
    pre_kickoff = np.zeros(n_frames, dtype=bool)
    if kickoff_2nd is not None:
        pre_kickoff[blocks[1][1]:kickoff_2nd] = True

    ball = interpolate_gaps(ball_raw, blocks, MAX_INTERP_GAP)
    d_home, d_away = nearest_distances(ball, xy, team_of_track, pitch)
    gap = GAP_DEAD / (1 + np.exp(-(gap_seconds(ball_raw, blocks) - GAP_DEAD_S) / GAP_SOFTNESS_S))
    gap[np.isfinite(ball_raw[:, 0])] = 0.0
    motion = team_motion(xy, pitch)
    walk = WALK_DEAD / (1 + np.exp(-(args.walk_ms - np.nan_to_num(motion, nan=args.walk_ms))
                                   / WALK_SOFTNESS_MS))
    line = np.where(line_exit(ball_raw, blocks, pitch), LINE_DEAD, 0.0)
    dead_sets = {"": 0.0, "_gap": gap, "_walk": walk, "_gapwalk": gap + walk,
                 "_walkline": walk + line}
    dead_sets = {key: dead_sets[key] for key in args.dead}

    gain, touch_team = touch_gain(ball, d_home, d_away, pitch)
    gain = gain if args.touch else 1.0
    p_learned = None
    if args.learned != "off":
        p_learned = learned_home_probability(
            local_features(ball, xy, team_of_track, pitch),
            between_touch_labels(touch_team, blocks, BETWEEN_TOUCH_MAX_GAP))
    onset = dead_onsets(ball_raw, pitch)
    if args.sticky_out:
        sticky = np.where(out_until_touch(ball_raw, touch_team, pitch), STICKY_DEAD, 0.0)
        dead_sets = {key: extra + sticky for key, extra in dead_sets.items()}

    shape = 0.0
    if args.shape:
        near = np.minimum(d_home, d_away)
        weak = (np.isfinite(ball[:, 0]) & (near < WEAK_NEAR_M)
                & (np.abs(d_away - d_home) > WEAK_MARGIN_M)
                & (np.nan_to_num(motion, nan=0.0) > args.walk_ms))
        p_shape = shape_home_probability(shape_features(xy, team_of_track, pitch),
                                         d_home < d_away, weak, blocks)
        shape = np.zeros((n_frames, 3))
        # Centred on 0.5 so the shape only moves H against A and never pushes towards D.
        shape[:, HOME] = SHAPE_WEIGHT * np.log(2 * p_shape)
        shape[:, AWAY] = SHAPE_WEIGHT * np.log(2 * (1 - p_shape))

    variants = {}
    for dead_key, dead_extra in dead_sets.items():
        emit, known, home_nearer = tracking_emissions(ball, d_home, d_away, pitch, dead_extra,
                                                      gain, p_learned, args.learned,
                                                      args.decay_m)
        emit = emit + shape
        tag_sets = {"": 0.0}
        for mode in args.tags:
            tag_sets["_" + mode] = tag_emissions(events, names, n_frames, known, home_nearer,
                                                 use_phases=(mode == "tags"))
        for tag_key, tags in tag_sets.items():
            for temper in args.temper:
                tempered = (emit + tags) / temper
                tempered[pre_kickoff, HOME] = tempered[pre_kickoff, AWAY] = np.log(FLOOR)
                variants["hmm{}{}_t{:g}".format(dead_key, tag_key, temper)] = tempered

    results = {}
    for name, emit in variants.items():
        states = decode(emit, blocks, n_frames, args.chain)
        for window in args.snap:
            key = name if len(args.snap) == 1 else "{}_snap{}".format(name, window)
            results[key] = (snap_boundaries(states, blocks, touch_team, onset, window)
                            if window else states)
    return results


if __name__ == "__main__":
    args = build_parser().parse_args()
    for name, states in predict(args).items():
        name = name.replace("hmm", args.prefix, 1)
        baseline.write_submission(os.path.join(args.out, "{}.csv".format(name)), LABELS[states])
        print("wrote {}".format(name))
