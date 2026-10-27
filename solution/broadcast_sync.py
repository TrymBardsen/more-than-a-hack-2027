"""Fill tactical ball gaps from the broadcast tracking, after syncing the two recordings.

The broadcast ATD is a separate recording on its own timeline (164488 frames against 199319),
with no mapping to the tactical video, and a TV feed has cuts, replays and close-ups, so no single
offset can align it. Syncing is done piecewise and without the ball:

  1. Shape signal. Per frame and per recording: centroid and spread of all tracked players in
     metres, ignoring team labels (the two exports need not agree on them). Smoothed, 5 fps.
  2. Window matching. Each 30 s broadcast window is slid along the whole tactical signal (an FFT
     makes that one pass) and placed where the mean squared difference is smallest.
  3. Acceptance. A placement is kept only if its RMS error is small in metres, clearly beats any
     placement more than 5 s away, agrees with a neighbouring window, and -- where both feeds see
     the ball -- puts the two balls close together. Replays and close-ups fail this and are
     dropped.
  4. Orientation. The broadcast coordinates might be mirrored; all four flips are tried on the
     shape signal and the best is used.
  5. Fill. A tactical frame without a ball takes the broadcast ball from the matched frame.

Never fills a frame where the tactical tracking already has a ball.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
STEP = 5                    # samples every 5 frames: 5 fps is plenty for a 30 s shape signal
WINDOW_S = 30.0
HOP_S = 10.0
MIN_PLAYERS = 10
MIN_VALID_SHARE = 0.8
RMS_MAX_M = 2.5
DISTINCT_RATIO = 0.5        # best cost must be under half the best one more than 5 s away
DISTINCT_S = 5.0
NEIGHBOUR_AGREE_S = 1.0
REFINE_FRAMES = 10
BALL_AGREE_M = 3.0
MIN_BALL_PAIRS = 25
ORIENTATIONS = ((False, False), (True, False), (False, True), (True, True))
# Subset matching, grown from the shape anchors (see extend_from_anchors).
MIN_SUBSET = 6
EXTEND_SEARCH = 2 * FPS
SUBSET_MAX_M = 1.5
FRAME_MAX_M = 2.0
ANCHOR_FRAMES = 4
ANCHOR_STEP = 10
ORIENTATION_SAMPLES = 16
ANCHOR_EVERY = 6


def shape_signal(xy, pitch):
    """(n, 4) centroid x, centroid y, spread x, spread y of all tracked players, metres; NaN if
    fewer than MIN_PLAYERS are tracked."""
    length, width = pitch
    x, y = xy[..., 0] * length, xy[..., 1] * width
    count = np.isfinite(x).sum(axis=1)
    ok = count >= MIN_PLAYERS
    safe = np.maximum(count, 1)
    mean_x = np.nansum(x, axis=1) / safe
    mean_y = np.nansum(y, axis=1) / safe
    spread_x = np.sqrt(np.nansum((x - mean_x[:, None]) ** 2, axis=1) / safe)
    spread_y = np.sqrt(np.nansum((y - mean_y[:, None]) ** 2, axis=1) / safe)
    out = np.column_stack([mean_x, mean_y, spread_x, spread_y])
    out[~ok] = np.nan
    return out


def flip(xy, flip_x, flip_y):
    out = xy.copy()
    if flip_x:
        out[..., 0] = 1.0 - out[..., 0]
    if flip_y:
        out[..., 1] = 1.0 - out[..., 1]
    return out


def smooth_downsample(signal, step):
    """Mean over each block of `step` frames, ignoring NaN rows; NaN where a block has none."""
    n = len(signal) // step
    blocks = signal[:n * step].reshape(n, step, -1)
    valid = np.isfinite(blocks).all(axis=2)
    counts = valid.sum(axis=1)
    sums = np.where(valid[..., None], blocks, 0.0).sum(axis=1)
    out = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], np.nan)
    return out


def sliding_cost(series, window):
    """Mean squared difference of `window` (W x k) against every W-long slice of `series`.

    Both must be NaN-free (see fill_nan); callers mask out slices with too little real data.
    """
    n, w = len(series), len(window)
    size = 1 << int(np.ceil(np.log2(n + w)))
    cross = np.zeros(n - w + 1)
    for j in range(series.shape[1]):
        f = np.fft.rfft(series[:, j], size)
        g = np.fft.rfft(window[::-1, j], size)
        cross += np.fft.irfft(f * g, size)[w - 1:n]
    squares = np.concatenate(([0.0], np.cumsum((series ** 2).sum(axis=1))))
    series_term = squares[w:] - squares[:-w]
    window_term = float((window ** 2).sum())
    return np.maximum(series_term - 2 * cross + window_term, 0.0) / (w * series.shape[1])


def fill_nan(signal):
    """Linear interpolation over NaN rows, column by column; edges take the nearest value."""
    out = signal.copy()
    index = np.arange(len(signal))
    for j in range(signal.shape[1]):
        good = np.isfinite(signal[:, j])
        if good.sum() >= 2:
            out[:, j] = np.interp(index, index[good], signal[good, j])
        else:
            out[:, j] = 0.0
    return out


def match_windows(tactical, broadcast):
    """[(broadcast_start_sample, offset_samples, rms_m)] for every accepted window placement.

    offset maps a broadcast sample s to tactical sample s + offset.
    """
    w, hop = int(WINDOW_S * FPS / STEP), int(HOP_S * FPS / STEP)
    distinct = int(DISTINCT_S * FPS / STEP)
    tactical_valid = np.isfinite(tactical).all(axis=1)
    tactical_filled = fill_nan(tactical)
    out = []
    for start in range(0, len(broadcast) - w + 1, hop):
        window = broadcast[start:start + w]
        valid = np.isfinite(window).all(axis=1)
        if valid.mean() < MIN_VALID_SHARE:
            continue
        cost = sliding_cost(tactical_filled, fill_nan(window))
        coverage = np.convolve(tactical_valid.astype(float), np.ones(w), mode="valid") / w
        cost[coverage < MIN_VALID_SHARE] = np.inf
        best = int(np.argmin(cost))
        if not np.isfinite(cost[best]):
            continue
        away = cost.copy()
        away[max(0, best - distinct):best + distinct + 1] = np.inf
        rms = float(np.sqrt(cost[best]))
        if rms < RMS_MAX_M and cost[best] < DISTINCT_RATIO * away.min():
            out.append((start, best - start, rms))
    return out


def consistent(matches):
    """Keep matches whose offset agrees with an accepted neighbour's within NEIGHBOUR_AGREE_S."""
    tolerance = NEIGHBOUR_AGREE_S * FPS / STEP
    keep = []
    for i, (start, offset, rms) in enumerate(matches):
        neighbours = [matches[j][1] for j in (i - 1, i + 1) if 0 <= j < len(matches)]
        if any(abs(offset - other) <= tolerance for other in neighbours):
            keep.append((start, offset, rms))
    return keep


def ball_agrees(mapping, tactical_ball, broadcast_ball, pitch):
    """Drop the mapping inside any HOP_S span where both balls are seen and disagree."""
    length, width = pitch
    out = mapping.copy()
    hop = int(HOP_S * FPS)
    for b0 in range(0, len(mapping), hop):
        claim = np.arange(b0, min(b0 + hop, len(mapping)))
        target = mapping[claim]
        ok = target >= 0
        if not ok.any():
            continue
        tb = tactical_ball[target[ok]]
        bb = broadcast_ball[claim[ok]]
        both = np.isfinite(tb[:, 0]) & np.isfinite(bb[:, 0])
        if both.sum() < MIN_BALL_PAIRS:
            continue  # no way to check; the shape match alone stands
        gap = np.hypot((tb[both, 0] - bb[both, 0]) * length, (tb[both, 1] - bb[both, 1]) * width)
        if np.median(gap) > BALL_AGREE_M:
            out[claim] = -1
    return out


def subset_distance(broadcast_m, tactical_m):
    """Per frame: median over the broadcast's visible players of the distance to the nearest
    tactical player, in metres. NaN where fewer than MIN_SUBSET players are visible.

    One-sided on purpose: a TV camera sees only part of the pitch (a median of 15 players
    against 21 here), so every broadcast player should have a tactical counterpart but not the
    other way round. Inputs are (..., frames, slots, 2) in metres; leading axes broadcast.
    """
    diff = broadcast_m[..., :, :, None, :] - tactical_m[..., :, None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=-1))
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest = distance.min(axis=-1)
    visible = np.isfinite(broadcast_m[..., 0])
    nearest = np.where(visible & np.isfinite(nearest), nearest, np.nan)
    enough = visible.sum(axis=-1) >= MIN_SUBSET
    with np.errstate(all="ignore"):
        median = np.nanmedian(np.where(enough[..., None], nearest, np.nan), axis=-1)
    return median


def global_anchor(b0, tactical_m, broadcast_m):
    """(offset_frames, cost) placing the broadcast window at b0 anywhere in the tactical match.

    Uses ANCHOR_FRAMES frames spread over the window and every offset in ANCHOR_STEP steps, with
    the subset distance, so it works on a partial view. None unless the best placement is under
    SUBSET_MAX_M and clearly beats every placement more than DISTINCT_S away.
    """
    n_t = len(tactical_m)
    frames = b0 + np.linspace(0, WINDOW_S * FPS - 1, ANCHOR_FRAMES).astype(int)
    frames = frames[frames < len(broadcast_m)]
    if len(frames) < ANCHOR_FRAMES:
        return None
    visible = np.isfinite(broadcast_m[frames, :, 0]).sum(axis=1)
    if (visible < MIN_SUBSET).any():
        return None
    offsets = np.arange(-frames.min(), n_t - frames.max(), ANCHOR_STEP)
    if not len(offsets):
        return None
    cost = np.zeros(len(offsets))
    for frame in frames:
        cost += subset_distance(broadcast_m[frame][None], tactical_m[frame + offsets])
    cost /= len(frames)
    cost = np.where(np.isfinite(cost), cost, np.inf)
    best = int(np.argmin(cost))
    if not np.isfinite(cost[best]) or cost[best] > SUBSET_MAX_M:
        return None
    away = np.abs(offsets - offsets[best]) > DISTINCT_S * FPS
    if away.any() and cost[best] >= DISTINCT_RATIO * cost[away].min():
        return None
    return int(offsets[best]), float(cost[best])


def global_anchors(tactical_m, broadcast_m, starts):
    """[(start_sample, offset_samples, cost)] for every window start (in frames) that anchors."""
    out = []
    for b0 in starts:
        found = global_anchor(b0, tactical_m, broadcast_m)
        if found is not None:
            out.append((b0 // STEP, found[0] // STEP, found[1]))
    return out


def extend_from_anchors(anchors, tactical_m, broadcast_m):
    """Frame mapping grown from the shape-matched anchors with the subset distance.

    anchors: [(broadcast_start_sample, offset_samples, rms)] from match_windows. Every window
    tries offsets within +/- EXTEND_SEARCH frames of the previous accepted offset and of the
    nearest anchors, scored on frames sampled every STEP. A window is accepted when its median
    subset distance is under SUBSET_MAX_M and the same offset 5 s either way is clearly worse;
    within it, each frame is then kept only if its own subset distance is under FRAME_MAX_M,
    which drops cuts to close-ups and replays inside an otherwise good window.
    """
    n_b, n_t = len(broadcast_m), len(tactical_m)
    w, hop = int(WINDOW_S * FPS), int(HOP_S * FPS)
    anchor_frames = np.array([a[0] * STEP for a in anchors])
    anchor_offsets = np.array([a[1] * STEP for a in anchors])
    mapping = np.full(n_b, -1, dtype=np.int64)
    previous = None
    shifts = np.arange(-EXTEND_SEARCH, EXTEND_SEARCH + 1, STEP)
    away = int(DISTINCT_S * FPS)

    def window_cost(b0, offsets):
        frames = np.arange(b0, min(b0 + w, n_b), STEP)
        costs = []
        for offset in offsets:
            target = frames + offset
            ok = (target >= 0) & (target < n_t)
            if ok.mean() < MIN_VALID_SHARE:
                costs.append(np.inf)
                continue
            per_frame = subset_distance(broadcast_m[frames[ok]], tactical_m[target[ok]])
            good = np.isfinite(per_frame)
            costs.append(np.median(per_frame[good]) if good.mean() >= MIN_VALID_SHARE
                         else np.inf)
        return np.array(costs)

    for b0 in range(0, n_b - w + 1, hop):
        centre = b0 + w // 2
        bases = set()
        if previous is not None:
            bases.add(previous)
        if len(anchor_frames):
            order = np.argsort(np.abs(anchor_frames - centre))[:2]
            bases.update(int(anchor_offsets[i]) for i in order)
        if not bases:
            continue
        candidates = np.unique(np.concatenate([base + shifts for base in bases]))
        costs = window_cost(b0, candidates)
        best = int(np.argmin(costs))
        if not np.isfinite(costs[best]) or costs[best] > SUBSET_MAX_M:
            continue
        offset = int(candidates[best])
        refine = np.arange(offset - STEP, offset + STEP + 1)
        refine_costs = window_cost(b0, refine)
        offset = int(refine[int(np.argmin(refine_costs))])
        best_cost = float(refine_costs.min())
        rivals = window_cost(b0, [offset - away, offset + away])
        if best_cost >= DISTINCT_RATIO * rivals.min():
            continue
        previous = offset
        claim = np.arange(b0 + (w - hop) // 2, min(b0 + (w + hop) // 2, n_b))
        target = claim + offset
        ok = (target >= 0) & (target < n_t)
        per_frame = np.full(len(claim), np.nan)
        per_frame[ok] = subset_distance(broadcast_m[claim[ok]], tactical_m[target[ok]])
        keep = ok & (per_frame < FRAME_MAX_M)
        mapping[claim[keep]] = target[keep]
    return mapping


def report(lines, text):
    """Collect an aggregate diagnostic line (counts and shares only, never data values).

    The user approved the assistant reading these lines, so they are written to their own file
    next to the results instead of only to the hidden run log.
    """
    lines.append(text)
    print(text)


def align(tactical_xy, broadcast_xy, pitch, lines):
    """(mapping, (flip_x, flip_y)) choosing the orientation with the most accepted windows."""
    tactical = smooth_downsample(shape_signal(tactical_xy, pitch), STEP)
    best = None
    for flips in ORIENTATIONS:
        broadcast = smooth_downsample(shape_signal(flip(broadcast_xy, *flips), pitch), STEP)
        raw = match_windows(tactical, broadcast)
        matches = consistent(raw)
        report(lines, "orientation {}: windows {}, passed shape test {}, consistent {}".format(
            flips, len(broadcast) // int(HOP_S * FPS / STEP), len(raw), len(matches)))
        score = (len(matches), -np.median([m[2] for m in matches]) if matches else 0.0)
        if best is None or score > best[0]:
            best = (score, flips, matches)
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)
    tactical_m = tactical_xy * scale
    hop = int(HOP_S * FPS)
    starts = np.arange(0, len(broadcast_xy) - int(WINDOW_S * FPS), hop)

    # The shape test is unreliable on a partial view, so orientation is also judged by how many
    # sample windows the subset distance can anchor, and the better of the two verdicts wins.
    sample = starts[np.linspace(0, len(starts) - 1, ORIENTATION_SAMPLES).astype(int)]
    subset_scores = {}
    for flips in ORIENTATIONS:
        found = global_anchors(tactical_m, flip(broadcast_xy, *flips) * scale, sample)
        subset_scores[flips] = len(found)
        report(lines, "orientation {}: subset anchors in {} of {} sample windows".format(
            flips, len(found), len(sample)))
    subset_flips = max(ORIENTATIONS, key=lambda f: subset_scores[f])
    flips = best[1] if best[0][0] > subset_scores[subset_flips] else subset_flips

    broadcast_m = flip(broadcast_xy, *flips) * scale
    shape_anchors = best[2] if flips == best[1] else []
    anchors = sorted(shape_anchors + global_anchors(tactical_m, broadcast_m,
                                                    starts[::ANCHOR_EVERY]))
    mapping = extend_from_anchors(anchors, tactical_m, broadcast_m)
    report(lines, "subset matching: {} anchors ({} from shape) grown to {:.1f}% of broadcast "
           "frames".format(len(anchors), len(shape_anchors), 100 * np.mean(mapping >= 0)))
    return mapping, flips, anchors


def fill_ball(tactical_ball, mapping, broadcast_ball):
    """Tactical ball with gaps filled from the broadcast where the mapping reaches."""
    out = tactical_ball.copy()
    source = np.flatnonzero(mapping >= 0)
    target = mapping[source]
    usable = np.isnan(out[target, 0]) & np.isfinite(broadcast_ball[source, 0])
    # The first broadcast frame mapped to a tactical frame wins; later duplicates are ignored.
    target, order = np.unique(target[usable], return_index=True)
    out[target] = broadcast_ball[source[usable][order]]
    return out


def broadcast_filled_ball(tactical_ball, tactical_xy, parse_atd, team_of_each_track,
                          diagnostics_path=None):
    """End to end: load the broadcast ATD, align it, and return the filled tactical ball.

    diagnostics_path, if given, receives the aggregate diagnostic lines (see report()).
    """
    lines = []
    pitch = GAMES[GAME]["pitch"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, source="broadcast")
    n_broadcast = source_frames("broadcast", GAME)
    slots = len(team_of_each_track(meta_path))
    broadcast_ball, broadcast_xy, _ = parse_atd(raw_path, n_broadcast, slots)
    tracked = np.isfinite(broadcast_xy[..., 0]).sum(axis=1)
    report(lines, "broadcast: median tracked players per frame {:.0f} (tactical {:.0f})".format(
        np.median(tracked[tracked > 0]) if (tracked > 0).any() else 0,
        np.median(np.isfinite(tactical_xy[..., 0]).sum(axis=1))))
    mapping, flips, matches = align(tactical_xy, broadcast_xy, pitch, lines)
    broadcast_ball = flip(broadcast_ball, *flips)
    checked = ball_agrees(mapping, tactical_ball, broadcast_ball, pitch)
    filled = fill_ball(tactical_ball, checked, broadcast_ball)
    report(lines, "broadcast: chosen orientation {}, mapped {:.1f}% of broadcast frames ({:.1f}% "
           "after the ball check), tactical frames with a ball {:.1f}% -> {:.1f}%".format(
               flips, 100 * np.mean(mapping >= 0), 100 * np.mean(checked >= 0),
               100 * np.mean(np.isfinite(tactical_ball[:, 0])),
               100 * np.mean(np.isfinite(filled[:, 0]))))
    if diagnostics_path:
        os.makedirs(os.path.dirname(diagnostics_path) or ".", exist_ok=True)
        with open(diagnostics_path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    return filled
