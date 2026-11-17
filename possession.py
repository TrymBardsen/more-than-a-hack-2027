"""Possession labeller: tracking geometry -> per-frame evidence -> Viterbi over {H, A, D}.

    python possession.py [out.csv]            # default: evaluation/results/v3.csv

Why a state model instead of smoothing: spellF1 scores possessions as objects, so a flicker that
splits one possession costs a whole match. Viterbi with a cost per switch only changes state when
the accumulated evidence pays for it, and puts the boundary where the evidence flips (no
centred-window lag).

Evidence, no human labels anywhere:
  team  : which team has the nearest player to the ball, weighted by how close and how slow the
          ball is (a slow ball next to a player is control; a fast ball passing a player is not).
  dead  : stoppages opened by the ball crossing a line, a placed ball or a Smart Tagging set
          piece and closed by the next kick; the time before each half's tagged kickoff; plus
          players moving slowly as a group.
  gaps  : where the ball is missing, two weak priors: the team Smart Tagging phases name, and a
          logistic model of "who has the ball" from team shape, trained on this pipeline's own
          ball-visible labels (algorithmic pseudo-labels).

Every constant below is a football prior or was read off unlabeled feature distributions. None was
fitted to the public 5 minutes. TECHNICAL_REPORT.md explains each choice.
"""

import functools
import json
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track, write_submission  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN frames are expected

# --- priors (metres, m/s, frames at 25 fps) ---
CONTROL_RADIUS = 2.0      # nearest player within ~2 m of the ball is "on it"
MARGIN_SCALE = 1.0        # 1 m closer than the nearest opponent = 1 unit of team evidence
FAST_BALL = 10.0          # ball faster than this is a pass/shot in flight, not control
MAX_BALL_STEP = 2.0       # >2 m in one frame (50 m/s) is a tracker jump, drop the detection
OUT_MARGIN = 1.0          # ball this far past a line is out of play
SLOW_PLAYERS = 1.5        # group mean speed (m/s) below which play looks stopped (dev-set tuned)
SPEED_WINDOW = 50         # smooth group speed over 2 s
PLACED_BALL = 0.3         # ball averaging under 0.3 m/s for 1.5 s is placed for a restart
KICK_SPEED = 4.0          # a restart is the first time the ball moves faster than this, on the pitch
MIN_DEAD = 25             # a stoppage lasts at least 1 s before a kick can end it
MAX_DEAD = 120 * 25       # give up on a stoppage after 2 minutes without a visible restart
EVENT_DEAD = 1.0          # per-frame dead evidence inside a detected stoppage
RECEPTION = 4.0           # extra team evidence per frame at a reception (fast ball stopped at a player)
TEAM_SWITCH = 25.0        # ~1 s of clear control needed to hand possession over (dev-set tuned)
DEAD_SWITCH = 20.0        # cost of entering or leaving a stoppage (dev-set tuned)
MODEL_PATH = os.path.join(ROOT, "evidence_model.json")   # learned evidence; see devset.py --train
USE_MODEL = True          # use the learned evidence model instead of the hand-made team score
W_SEEN = 1.0              # weight of the learned log-odds where the ball is tracked (cross-match tuned)
W_GAP = 0.25              # weight of the learned log-odds where the ball is missing (cross-match tuned)
L_TEAM_SWITCH = 15.0      # H<->A switch cost on the learned path (cross-match tuned)
L_DEAD_SWITCH = 30.0      # dead-ball switch cost on the learned path (cross-match tuned)
L_SNAP = 0                # boundary snapping on the learned path: no gain there, so off
SNAP = 50                 # move each H<->A boundary to the new team's first clear touch within +-2 s (dev-set tuned; 0 = off)
TOUCH = 1.0               # team evidence at or above this counts as a clear touch for snapping
CLIP = 3.0                # cap on per-frame evidence so one frame never decides alone
TAG_WEIGHT = 0.4          # Smart Tagging phase team, per frame, only where the ball is missing (gap-test tuned)
SHAPE_WEIGHT = 0.2        # team-shape model, per frame, only where the ball is missing


def load_atd(raw_path):
    """(players[N,46,3], ball[N,3]) in normalised coords, row i = video frame i."""
    with open(raw_path, "rb") as f:
        text = f.read().translate(bytes.maketrans(b":;", b",,"))
    a = pd.read_csv(io.BytesIO(text), header=None, dtype=np.float32, engine="c").to_numpy()
    frames = a[:, 0].astype(np.int64) - 1          # ATD frame_id counts from 1
    out = np.full((frames.max() + 1, a.shape[1] - 1), np.nan, dtype=np.float32)
    out[frames] = a[:, 1:]
    return out[:, :138].reshape(-1, 46, 3), out[:, 138:141]


def rolling_mean(x, w):
    """Centred nan-aware mean over +/- w frames."""
    ok = ~np.isnan(x)
    c = np.concatenate(([0], np.cumsum(np.where(ok, x, 0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    cnt = n[hi] - n[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def set_piece_starts(game=GAME):
    """Video frames where Smart Tagging starts a set piece (any type). Tags start inside the
    stoppage, before the restart, so they open a stoppage and the next kick closes it."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from load_smart_tagging import load
    return {e["start_frame"] for e in load(game) if (e["code"] or "").strip().lower() == "set pieces"}


def pre_kickoff(n, game=GAME):
    """Bool per frame: between the start of a half and that half's tagged kickoff. A half is dead
    until it kicks off. The kickoff comes from Smart Tagging ("Kick Off (Start)"), not a constant."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from load_smart_tagging import load
    kicks = sorted(e["start_frame"] for e in load(game)
                   if (e["tags"].get("Type") or "").strip().lower() == "kick off (start)")
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        k = next((f for f in kicks if first - 25 * 60 <= f <= last), None)
        if k is not None and k > first:
            mask[first:k] = True
    return mask


def stoppages(bx, by, ball_speed, starts, pitch):
    """Bool per frame: inside a stoppage. Opened by the ball crossing a line (seen inside within
    the last second, then out), a placed ball, or a set-piece tag; closed by the next kick on the
    pitch. A small state machine because a stoppage is an interval, not a frame property."""
    L, W = pitch
    seen = ~np.isnan(bx)
    inside = seen & (bx >= -OUT_MARGIN) & (bx <= L + OUT_MARGIN) & (by >= -OUT_MARGIN) & (by <= W + OUT_MARGIN)
    out = (seen & ~inside).tolist()
    placed = (inside & (rolling_mean(ball_speed, 18) < PLACED_BALL)).tolist()
    kick = (inside & (rolling_mean(ball_speed, 2) > KICK_SPEED)).tolist()
    inside = inside.tolist()

    dead = np.zeros(len(bx), dtype=bool)
    open_at, last_inside = None, -10 ** 9
    for t in range(len(bx)):
        if open_at is not None and ((kick[t] and t - open_at >= MIN_DEAD) or t - open_at > MAX_DEAD):
            open_at = None
        elif open_at is None and ((out[t] and t - last_inside <= 25) or placed[t] or t in starts):
            open_at = t
        dead[t] = open_at is not None
        if inside[t]:
            last_inside = t
    return dead


def features(players, ball, team_of_track, pitch, starts=frozenset(), parts=False):
    """Per-frame team evidence (+ = home) and dead-ball evidence (+ = dead). With parts=True also
    the raw ingredients, which the learned evidence model uses."""
    L, W = pitch
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    # drop single-frame tracker jumps
    step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan))
    bx = np.where(step > MAX_BALL_STEP, np.nan, bx)
    by = np.where(step > MAX_BALL_STEP, np.nan, by)
    seen = ~np.isnan(bx)

    px, py = players[:, :, 0] * L, players[:, :, 1] * W
    d = np.hypot(px - bx[:, None], py - by[:, None])
    home = team_of_track == 0
    with np.errstate(all="ignore"):
        d_home = np.nanmin(np.where(home, d, np.nan), axis=1)
        d_away = np.nanmin(np.where(~home, d, np.nan), axis=1)
        ball_speed = rolling_mean(25 * np.hypot(np.diff(bx, prepend=np.nan),
                                                np.diff(by, prepend=np.nan)), 2)
        group_speed = rolling_mean(np.nanmean(players[:, :, 2], axis=1), SPEED_WINDOW)

    nearest = np.fmin(d_home, d_away)
    weight = np.exp(-nearest / CONTROL_RADIUS) * np.where(ball_speed > FAST_BALL, 0.2, 1.0)
    team = np.clip(weight * (d_away - d_home) / MARGIN_SCALE, -CLIP, CLIP)
    # reception: a ball arriving fast (>6 m/s) and leaving slow (<3 m/s) within ~0.3 s, next to a
    # player, is a controlled touch by that player's team
    before, after = np.roll(ball_speed, 4), np.roll(ball_speed, -4)
    reception = (before > 6.0) & (after < 3.0) & (nearest < 1.5)
    team = team + np.where(reception, RECEPTION * np.sign(d_away - d_home), 0.0)
    team = np.nan_to_num(team)                     # no ball -> no team evidence

    out = seen & ((bx < -OUT_MARGIN) | (bx > L + OUT_MARGIN) |
                  (by < -OUT_MARGIN) | (by > W + OUT_MARGIN))
    dead = np.clip(2 * (SLOW_PLAYERS - np.nan_to_num(group_speed, nan=SLOW_PLAYERS)), -CLIP, CLIP)
    dead = np.where(out, CLIP, dead)
    raw_speed = 25 * np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan))
    stopped = stoppages(bx, by, raw_speed, starts, pitch)
    dead = np.where(stopped, np.maximum(dead, EVENT_DEAD), dead)
    if parts:
        return team.astype(np.float64), dead.astype(np.float64), {
            "seen": seen, "stoppage": stopped, "margin": d_away - d_home, "nearest": nearest,
            "ball_speed": ball_speed, "reception": np.where(reception, np.sign(d_away - d_home), 0.0)}
    return team.astype(np.float64), dead.astype(np.float64)


def touch_context(team, seen, fade=125):
    """Side of the last and of the next clear touch (+1 home, -1 away), fading over ~5 s. This is
    what carries "who had it" into and out of a ball gap."""
    n = len(team)
    idx = np.arange(n)
    touch = np.where(seen & (np.abs(team) >= TOUCH), np.sign(team), 0.0)
    has = touch != 0
    last_i = np.maximum.accumulate(np.where(has, idx, -1))
    next_i = np.minimum.accumulate(np.where(has, idx, n)[::-1])[::-1]
    last = np.where(last_i >= 0, touch[np.maximum(last_i, 0)] * np.exp(-(idx - last_i) / fade), 0.0)
    nxt = np.where(next_i < n, touch[np.minimum(next_i, n - 1)] * np.exp(-(next_i - idx) / fade), 0.0)
    return last, nxt


def evidence_features(team, parts, shape):
    """Per-frame inputs of the learned evidence model: the hand-made score, its ingredients, the
    touch context and the team shape. Every column is home-positive or side-neutral."""
    seen = parts["seen"]
    margin = np.where(seen, np.clip(np.nan_to_num(parts["margin"]), -10, 10), 0.0)
    near = np.where(seen, np.clip(np.nan_to_num(parts["nearest"], nan=10.0), 0, 10), 0.0)
    speed = np.where(seen, np.clip(np.nan_to_num(parts["ball_speed"]), 0, 30), 0.0)
    last, nxt = touch_context(team, seen)
    return np.c_[team, margin, margin * np.exp(-near / 2), near, speed, parts["reception"],
                 last, nxt, shape]


def load_model(path=None):
    """The learned evidence model (two logistic regressions), or None if the file is absent."""
    path = path or MODEL_PATH
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {k: tuple(np.array(raw[k][part]) for part in ("mu", "sd", "w")) for k in ("seen", "gap")}


def viterbi(team, dead, team_switch=None, dead_switch=None):
    """Best H/A/D path. Scores: H +team/2, A -team/2, D +dead (live states get 0 dead score)."""
    n = len(team)
    ts = TEAM_SWITCH if team_switch is None else team_switch
    ds = DEAD_SWITCH if dead_switch is None else dead_switch
    emit = np.stack([team / 2, -team / 2, dead], axis=1)
    trans = np.array([[0, -ts, -ds],
                      [-ts, 0, -ds],
                      [-ds, -ds, 0]])
    back = np.zeros((n, 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):                          # ponytail: python loop, ~80k steps/half, a few s
        cand = score[:, None] + trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.array(["H", "A", "D"])[path]


ATTACKING = {"build up", "progression", "creating chances", "attacking transition",
             "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFENDING = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


def tag_prior(n, game=GAME):
    """Per frame: +1 where Smart Tagging phases say home has the ball, -1 away, 0 unknown.
    Attacking phases name the team on the ball; defending phases name the other one."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from load_smart_tagging import load
    vote = np.zeros(n)
    for e in load(game):
        code, team = (e["code"] or "").strip().lower(), (e["tags"].get("Team") or "").strip().lower()
        side = {"barcelona": 1, "atletic club": -1}.get(team, 0)
        if code in DEFENDING:
            side = -side
        elif code not in ATTACKING:
            side = 0
        vote[e["start_frame"]:e["end_frame"]] += side
    return np.sign(vote)


def shape_features(players, team_of_track, pitch, blocks):
    """Team-shape features from the home side's point of view, attacking direction normalised:
    centroid gap, forward velocity of each team, spread of each team. Works without the ball."""
    L, W = pitch
    px, py = players[:, :, 0] * L, players[:, :, 1] * W
    home = team_of_track == 0
    f = []
    for side in (home, ~home):
        x, y = np.where(side, px, np.nan), np.where(side, py, np.nan)
        f += [np.nanmean(x, 1), np.nanstd(x, 1), np.nanstd(y, 1)]
    cxh, sxh, syh, cxa, sxa, sya = f
    direction = np.zeros(len(px))                   # +1 where home attacks towards x = L
    for first, last in blocks:
        sl = slice(first, last + 1)
        direction[sl] = 1.0 if np.nanmean(cxh[sl]) < np.nanmean(cxa[sl]) else -1.0
    vh = direction * np.gradient(rolling_mean(cxh, 12)) * 25
    va = direction * np.gradient(rolling_mean(cxa, 12)) * 25
    # direction-normalised x: 0 = own goal line for home, L = goal home attacks
    hx = np.where(direction[:, None] > 0, px, L - px)
    hxh, hxa = np.where(home, hx, np.nan), np.where(~home, hx, np.nan)
    # back line = second-deepest player (skips the keeper); sorting with NaN as +inf is fast
    deep_h = np.sort(np.where(np.isnan(hxh), np.inf, hxh), axis=1)
    deep_a = np.sort(np.where(np.isnan(hxa), np.inf, L - hxa), axis=1)
    line_h = np.where(np.isfinite(deep_h[:, 1]), deep_h[:, 1], np.nan)
    line_a = np.where(np.isfinite(deep_a[:, 1]), L - deep_a[:, 1], np.nan)
    top_h, top_a = np.nanmax(hxh, axis=1), np.nanmin(hxa, axis=1)
    up_h = np.nanmean(hxh > L / 2, axis=1)          # share of home players in attacking half
    up_a = np.nanmean(hxa < L / 2, axis=1)          # share of away players in their attacking half
    X = np.stack([direction * (cxh - cxa), direction * (cxh - L / 2), vh, va,
                  sxh - sxa, syh - sya, sxh + sxa, syh + sya,
                  line_h, L - line_a, top_h, L - top_a, up_h - up_a], axis=1)
    X = np.stack([rolling_mean(c, 12) for c in X.T], axis=1)
    return np.nan_to_num(X)


def fit_logistic(X, y, l2=1.0, iters=25):
    """Plain IRLS logistic regression, deterministic, numpy only. Returns (mean, std, weights)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        w += np.linalg.solve(H, Z.T @ (y - p) - l2 * w)
    return mu, sd, w


def shape_logit(X, model):
    mu, sd, w = model
    return np.c_[np.ones(len(X)), (X - mu) / sd] @ w


def snap(labels, touch, window):
    """Move each direct H<->A boundary to the new team's first clear touch within +-window frames,
    if the old team has no clear touch after it. touch: +1 home touch, -1 away touch, 0 none."""
    out = labels.copy()
    flips = np.flatnonzero((labels[1:] != labels[:-1]) & np.isin(labels[1:], ["H", "A"])
                           & np.isin(labels[:-1], ["H", "A"])) + 1
    for t in flips:
        new = 1 if labels[t] == "H" else -1
        lo, hi = max(t - window, 0), min(t + window + 1, len(labels))
        seg = touch[lo:hi]
        last_old = np.flatnonzero(seg == -new)
        cand = np.flatnonzero(seg == new)
        if len(last_old):
            cand = cand[cand > last_old[-1]]
        if not len(cand):
            continue
        f = lo + cand[0]
        if f < t and np.all(np.isin(out[f:t], ["H", "A"])):
            out[f:t] = labels[t]
        elif f > t and np.all(np.isin(out[t:f], ["H", "A"])):
            out[t:f] = labels[t - 1]
    return out


def decode(team, dead, blocks, team_switch=None, dead_switch=None):
    labels = np.full(len(team), "X", dtype="<U1")
    for first, last in blocks:
        labels[first:last + 1] = viterbi(team[first:last + 1], dead[first:last + 1],
                                         team_switch, dead_switch)
    return labels


@functools.lru_cache(maxsize=1)
def inputs(game=GAME):
    meta, raw = require(["atd_meta", "atd_raw"], game)
    players, ball = load_atd(raw)
    return players, ball, team_of_each_track(meta)


def run(players, ball, team_of_track, pitch, blocks, starts=frozenset(), tag_vote=None,
        dead_before=None, tag_weight=TAG_WEIGHT, shape_weight=SHAPE_WEIGHT, verbose=False,
        model="default"):
    """The pipeline on plain arrays, so any match in this format can go through it.
    blocks: [(first, last)] frames to label; everything else is X. tag_vote: per-frame +1/-1/0
    phase prior or None. dead_before: bool per frame forced towards dead (before kickoff)."""
    if model == "default":
        model = load_model() if USE_MODEL else None
        if USE_MODEL and model is None:
            raise IOError("{} is missing; run `python devset.py --train` or set USE_MODEL = False"
                          .format(MODEL_PATH))
    if model is not None:
        return run_learned(players, ball, team_of_track, pitch, blocks, starts, tag_vote,
                           dead_before, tag_weight, model)
    team, dead = features(players, ball, team_of_track, pitch, starts)
    if dead_before is not None:
        # a rule, not a soft cue: before the kickoff the ball is dead, at full strength
        dead = np.where(dead_before, CLIP, dead)
    seen = ~np.isnan(ball[:, 0])
    # priors only speak where the ball is missing; with the ball tracked, the touch decides
    if tag_vote is not None:
        team = team + np.where(seen, 0.0, tag_weight * tag_vote)
    labels = decode(team, dead, blocks)
    if shape_weight:
        # second pass: learn "which team has the ball" from team shape alone, using the first
        # pass's own labels on frames where the ball is tracked (algorithmic pseudo-labels, no
        # human labels), then add it as evidence where the ball is missing.
        X = shape_features(players, team_of_track, pitch, blocks)
        train = seen & np.isin(labels, ["H", "A"])
        model = fit_logistic(X[train], (labels[train] == "H").astype(float))
        logit = np.clip(shape_logit(X, model), -CLIP, CLIP)
        if verbose:
            acc = np.mean((logit[train] > 0) == (labels[train] == "H"))
            print("shape model: trained on {} frames, agreement {:.3f}".format(train.sum(), acc))
        team = team + np.where(seen, 0.0, shape_weight * logit)
        labels = decode(team, dead, blocks)
    if SNAP:
        touch = np.where(seen & (np.abs(team) >= TOUCH), np.sign(team), 0)
        labels = snap(labels, touch, SNAP)
    return labels


def run_learned(players, ball, team_of_track, pitch, blocks, starts, tag_vote, dead_before,
                tag_weight, model):
    """Same decoder, but team evidence is the learned log-odds that home has the ball, from one
    model where the ball is tracked and one where it is missing. Dead-ball evidence is unchanged."""
    team, dead, parts = features(players, ball, team_of_track, pitch, starts, parts=True)
    seen = parts["seen"]
    F = evidence_features(team, parts, shape_features(players, team_of_track, pitch, blocks))
    logit = np.where(seen, W_SEEN * shape_logit(F, model["seen"]), W_GAP * shape_logit(F, model["gap"]))
    team = np.clip(logit, -2 * CLIP, 2 * CLIP)
    if dead_before is not None:
        # before the kickoff nobody has the ball: dead at full strength, and no team evidence. The
        # kickoff tag starts a few seconds before the kick, so carry the mask on through the
        # stoppage the tag opens, up to the kick that closes it.
        dead_before = dead_before.copy()
        ends = np.flatnonzero(dead_before[:-1] & ~dead_before[1:]) + 1
        for e in ends:
            while e < len(dead_before) and parts["stoppage"][e]:
                dead_before[e] = True
                e += 1
        dead = np.where(dead_before, CLIP, dead)
        team = np.where(dead_before, 0.0, team)
    if tag_vote is not None:
        team = team + np.where(seen, 0.0, tag_weight * tag_vote)
    labels = decode(team, dead, blocks, L_TEAM_SWITCH, L_DEAD_SWITCH)
    if L_SNAP:
        touch = np.where(seen & (np.abs(team) >= TOUCH), np.sign(team), 0)
        labels = snap(labels, touch, L_SNAP)
    return labels


def label(game=GAME, ball=None, tag_weight=TAG_WEIGHT, shape_weight=SHAPE_WEIGHT, verbose=False):
    """The competition match. `ball` overrides the tracked ball (gapcheck.py hides it)."""
    players, tracked, team_of_track = inputs(game)
    ball = tracked if ball is None else ball
    n = source_frames("tactical", game)
    blocks = [(first, last) for _, first, last in half_video_blocks(game)]
    return run(players[:n], ball[:n], team_of_track, GAMES[game]["pitch"], blocks,
               set_piece_starts(game), tag_prior(n, game), pre_kickoff(n, game),
               tag_weight, shape_weight, verbose)


def _selftest():
    # a clean hand-over with a 3-frame deflection in the middle must give exactly two possessions
    team = np.r_[np.full(100, 2.0), np.full(3, -2.0), np.full(100, 2.0), np.full(100, -2.0)]
    path = "".join(viterbi(team, np.full(len(team), -1.0)))
    assert path == "H" * 203 + "A" * 100, path
    # a stoppage between two home possessions must survive as D
    dead = np.r_[np.full(100, -1.0), np.full(200, 2.0), np.full(100, -1.0)]
    path = "".join(viterbi(np.where(dead < 0, 2.0, 0.0), dead))
    assert path == "H" * 100 + "D" * 200 + "H" * 100, path
    # ball rolls out over the touchline, vanishes, then is thrown in: one stoppage, closed by the throw
    bx = np.r_[np.linspace(50, 50, 50), np.full(50, 50.0), np.full(100, np.nan), 50 + 0.4 * np.arange(50)]
    by = np.r_[np.linspace(60, 70, 50), np.full(50, 70.0), np.full(100, np.nan), np.full(50, 67.0)]
    speed = 25 * np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan))
    dead = stoppages(bx, by, speed, set(), (105.3, 68.0))
    assert dead[45] and dead[150] and not dead[-1] and not dead[10], np.flatnonzero(dead)[[0, -1]]
    # snapping moves a late boundary back to the new team's first touch, and never past an old touch
    labels = np.array(list("HHHHHHHHHHAAAAA"))
    touch = np.array([1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert "".join(snap(labels, touch, 5)) == "HHHHHHAAAAAAAAA"
    touch[8] = 1
    assert "".join(snap(labels, touch, 5)) == "HHHHHHHHHHAAAAA"


if __name__ == "__main__":
    _selftest()
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "v3.csv")
    labels = label(verbose=True)
    path = write_submission(out, labels)
    print({k: int(np.sum(labels == k)) for k in "HAD"}, "->", path)
