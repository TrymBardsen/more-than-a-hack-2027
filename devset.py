"""Labelled dev set from OTHER matches: Metrica's open sample games, with this match's noise.

    python devset.py            # score the current pipeline settings
    python devset.py --sweep    # coordinate-descent search over the pipeline's priors
    python devset.py --lomo     # leave one match out: tune on one match, score on the other
    python devset.py --train    # train the learned evidence model on both matches -> evidence_model.json
    python devset.py --tune-learned   # tune its weights and switching costs, scored across matches

Data (public, not redistributed here): https://github.com/metrica-sports/sample-data
Put Sample_Game_1_* and Sample_Game_2_* files in data/external/ (gitignored).

Why: the competition match has no labels we may tune on, so every threshold in possession.py was
a guess. These two matches have human event data (allowed: RULES.md permits labels on other
games). We turn the events into frame labels with the challenge's own possession rules, make the
clean sample tracking look like ours (this match's real ball-missing pattern, player dropouts),
and score with the official metric code.

Event -> label rules (the same ones the challenge describes):
  * a team is in possession from its PASS, RECOVERY, SET PIECE or SHOT until the other team has one
    (BALL LOST and CHALLENGE alone do not change possession: control has to be gained)
  * the ball is dead from BALL OUT, FAULT RECEIVED, or a shot that ends out or in the goal, until the
    next SET PIECE
  * frames before a period's kickoff and after its last event are X (ungraded)
"""

import functools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession as P  # noqa: E402
import possession_metrics as pm  # noqa: E402

EXT = os.path.join(ROOT, "data", "external")
PITCH = (105.0, 68.0)
GAMES = (1, 2)
TEAMS = {"Home": 0, "Away": 1}


def _tracking(path):
    a = pd.read_csv(path, skiprows=2).to_numpy(dtype=np.float64)
    return a[:, 0], a[:, 3:-2].reshape(len(a), -1, 2), a[:, -2:]


def event_labels(events, n):
    """Frame labels (H/A/D/X) from the event list, by the rules in the module docstring."""
    labels = np.full(n, "X", dtype="<U1")
    actions = []                                     # (frame, kind, team)
    for e in events.itertuples(index=False):
        start, end = int(e[4]) - 1, int(e[6]) - 1
        kind, sub, team = e[1], str(e[2]), TEAMS.get(e[0])
        if kind in ("PASS", "RECOVERY", "SHOT", "SET PIECE"):
            actions.append((start, "team", team))
        if kind in ("BALL OUT", "FAULT RECEIVED"):
            actions.append((start, "dead", None))
        if kind == "SHOT" and ("OUT" in sub or "GOAL" in sub):
            actions.append((max(end, start), "dead", None))
    for period in sorted(events["Period"].unique()):
        ev = events[events["Period"] == period]
        kick = ev[(ev["Type"] == "SET PIECE") & ev["Subtype"].astype(str).str.startswith("KICK OFF")]
        first = int(kick["Start Frame"].min()) - 1
        last = int(ev[["Start Frame", "End Frame"]].max().max()) - 1
        state = "D"
        acts = sorted(a for a in actions if first <= a[0] <= last)
        bounds = [a[0] for a in acts] + [last + 1]
        for (f, kind, team), nxt in zip(acts, bounds[1:]):
            if kind == "dead":
                state = "D"
            elif team is not None:
                state = "HA"[team]
            labels[f:nxt] = state
    return labels


def player_dropout(players, rng, mean_gone=50, mean_seen=625):
    """Each player vanishes for ~2 s every ~25 s, about the 7% missing this match's tracking has."""
    out = players.copy()
    n = len(players)
    for j in range(players.shape[1]):
        t = int(rng.exponential(mean_seen))
        while t < n:
            gone = int(rng.exponential(mean_gone)) + 1
            out[t:t + gone, j] = np.nan
            t += gone + int(rng.exponential(mean_seen))
    return out


def ball_mask_from_match(period, n):
    """Missing-ball mask copied from the competition match: our half h's pattern on sample period h."""
    _, ball, _ = P.inputs()
    from games import half_video_blocks
    ours = [np.isnan(ball[first:last + 1, 0]) for _, first, last in half_video_blocks("BAR-ATH")]
    mask = np.zeros(n, dtype=bool)
    for h, ref in enumerate(ours, start=1):
        idx = np.flatnonzero(period == h)
        k = min(len(idx), len(ref))
        mask[idx[:k]] = ref[:k]
    return mask


@functools.lru_cache(maxsize=4)
def load_game(g, harsh=False, seed=0):
    """harsh=False: the sample's own ball gaps (it drops the ball when out of play, so its gaps lean
    towards dead ball) plus player dropouts. harsh=True: this match's ball-missing pattern on top."""
    base = os.path.join(EXT, "Sample_Game_{}_".format(g))
    period, home, ball = _tracking(base + "RawTrackingData_Home_Team.csv")
    _, away, _ = _tracking(base + "RawTrackingData_Away_Team.csv")
    xy = np.concatenate([home, away], axis=1)
    team_of_track = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    labels = event_labels(pd.read_csv(base + "RawEventsData.csv"), len(xy))
    xy = player_dropout(xy, np.random.default_rng(seed + g))
    if harsh:
        ball = np.where(ball_mask_from_match(period, len(ball))[:, None], np.nan, ball)
    # speed per player in m/s from positions, smoothed over ~0.5 s like a tracker would report
    m = xy * np.array(PITCH)
    v = 25 * np.hypot(*np.moveaxis(np.diff(m, axis=0, prepend=np.nan), -1, 0))
    v = np.stack([P.rolling_mean(c, 6) for c in v.T], axis=1)
    players = np.concatenate([xy, v[..., None]], axis=2)
    ball3 = np.c_[ball, np.full(len(ball), np.nan)]
    blocks = []
    for p in np.unique(period):
        idx = np.flatnonzero((period == p) & (labels != "X"))
        blocks.append((int(idx[0]), int(idx[-1])))
    return players, ball3, team_of_track, blocks, labels


def score(games, **run_opts):
    """Mean spellF1 and macroF1 over the dev games, plus possession counts."""
    out = []
    for players, ball, team_of_track, blocks, truth in games:
        pred = P.run(players, ball, team_of_track, PITCH, blocks, tag_vote=None, **run_opts)
        gt, pr = pm.label_codes(truth), pm.label_codes(pred)
        runs = pm.valid_runs(gt)
        s = pm.spell_scores(gt, pr, runs)
        out.append((s["mean_f1"], pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pr))),
                    s["n_gt"], s["n_pred"]))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].sum(), a[:, 3].sum()


SEARCH = {
    "SLOW_PLAYERS": [1.1, 1.3, 1.5, 1.7],
    "KICK_SPEED": [3.0, 4.0, 5.0],
    "EVENT_DEAD": [0.5, 1.0, 1.5],
    "RECEPTION": [0.0, 1.0, 2.0, 4.0],
    "TEAM_SWITCH": [12, 16, 20, 25],
    "DEAD_SWITCH": [10, 15, 20],
    "PLACED_BALL": [0.3, 0.5, 0.8],
    "SHAPE_WEIGHT": [0.1, 0.2, 0.3],
}


def sweep(games, rounds=2):
    """Coordinate descent on mean spellF1: one prior at a time, keep a change only if it helps."""
    best = score(games, shape_weight=P.SHAPE_WEIGHT)[0]
    print("start  spellF1 {:.4f}".format(best))
    for r in range(rounds):
        for name, values in SEARCH.items():
            keep = getattr(P, name)
            results = []
            for v in values:
                setattr(P, name, v)
                results.append((score(games, shape_weight=P.SHAPE_WEIGHT)[0], v))
            top, v = max(results)
            setattr(P, name, v if top > best + 1e-4 else keep)
            best = max(best, top)
            print("round {} {:<15} {}  -> {} (spellF1 {:.4f})".format(
                r + 1, name, " ".join("{}:{:.3f}".format(v, s) for s, v in results),
                getattr(P, name), best))
    return {name: getattr(P, name) for name in SEARCH}


def training_rows(games):
    """Learned-evidence inputs and targets (1 = home has the ball) on every labelled live frame."""
    Fs, ys, ss = [], [], []
    for players, ball, team_of_track, blocks, truth in games:
        team, _, parts = P.features(players, ball, team_of_track, PITCH, parts=True)
        F = P.evidence_features(team, parts, P.shape_features(players, team_of_track, PITCH, blocks))
        m = np.isin(truth, ["H", "A"])
        Fs.append(F[m]); ys.append((truth[m] == "H").astype(float)); ss.append(parts["seen"][m])
    return np.concatenate(Fs), np.concatenate(ys), np.concatenate(ss)


def mirror(F):
    """The same frames with home and away swapped, column by column (see evidence_features and
    shape_features for the column order). Training on both makes the model side-neutral: it can't
    learn the sample matches' home possession share as a bias."""
    M = F.copy()
    for c in (0, 1, 2, 5, 6, 7):                    # home-positive evidence columns flip sign
        M[:, c] = -F[:, c]
    s = 8                                          # shape columns start here
    # centroid gap in the attacking direction is side-neutral (the defending team is always nearer
    # its own goal), so column s + 0 stays as it is
    M[:, s + 1] = F[:, s + 0] - F[:, s + 1]        # home centroid -> away centroid, mirrored
    M[:, s + 2], M[:, s + 3] = -F[:, s + 3], -F[:, s + 2]   # forward velocities swap
    M[:, s + 4], M[:, s + 5] = -F[:, s + 4], -F[:, s + 5]   # spread differences flip
    M[:, s + 8], M[:, s + 9] = F[:, s + 9], F[:, s + 8]     # back lines swap
    M[:, s + 10], M[:, s + 11] = F[:, s + 11], F[:, s + 10] # most advanced players swap
    M[:, s + 12] = -F[:, s + 12]                   # attacking-half occupancy difference
    return M


MIRROR = True


def train(games):
    """Two logistic regressions on the dev labels: ball tracked, and ball missing."""
    F, y, seen = training_rows(games)
    if MIRROR:
        F, y, seen = np.r_[F, mirror(F)], np.r_[y, 1 - y], np.r_[seen, seen]
    return {"seen": P.fit_logistic(F[seen], y[seen]), "gap": P.fit_logistic(F[~seen], y[~seen])}


def cross_eval(**run_opts):
    """Train the evidence model on one sample match, score the other; both directions."""
    out = []
    for a, b in ((1, 2), (2, 1)):
        model = train([load_game(a, h) for h in (False, True)])
        out.append(score([load_game(b, h) for h in (False, True)], model=model, **run_opts)[0])
    return float(np.mean(out)), out


LEARNED_SEARCH = {
    "W_SEEN": [0.5, 1.0, 2.0],
    "W_GAP": [0.1, 0.25, 0.5, 1.0],
    "L_TEAM_SWITCH": [10.0, 15.0, 25.0],
    "L_DEAD_SWITCH": [20.0, 30.0, 45.0],
    "L_SNAP": [0, 50],
}


def sweep_learned(rounds=2):
    """Coordinate descent for the learned path, every candidate scored on the match the model was
    NOT trained on (cross_eval), so the choice can't reward memorising a match."""
    best = cross_eval()[0]
    print("start  cross-match spellF1 {:.4f}".format(best))
    for r in range(rounds):
        for name, values in LEARNED_SEARCH.items():
            keep = getattr(P, name)
            results = []
            for v in values:
                setattr(P, name, v)
                results.append((cross_eval()[0], v))
            top, v = max(results)
            setattr(P, name, v if top > best + 1e-4 else keep)
            best = max(best, top)
            print("round {} {:<12} {}  -> {} ({:.4f})".format(
                r + 1, name, " ".join("{}:{:.3f}".format(v, s) for s, v in results),
                getattr(P, name), best), flush=True)
    return {name: getattr(P, name) for name in LEARNED_SEARCH}


def save_model(model, path=None):
    with open(path or P.MODEL_PATH, "w") as f:
        json.dump({k: {"mu": m[0].tolist(), "sd": m[1].tolist(), "w": m[2].tolist()}
                   for k, m in model.items()}, f, indent=1)


# the hand-set priors before any dev-set tuning, to measure what tuning buys on an unseen match
START = {"SLOW_PLAYERS": 1.0, "KICK_SPEED": 6.0, "EVENT_DEAD": 2.0, "RECEPTION": 0.0,
         "TEAM_SWITCH": 15.0, "DEAD_SWITCH": 20.0, "PLACED_BALL": 0.5, "SHAPE_WEIGHT": 0.2}


def lomo():
    """Leave one match out: tune from the hand-set priors on one match, score on the other.
    If tuning on one match helps the match it never saw, the settings generalise."""
    current = {name: getattr(P, name) for name in SEARCH}
    rows = []
    for tune_g, test_g in ((1, 2), (2, 1)):
        tune = [load_game(tune_g, h) for h in (False, True)]
        test = [load_game(test_g, h) for h in (False, True)]
        for name, v in START.items():
            setattr(P, name, v)
        base = score(test, shape_weight=P.SHAPE_WEIGHT)[0]
        tuned = sweep(tune)
        held = score(test, shape_weight=P.SHAPE_WEIGHT)[0]
        for name, v in current.items():
            setattr(P, name, v)
        both = score(test, shape_weight=P.SHAPE_WEIGHT)[0]
        rows.append((tune_g, test_g, base, held, both, tuned))
    for name, v in current.items():
        setattr(P, name, v)
    print("\nleave one match out (spellF1 on the held-out match)")
    for tune_g, test_g, base, held, both, tuned in rows:
        print("  tune on {} -> test on {}: hand-set {:.4f}  tuned on other match {:.4f}  "
              "submitted (tuned on both) {:.4f}".format(tune_g, test_g, base, held, both))
        print("    settings found: {}".format(tuned))


def dev_games():
    """Both matches under both noise conditions: four scored halves-pairs."""
    return [load_game(g, harsh) for harsh in (False, True) for g in GAMES]


if __name__ == "__main__":
    if "--tune-learned" in sys.argv:
        print(sweep_learned())
        sys.exit()
    if "--train" in sys.argv:
        save_model(train([load_game(g, h) for g in GAMES for h in (False, True)]))
        print("wrote", P.MODEL_PATH)
        sys.exit()
    if "--lomo" in sys.argv:
        lomo()
        sys.exit()
    games = dev_games()
    for g, (_, ball, _, blocks, truth) in zip(GAMES * 2, games):
        graded = truth != "X"
        print("game {}: {} graded frames, D {:.0%}, ball missing {:.0%}".format(
            g, graded.sum(), np.mean(truth[graded] == "D"), np.isnan(ball[graded, 0]).mean()))
    if "--sweep" in sys.argv:
        print(sweep(games))
    s, m, ng, npred = score(games, shape_weight=P.SHAPE_WEIGHT)
    print("dev spellF1 {:.4f}  macroF1 {:.4f}  possessions gt {} pred {}".format(s, m, ng, npred))
