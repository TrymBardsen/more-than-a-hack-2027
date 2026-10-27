"""Shared inputs for the alternative approaches, built with the same feature code as the HMM.

Every approach here sees exactly the evidence solution/hmm_possession.py sees -- nearest players,
touches, walking pace, ball out of the pitch -- so a comparison measures the decoding method
and nothing else.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import baseline  # noqa: E402
import hmm_possession as hmm  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FPS = GAMES[GAME]["fps"]
HOME, AWAY, DEAD = hmm.HOME, hmm.AWAY, hmm.DEAD
LABELS = hmm.LABELS


def parser(description):
    """Argument parser with the --out / --prefix contract tools/experiment.py relies on."""
    out = argparse.ArgumentParser(description=description)
    out.add_argument("--out", default=RESULTS_DIR, help="where to write")
    out.add_argument("--prefix", required=True, help="file name prefix")
    out.add_argument("--walk-ms", type=float, default=hmm.WALK_MS,
                     help="median player speed below which play looks stopped")
    return out


def load_match(walk_ms=hmm.WALK_MS):
    """Everything the approaches need, as a dict of per-frame arrays plus the half blocks."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    team_of_track = baseline.team_of_each_track(meta_path)
    blocks = half_video_blocks(GAME)
    events = load_smart_tagging.load(GAME)

    ball_raw, xy, _ = hmm.parse_atd(raw_path, n_frames, len(team_of_track))
    ball = hmm.interpolate_gaps(ball_raw, blocks, hmm.MAX_INTERP_GAP)
    d_home, d_away = hmm.nearest_distances(ball, xy, team_of_track, pitch)
    _, touch_team = hmm.touch_gain(ball, d_home, d_away, pitch)
    motion = hmm.team_motion(xy, pitch)

    length, width = pitch
    seen = ~np.isnan(ball[:, 0])
    margin_x = hmm.OUT_OF_PITCH_MARGIN_M / length
    margin_y = hmm.OUT_OF_PITCH_MARGIN_M / width
    out = seen & ((ball[:, 0] < -margin_x) | (ball[:, 0] > 1 + margin_x)
                  | (ball[:, 1] < -margin_y) | (ball[:, 1] > 1 + margin_y))
    walking = np.nan_to_num(motion, nan=walk_ms) < walk_ms
    stationary = hmm.stationary_ball(ball, pitch)

    pre_kickoff = np.zeros(n_frames, dtype=bool)
    kickoff_2nd = hmm.second_half_kickoff(events)
    if kickoff_2nd is not None:
        pre_kickoff[blocks[1][1]:kickoff_2nd] = True

    return {
        "n_frames": n_frames, "pitch": pitch, "blocks": blocks, "ball": ball,
        "d_home": d_home, "d_away": d_away, "touch_team": touch_team,
        "dead_raw": out | walking | stationary, "pre_kickoff": pre_kickoff,
        "onset": hmm.dead_onsets(ball_raw, pitch), "match": (ball, xy, team_of_track),
    }


def runs(mask):
    """[(start, stop)] half-open runs of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def write(states, out_dir, name):
    path = os.path.join(out_dir, "{}.csv".format(name))
    baseline.write_submission(path, LABELS[states])
    print("wrote {}".format(name))
