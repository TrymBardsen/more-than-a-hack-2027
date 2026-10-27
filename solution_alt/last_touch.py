"""Possession as a state machine: the team of the last confirmed touch has the ball.

No probabilities and no decoding -- the rules a referee's assistant would write down:

  1. Events. A touch (the ball's velocity changes sharply with a player close) or sustained
     control (one side's player within CONTROL_M of the ball, the nearest opponent clearly
     further, for CONTROL_S) is an event for that side.
  2. Dead ball. Players at walking pace, the ball out of the pitch or standing still, held for at
     least DEAD_ENTER_S, is a stoppage for as long as it lasts.
  3. Possession. Between stoppages the ball belongs to the side of the last event, but a change
     must be confirmed: the new side needs a control event, or CONFIRM_TOUCHES touches before the
     old side touches again. A lone deflection does not hand the ball over. The change is dated
     to the new side's first event.

Two settings follow the HMM's full-match plausibility rule (dead ball ~40-45% of graded frames,
possessions ~14 s): --walk-ms for the stoppage share and --confirm for possession length.

    .venv/Scripts/python.exe tools/experiment.py <name> --script solution_alt/last_touch.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import AWAY, DEAD, FPS, HOME  # noqa: E402

# Params
CONTROL_M = 1.0
CONTROL_MARGIN_M = 1.0
CONTROL_S = 0.4
DEAD_ENTER_S = 1.5
CONFIRM_TOUCHES = 2


def events_for(data):
    """Sorted [(frame, side, is_control)] of touches and starts of sustained control."""
    d_home, d_away = data["d_home"], data["d_away"]
    near = np.minimum(d_home, d_away)
    clear = np.abs(d_away - d_home) > CONTROL_MARGIN_M
    out = []
    for side, mine, theirs in ((HOME, d_home, d_away), (AWAY, d_away, d_home)):
        control = (mine == near) & (near < CONTROL_M) & clear & np.isfinite(mine)
        for start, stop in common.runs(control):
            if stop - start >= CONTROL_S * FPS:
                out.append((start, side, True))
        for start, _ in common.runs(data["touch_team"] == side):
            out.append((start, side, False))
    return sorted(out)


def label(data, confirm_touches):
    n = data["n_frames"]
    states = np.full(n, DEAD, dtype=np.int8)
    dead = np.zeros(n, dtype=bool)
    for start, stop in common.runs(data["dead_raw"]):
        if stop - start >= DEAD_ENTER_S * FPS:
            dead[start:stop] = True
    dead |= data["pre_kickoff"]
    events = events_for(data)
    frames = np.array([e[0] for e in events])

    for _, first, last in data["blocks"]:
        for start, stop in common.runs(~dead[first:last + 1]):
            start, stop = start + first, stop + first
            low, high = np.searchsorted(frames, start), np.searchsorted(frames, stop)
            spell = events[low:high]
            if not spell:
                continue  # an alive stretch with no evidence at all stays D
            owner, since = spell[0][1], start
            pending, pending_from = 0, None
            for frame, side, is_control in spell:
                if side == owner:
                    pending, pending_from = 0, None
                    continue
                pending += 1
                pending_from = frame if pending_from is None else pending_from
                if is_control or pending >= confirm_touches:
                    states[since:pending_from] = owner
                    owner, since = side, pending_from
                    pending, pending_from = 0, None
            states[since:stop] = owner
    return states


if __name__ == "__main__":
    parser = common.parser(__doc__.splitlines()[0])
    parser.add_argument("--confirm", type=int, nargs="+", default=[CONFIRM_TOUCHES],
                        help="opposing touches needed to hand the ball over; one variant each")
    args = parser.parse_args()
    data = common.load_match(args.walk_ms)
    for confirm in args.confirm:
        common.write(label(data, confirm), args.out, "{}_c{}".format(args.prefix, confirm))
