"""Per-frame majority vote of several submissions; ties go to the first one.

    run_blind.py solution_alt/ensemble.py --out experiments/runs/<name> --prefix <name> \
        --inputs first.csv second.csv third.csv

The inputs are submissions written by the other approaches, so this reads data-derived files and
runs blind like them. A vote can create short segments where the inputs disagree in turn; it is
compared as is, without any clean-up.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

# Params
CODES = {label: code for code, label in enumerate(common.LABELS)}


def read(path):
    labels = pd.read_csv(path, dtype={"label": "object"})["label"].str.strip().str.upper()
    return labels.map(CODES).fillna(common.DEAD).to_numpy(dtype=np.int8)


def vote(stack):
    """Majority over rows of stack (inputs x frames); the first input breaks ties."""
    counts = np.stack([(stack == code).sum(axis=0) for code in range(len(common.LABELS))])
    best = counts.max(axis=0)
    winner = counts.argmax(axis=0).astype(np.int8)
    tied = (counts == best).sum(axis=0) > 1
    first_is_best = counts[stack[0], np.arange(stack.shape[1])] == best
    return np.where(tied & first_is_best, stack[0], winner).astype(np.int8)


def merge_blips(states, blocks, min_frames):
    """Fold segments shorter than min_frames into their neighbours, half by half."""
    import segment_clean
    out = states.copy()
    for _, first, last in blocks:
        out[first:last + 1] = segment_clean.clean(states[first:last + 1], min_frames)
    return out


if __name__ == "__main__":
    parser = common.parser(__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--cleanup", action="store_true",
                        help="also write variants with blips under 1 s merged and/or boundaries "
                             "snapped to events within 1 s (reads the match data for the events)")
    args = parser.parse_args()
    stack = np.stack([read(path) for path in args.inputs])
    name = "{}_vote{}".format(args.prefix, len(args.inputs))
    voted = vote(stack)
    common.write(voted, args.out, name)
    if args.cleanup:
        data = common.load_match(args.walk_ms)
        blocks = data["blocks"]
        merged = merge_blips(voted, blocks, common.FPS)
        snap = common.hmm.snap_boundaries
        common.write(merged, args.out, name + "_merge1")
        common.write(snap(voted, blocks, data["touch_team"], data["onset"], common.FPS),
                     args.out, name + "_snap25")
        common.write(snap(merged, blocks, data["touch_team"], data["onset"], common.FPS),
                     args.out, name + "_merge1_snap25")
