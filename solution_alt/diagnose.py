"""Where a submission's errors happen: counts by ball visibility and nearest-player distance.

    run_blind.py solution_alt/diagnose.py --submission <csv> --gt <csv> --out <dir> --prefix diag

Writes <out>/diagnostics.txt with aggregate counts only (no positions, no frame-level data), the
kind of summary the user approved the assistant reading.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import ensemble  # noqa: E402
from common import AWAY, DEAD, HOME  # noqa: E402

# Params
NEAR_BINS = ((0.0, 2.0), (2.0, 5.0), (5.0, np.inf))
UNGRADED = -1


def breakdown(truth, pred, data):
    near = np.minimum(data["d_home"], data["d_away"])
    seen = np.isfinite(data["ball"][:, 0])
    alive = (truth == HOME) | (truth == AWAY)
    situations = [("ball missing", alive & ~seen)]
    situations += [("ball seen, nearest {:g}-{:g} m".format(lo, hi),
                    alive & seen & (near >= lo) & (near < hi)) for lo, hi in NEAR_BINS]
    lines = ["in-play frames {}, wrong team {}, called dead {}".format(
        int(alive.sum()), int((alive & (pred != truth) & (pred != DEAD)).sum()),
        int((alive & (pred == DEAD)).sum()))]
    for name, mask in situations:
        wrong = mask & (pred != truth) & (pred != DEAD)
        lines.append("  {:<30} frames {:>6}  wrong team {:>5} ({:4.1f}%)  called dead {:>5}".format(
            name, int(mask.sum()), int(wrong.sum()), 100.0 * wrong.sum() / max(mask.sum(), 1),
            int((mask & (pred == DEAD)).sum())))
    dead = truth == DEAD
    lines.append("dead frames {}, called in play {} (ball missing {}, seen {})".format(
        int(dead.sum()), int((dead & (pred != DEAD)).sum()),
        int((dead & (pred != DEAD) & ~seen).sum()), int((dead & (pred != DEAD) & seen).sum())))
    return lines


if __name__ == "__main__":
    parser = common.parser(__doc__.splitlines()[0])
    parser.add_argument("--submission", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()
    data = common.load_match(args.walk_ms)
    gt = pd.read_csv(args.gt, dtype={"label": "object"})["label"].str.strip().str.upper()
    truth = gt.map(ensemble.CODES).fillna(UNGRADED).to_numpy(dtype=np.int8)  # X -> ungraded
    lines = breakdown(truth, ensemble.read(args.submission), data)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "diagnostics.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
