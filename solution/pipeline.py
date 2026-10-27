"""Regenerate the submission end to end, in one command.

    .venv/Scripts/python.exe tools/run_blind.py solution/pipeline.py [--name NAME] [--out DIR]

Three possession models built on the same evidence, combined by a per-frame vote:

  1. solution/hmm_possession.py   HMM with touches and boundary snapping, temper 6
  2. solution_alt/last_touch.py   state machine, 2 opposing touches to hand the ball over
  3. solution/hmm_possession.py   the same HMM plus the self-supervised team-shape model
                                  (--shape --walk-ms 1.25 --temper 3), the best at telling
                                  the teams apart on the hand-labelled validation minutes

The vote takes the majority label per frame, with the first HMM breaking three-way ties; then
segments under 1 s are merged away and each boundary is snapped to the nearest supporting event
within 1 s, as the HMM does. Members and clean-up were chosen on the public window plus 10
hand-labelled minutes (validation/README.md), among candidates fixed in advance, and every
setting passes the full-match plausibility rule; solution/ROADMAP.md has the numbers.
Deterministic: no randomness anywhere.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "solution_alt"))
import hmm_possession as hmm  # noqa: E402
import common  # noqa: E402
import ensemble  # noqa: E402
import last_touch  # noqa: E402

# Params
HMM_ARGS = ["--temper", "6"]
SHAPE_ARGS = ["--shape", "--walk-ms", "1.25", "--temper", "3"]
CONFIRM_TOUCHES = 2
MERGE_FRAMES = common.FPS
SNAP_FRAMES = hmm.SNAP_FRAMES


def run_hmm(options):
    """States of the single HMM variant these options describe."""
    states = list(hmm.predict(hmm.build_parser().parse_args(options)).values())
    if len(states) != 1:
        raise RuntimeError("expected one HMM variant for {}".format(options))
    return states[0]


def members(extra=()):
    """[(name, states)] of the three models, the plain HMM first so it breaks ties."""
    data = common.load_match()
    return data, [("hmm", run_hmm(HMM_ARGS + list(extra))),
                  ("last_touch", last_touch.label(data, CONFIRM_TOUCHES)),
                  ("hmm_shape", run_hmm(SHAPE_ARGS + list(extra)))]


def combine(data, states):
    voted = ensemble.vote(np.stack(states))
    merged = ensemble.merge_blips(voted, data["blocks"], MERGE_FRAMES)
    return hmm.snap_boundaries(merged, data["blocks"], data["touch_team"], data["onset"],
                               SNAP_FRAMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=common.RESULTS_DIR, help="where to write")
    parser.add_argument("--name", default="ensemble", help="submission file name, no extension")
    parser.add_argument("--prefix", help="alias for --name, for tools/experiment.py")
    parser.add_argument("--members", action="store_true", help="also write each member")
    # Unrecognised options go to both HMM members, e.g. --decay-m 1 for an experiment.
    args, extra = parser.parse_known_args()

    data, models = members(extra)
    name = args.prefix or args.name
    common.write(combine(data, [states for _, states in models]), args.out, name)
    if args.members:
        for member, states in models:
            common.write(states, args.out, "{}_{}".format(name, member))
