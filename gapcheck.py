"""Label-free check of how well the pipeline bridges ball gaps. No ground truth involved.

    python gapcheck.py

The reference is the pipeline's own answer with the tracked ball. Then the ball is hidden in random
5-20 s windows where it *was* tracked, which mimics the real gaps, and each variant's answer inside
those windows is spliced into the reference and scored with the official spellF1 (plus plain
team agreement inside the windows). The priors only act where the ball is missing, so with the
ball tracked every variant gives the same reference. A variant that recovers the reference better under ball loss bridges real
gaps better. Three seeds are averaged so a lucky mask can't decide.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession as P  # noqa: E402
import possession_metrics as pm  # noqa: E402


def hide_ball(ball, rng, share=0.3):
    """Hide the ball in random 5-20 s windows inside tracked stretches of 15 s or more."""
    seen = ~np.isnan(ball[:, 0])
    edges = np.flatnonzero(np.diff(np.r_[0, seen.astype(int), 0]))
    out = ball.copy()
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 375 and rng.random() < share * 2:
            n = int(rng.integers(125, 500))
            start = int(rng.integers(a, max(a + 1, b - n)))
            out[start:start + n] = np.nan
    return out


VARIANTS = {
    "v2 (no priors)": dict(tag_weight=0.0, shape_weight=0.0),
    "shape 0.1": dict(tag_weight=0.0, shape_weight=0.1),
    "tags 0.1 + shape 0.1": dict(tag_weight=0.1, shape_weight=0.1),
    "tags 0.1 + shape 0.2": dict(tag_weight=0.1, shape_weight=0.2),
    "tags 0.2 + shape 0.2": dict(tag_weight=0.2, shape_weight=0.2),
}

if __name__ == "__main__":
    _, ball, _ = P.inputs()
    ref = pm.label_codes(P.label(tag_weight=0.0, shape_weight=0.0))
    runs = pm.valid_runs(ref)
    masks = [hide_ball(ball, np.random.default_rng(seed)) for seed in range(3)]
    hidden = np.mean([np.isnan(m[:, 0]).mean() - np.isnan(ball[:, 0]).mean() for m in masks])
    print("extra frames hidden: {:.1%} of the video".format(hidden))
    for name, opts in VARIANTS.items():
        f1, acc = [], []
        for m in masks:
            hid = np.isnan(m[:, 0]) & ~np.isnan(ball[:, 0])
            pred = pm.label_codes(P.label(ball=m, **opts))
            # only the hidden windows differ from the reference, so only they are judged
            f1.append(pm.spell_scores(ref, np.where(hid, pred, ref), runs)["mean_f1"])
            live = hid & (ref <= 1)
            acc.append(np.mean(pred[live] == ref[live]))
        print("{:<24} spellF1 {:.4f}   team agreement in hidden windows {:.4f}".format(
            name, np.mean(f1), np.mean(acc)))
