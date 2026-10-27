"""Possession by segment cleaning: label every frame on its own, then merge away short segments.

No model of time at all, just a clean-up pass -- which asks whether the HMM's machinery earns its
keep:

  1. Frame label. D where the dead-ball signals fire (walking pace, ball out, ball standing
     still); otherwise the side of the nearest player when the ball is within NEAR_M of someone;
     otherwise unknown, carried forward from the last known label in the half.
  2. Clean-up. Repeatedly take the shortest segment; if it is shorter than --min-s, merge it into
     its neighbours: into both when they agree (the blip disappears), else into the longer one.

--min-s sets possession length and follows the full-match plausibility rule (~14 s), like the
HMM's temper; --walk-ms sets the dead-ball share.

    run_blind.py solution_alt/segment_clean.py --out experiments/runs/<name> --prefix <name>
"""

import heapq
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import AWAY, DEAD, FPS, HOME  # noqa: E402

# Params
NEAR_M = 3.0
MIN_SEGMENT_S = 2.0
UNKNOWN = -1


def frame_labels(data):
    near = np.minimum(data["d_home"], data["d_away"])
    labels = np.where(near < NEAR_M, np.where(data["d_home"] < data["d_away"], HOME, AWAY),
                      UNKNOWN).astype(np.int8)
    labels[data["dead_raw"] | data["pre_kickoff"]] = DEAD
    return labels


def carry_forward(labels):
    """Fill UNKNOWN with the last known label; leading unknowns become D."""
    index = np.where(labels != UNKNOWN, np.arange(len(labels)), -1)
    index = np.maximum.accumulate(index)
    out = np.where(index >= 0, labels[np.maximum(index, 0)], DEAD)
    return out.astype(np.int8)


def clean(labels, min_frames):
    """Merge segments shorter than min_frames, shortest first. Linked list plus a heap."""
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], breaks)).tolist()
    stops = np.concatenate((breaks, [len(labels)])).tolist()
    value = [int(labels[s]) for s in starts]
    length = [b - a for a, b in zip(starts, stops)]
    n = len(starts)
    prev, nxt = list(range(-1, n - 1)), list(range(1, n + 1))
    nxt[-1] = -1
    alive = [True] * n
    heap = [(length[i], i) for i in range(n)]
    heapq.heapify(heap)

    def absorb(keep, gone):
        """Fold segment `gone` into its neighbour `keep`, unlinking it."""
        length[keep] += length[gone]
        starts[keep] = min(starts[keep], starts[gone])
        alive[gone] = False
        left, right = prev[gone], nxt[gone]
        if left >= 0:
            nxt[left] = right
        if right >= 0:
            prev[right] = left

    while heap:
        size, i = heapq.heappop(heap)
        if not alive[i] or size != length[i]:
            continue
        if size >= min_frames:
            break
        left, right = prev[i], nxt[i]
        if left < 0 and right < 0:
            break
        if left >= 0 and right >= 0 and value[left] == value[right]:
            absorb(left, i)
            absorb(left, right)
            target = left
        elif right < 0 or (left >= 0 and length[left] >= length[right]):
            absorb(left, i)
            target = left
        else:
            absorb(right, i)
            target = right
        heapq.heappush(heap, (length[target], target))

    out = np.empty_like(labels)
    i = next(k for k in range(n) if alive[k] and prev[k] < 0)
    position = 0
    while i >= 0:
        out[position:position + length[i]] = value[i]
        position += length[i]
        i = nxt[i]
    return out


def label(data, min_s):
    states = np.full(data["n_frames"], DEAD, dtype=np.int8)
    raw = frame_labels(data)
    for _, first, last in data["blocks"]:
        block = carry_forward(raw[first:last + 1])
        states[first:last + 1] = clean(block, int(min_s * FPS))
    states[data["pre_kickoff"]] = DEAD
    return states


if __name__ == "__main__":
    parser = common.parser(__doc__.splitlines()[0])
    parser.add_argument("--min-s", type=float, nargs="+", default=[MIN_SEGMENT_S],
                        help="shortest segment kept, in seconds; one variant per value")
    args = parser.parse_args()
    data = common.load_match(args.walk_ms)
    for min_s in args.min_s:
        common.write(label(data, min_s), args.out, "{}_m{:g}".format(args.prefix, min_s))
