"""Turn hand-labelled possession intervals into a ground-truth CSV the official scorer accepts.

Label a few minutes of the tactical video outside the public window (11185-18684), one row per
stretch, in validation/intervals.csv:

    start,end,label
    12:30.0,12:41.2,H
    12:41.2,12:47.0,D
    12:47.0,13:02.5,A

start and end are tactical-video times, mm:ss.s (or a bare frame number); end is exclusive.
Labels are H, A or D. Frames you did not label become X and are ignored by the scorer. Then:

    .venv/Scripts/python.exe tools/make_validation_gt.py
    .venv/Scripts/python.exe evaluation/run_evaluation.py --gt validation/validation_gt.csv

The rules allow small hand-labelled sets for validation. Never hand-correct a submission.
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
PUBLIC_WINDOW = (11185, 18684)
VALID_LABELS = ("H", "A", "D")


def to_frame(text):
    """'mm:ss.s', 'hh:mm:ss.s' or a bare frame number -> video frame."""
    text = text.strip()
    if ":" not in text:
        return int(text)
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return int(round(seconds * FPS))


def read_intervals(path):
    rows = []
    with open(path, newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            start, end = to_frame(row["start"]), to_frame(row["end"])
            label = row["label"].strip().upper()
            if label not in VALID_LABELS:
                raise ValueError("line {}: label {!r} is not H, A or D".format(number, label))
            if not 0 <= start < end <= N_FRAMES:
                raise ValueError("line {}: bad interval {}..{}".format(number, start, end))
            rows.append((start, end, label, number))
    return sorted(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intervals", default=os.path.join(ROOT, "validation", "intervals.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "validation", "validation_gt.csv"))
    args = parser.parse_args()

    labels = ["X"] * N_FRAMES
    previous_end = 0
    in_public = 0
    for start, end, label, number in read_intervals(args.intervals):
        if start < previous_end:
            sys.exit("line {}: overlaps the previous interval".format(number))
        in_public += start <= PUBLIC_WINDOW[1] and end > PUBLIC_WINDOW[0]
        labels[start:end] = [label] * (end - start)
        previous_end = end
    if in_public:
        print("note: {} interval(s) overlap the public window -- right for calibration, no new "
              "information for validation".format(in_public))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    graded = sum(label != "X" for label in labels)
    print("wrote {} with {} labelled frames ({:.1f} min)".format(
        os.path.relpath(args.out, ROOT), graded, graded / FPS / 60.0))
