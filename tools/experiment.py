"""Run one experiment end to end: blind run, validation, full-match plausibility, public score.

    .venv/Scripts/python.exe tools/experiment.py <name> [--compare NAME ...] -- [model args]

Runs solution/hmm_possession.py blind with its output in experiments/runs/<name>/, then for every
CSV it wrote:

  - validates it (the official checks),
  - computes the full-match plausibility numbers, which need no ground truth: the share of graded
    frames labelled D (target 38-48%, LaLiga ball-out-of-play time) and the mean possession
    length (target 11-17 s),
  - scores it on the public window with a 1000-draw bootstrap,

and appends one line per CSV to experiments/log.jsonl. --compare prints earlier runs alongside.

Accept a change only if it stays plausible and does not lower public macroF1 or spellF1 beyond
noise. Never pick between plausible variants by public spellF1 alone: 17 possessions cannot
rank them.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
PYTHON = sys.executable
RUNS_DIR = os.path.join(ROOT, "experiments", "runs")
LOG_PATH = os.path.join(ROOT, "experiments", "log.jsonl")
N_FRAMES = GAMES[GAME]["n_video_frames"]
GRADED = sum(last - first + 1 for _, first, last in half_video_blocks(GAME))
UNGRADED = N_FRAMES - GRADED
D_SHARE_RANGE = (38.0, 48.0)
POSSESSION_S_RANGE = (11.0, 17.0)


def validate(path):
    """(ok, graded D share %, full-match possessions, mean possession s) from the validator."""
    result = subprocess.run([PYTHON, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                             path], capture_output=True, text=True)
    if result.returncode != 0:
        return False, None, None, None
    dead = int(re.search(r"^\s+D\s+(\d+)", result.stdout, re.M).group(1))
    spells = re.search(r"reports (\d+) possessions of mean ([\d.]+) s", result.stdout)
    return (True, 100.0 * (dead - UNGRADED) / GRADED, int(spells.group(1)),
            float(spells.group(2)))


def score(run_dir):
    """Public-window report per submission name."""
    out_dir = os.path.join(run_dir, "score")
    subprocess.run([PYTHON, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                    "--results", run_dir, "--out", out_dir, "--bootstrap", "1000", "--no-detail"],
                   capture_output=True, text=True)
    reports = {}
    for path in glob.glob(os.path.join(out_dir, "reports", "*.json")):
        with open(path) as handle:
            reports[os.path.splitext(os.path.basename(path))[0]] = json.load(handle)
    return reports


def row(entry):
    flag = "ok " if entry["plausible"] else "OUT"
    return ("{flag} {name:<34} D {d_share:5.1f}%  poss {full_mean_s:5.1f}s | macroF1 {macro_f1:.3f}"
            "  spellF1 {spell_f1:.3f} [{spell_low:.2f}-{spell_high:.2f}]  Df1 {d_f1:.2f}"
            "  public poss {public_spells}  wrong-team {wrong_team}").format(flag=flag, **entry)


if __name__ == "__main__":
    argv = sys.argv[1:]
    model_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--script", default="solution/hmm_possession.py")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="earlier experiment names (or file names) to print alongside")
    parser.add_argument("--score-only", action="store_true",
                        help="skip the run and score the CSVs already in experiments/runs/<name>/, "
                             "e.g. after run_blind.py <script> --out experiments/runs/<name> "
                             "--prefix <name>")
    args = parser.parse_args(argv)

    run_dir = os.path.join(RUNS_DIR, args.name)
    os.makedirs(run_dir, exist_ok=True)
    if not args.score_only:
        for old in glob.glob(os.path.join(run_dir, "*.csv")):
            os.remove(old)
        blind = subprocess.run([PYTHON, os.path.join(ROOT, "tools", "run_blind.py"), args.script,
                                "--out", run_dir, "--prefix", args.name] + model_args,
                               capture_output=True, text=True, cwd=ROOT)
        print(blind.stdout.strip())
        if blind.returncode != 0:
            sys.exit(blind.returncode)

    reports = score(run_dir)
    entries = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        ok, d_share, full_spells, full_mean_s = validate(path)
        report = reports.get(name)
        if not ok or report is None:
            print("FAIL {}".format(name))
            continue
        confusion = report["confusion"]
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": args.name, "name": name,
            "args": model_args, "d_share": d_share, "full_spells": full_spells,
            "full_mean_s": full_mean_s, "macro_f1": report["macro_f1"],
            "spell_f1": report["spell_f1_mean"], "spell_low": report["spell_bootstrap_low"],
            "spell_high": report["spell_bootstrap_high"],
            "d_f1": report["per_class"]["D"]["f1"], "public_spells": report["n_pred_spells"],
            "wrong_team": confusion[0][1] + confusion[1][0],
        }
        entry["plausible"] = (D_SHARE_RANGE[0] <= d_share <= D_SHARE_RANGE[1]
                              and POSSESSION_S_RANGE[0] <= full_mean_s <= POSSESSION_S_RANGE[1])
        entries.append(entry)
        with open(LOG_PATH, "a") as log:
            log.write(json.dumps(entry) + "\n")

    if args.compare and os.path.isfile(LOG_PATH):
        latest = {}
        with open(LOG_PATH) as log:
            for line in log:
                old = json.loads(line)
                if old["experiment"] in args.compare or old["name"] in args.compare:
                    latest[old["name"]] = old
        for old in latest.values():
            print("  ref " + row(old))
    for entry in entries:
        print("  new " + row(entry))
