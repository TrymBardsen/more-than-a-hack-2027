"""Run a script that reads the match data without showing its output.

    .venv/Scripts/python.exe tools/run_blind.py starter/baseline.py --window 50

The competition forbids passing the data, or extracts of it, to a hosted LLM, and anything a script
prints can be an extract. So this runs the script with the same interpreter, sends all of its
stdout and stderr to a log file under .blind_logs/ (for you to read, not the assistant), and
reports only what carries no data: the exit code, the run time, which files under
evaluation/results/ changed, and on a crash the exception type and the file:line frames in this
repo's own code -- never the exception message, which can quote values.
"""

import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, ".blind_logs")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

FRAME = re.compile(r'^\s*File "(.+?)", line (\d+), in (\S+)')
EXCEPTION = re.compile(r"^([A-Za-z_][\w.]*)(:|$)")


def snapshot(directory):
    if not os.path.isdir(directory):
        return {}
    return {name: os.path.getmtime(os.path.join(directory, name)) for name in os.listdir(directory)}


def own_frames(output):
    """file:line frames of the last traceback that point into this repo, outside the venv."""
    lines = output.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Traceback (most recent call")]
    if not starts:
        return [], None
    frames, exception = [], None
    for line in lines[starts[-1] + 1:]:
        match = FRAME.match(line)
        if match:
            path = os.path.abspath(match.group(1))
            inside = path.startswith(ROOT) and not path.startswith(os.path.join(ROOT, ".venv"))
            if inside:
                frames.append("{}:{} in {}".format(
                    os.path.relpath(path, ROOT), match.group(2), match.group(3)))
        elif not line.startswith(" "):
            found = EXCEPTION.match(line)
            if found:
                exception = found.group(1)
    return frames, exception


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: run_blind.py <script.py> [args...]")
        sys.exit(2)

    os.makedirs(LOG_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    log_path = os.path.join(LOG_DIR, "{}_{}.log".format(time.strftime("%Y%m%d-%H%M%S"), name))

    before = snapshot(RESULTS_DIR)
    started = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        code = subprocess.call([sys.executable] + sys.argv[1:], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    after = snapshot(RESULTS_DIR)

    print("exit code {} after {:.1f} s".format(code, elapsed))
    print("full output (for the user only): {}".format(os.path.relpath(log_path, ROOT)))
    changed = sorted(n for n in after if before.get(n) != after[n])
    print("changed in evaluation/results/: {}".format(", ".join(changed) if changed else "nothing"))

    if code != 0:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            frames, exception = own_frames(log.read())
        print("exception type: {}".format(exception or "unknown (no Python traceback)"))
        for frame in frames:
            print("  at {}".format(frame))
    sys.exit(code)
