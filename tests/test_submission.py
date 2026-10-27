"""Checks on submission files that need no match data: format validation and scoring sanity.

    python -m pytest tests
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, ROOT)

import possession_metrics as pm  # noqa: E402  (needs the path above)
from games import GAME, GAMES  # noqa: E402

N_ROWS = GAMES[GAME]["n_video_frames"]
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def write_csv(path, labels, **to_csv_kwargs):
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(
        path, index=to_csv_kwargs.pop("index", False), **to_csv_kwargs)
    return str(path)


def validate(path):
    return subprocess.run([sys.executable, VALIDATOR, path], capture_output=True, text=True)


def test_row_count_matches_tactical_video():
    assert N_ROWS == 199319


def test_valid_submission_passes(tmp_path):
    labels = np.array(["H", "A", "D", "X"])[np.arange(N_ROWS) % 4]
    result = validate(write_csv(tmp_path / "ok.csv", labels))
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("name, make", [
    ("short", lambda: (["H"] * (N_ROWS - 1), {})),
    ("bad_label", lambda: (["H"] * (N_ROWS - 1) + ["Q"], {})),
    ("blank", lambda: (["H"] * (N_ROWS - 1) + [None], {})),
])
def test_invalid_submission_fails(tmp_path, name, make):
    labels, kwargs = make()
    result = validate(write_csv(tmp_path / (name + ".csv"), labels, **kwargs))
    assert result.returncode == 1, result.stdout


def test_public_gt_scores_perfectly_against_itself():
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    scores = pm.evaluate(gt, gt.copy(), bootstrap_n=0)
    assert scores["spell_f1_mean"] == pytest.approx(1.0)
    assert scores["macro_f1"] == pytest.approx(1.0)
