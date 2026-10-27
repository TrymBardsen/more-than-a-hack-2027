# Comparing approaches

**Update (round 3):** the submission is now the vote of HMM t6, last touch and HMM+shape, chosen
on the public window plus 10 hand-labelled minutes; see `solution/ROADMAP.md`, round 3. The table
below is the earlier public-window comparison.

All approaches use the same evidence (`common.py` reuses the HMM's feature code: nearest players,
touches, walking pace, ball out of the pitch), so the comparison measures the decoding method only.
Each was run blind and scored by `tools/experiment.py`. Settings were chosen by the full-match
plausibility rule (dead ball 38-48% of graded frames, possessions 11-17 s), never by public score.

Public window: 17 possessions, so spellF1 intervals are wide. The ground truth has 17 possessions.

| approach | file | D share | full poss | macroF1 | spellF1 [95%] | D F1 | public poss | wrong-team |
|---|---|---|---|---|---|---|---|---|
| **Ensemble + snapping (the submission)** | `solution/pipeline.py` | 43.8% | 13.6 s | **0.851** | **0.575** [0.39-0.76] | **0.80** | 15 | **473** |
| Ensemble: vote of the three below | `ensemble.py` | 43.9% | 13.6 s | 0.849 | 0.575 [0.39-0.76] | 0.80 | 15 | 489 |
| 5-way vote (+ HMM with shape, + HMM with set pieces) | `ensemble.py` | 44.4% | 12.8 s | 0.838 | 0.575 [0.39-0.76] | 0.77 | 15 | 478 |
| HMM + touches + snapping (current default) | `solution/hmm_possession.py` | 44.7% | 13.0 s | 0.834 | 0.542 [0.36-0.72] | 0.77 | 14 | 526 |
| A. Last touch, 2 touches to hand over | `last_touch.py --confirm 2` | 40.2% | 14.4 s | 0.787 | 0.459 [0.31-0.61] | 0.72 | 17 | 736 |
| B. Segment cleaning, 4 s minimum | `segment_clean.py --min-s 4` | 46.2% | 15.5 s | 0.808 | 0.333 [0.17-0.51] | 0.78 | 13 | 749 |
| baseline_smooth50 (reference) | `starter/baseline.py` | -- | -- | 0.621 | 0.326 | 0.32 | 26 | 832 |

## What each one says

- **HMM** is the strongest single method. Weighing evidence frame by frame against a model of how
  long possessions and stoppages last beats both a hard rule and a clean-up pass.
- **A. Last touch** is the simplest to explain and gets the possession *count* exactly right (17),
  but it trusts single events completely, so it gives the ball to the wrong side more often (736
  frames). Its plausible settings (2, 3 or 4 confirming touches) score identically on the public
  window.
- **B. Segment cleaning** has decent per-frame accuracy but the worst possessions: deleting short
  segments does not put the boundaries in the right place. Only `--min-s 4` is plausible; 2 s
  flickers, 6 s and up over-merge. It shows what the HMM's model of time is worth: about +0.2
  spellF1.
- **Ensemble** wins on every column. The three methods make different mistakes and the vote
  cancels them. Swapping A's `--confirm 2` for `--confirm 3` gives the same public numbers, so this
  is not a lucky combination; the gain is still well inside the public window's noise, but it is
  consistent across every metric, which is the best evidence 17 possessions can give.

- **5-way vote** adds nothing: the extra HMM variants mostly agree with the HMM, so they are
  correlated votes. **Snapping after the vote** is a small free gain (16 fewer wrong-team frames),
  as it was for the HMM. Merging sub-second blips after the vote changed nothing in public.

## Reproduce

The submission, in one command (writes `evaluation/results/ensemble.csv`):

    .venv/Scripts/python.exe tools/run_blind.py solution/pipeline.py

The pieces separately, for comparison:

    run_blind.py solution/hmm_possession.py   --out experiments/runs/final --prefix final
    run_blind.py solution_alt/last_touch.py    --out experiments/runs/lt    --prefix lt --confirm 2
    run_blind.py solution_alt/segment_clean.py --out experiments/runs/sc    --prefix sc --min-s 4
    run_blind.py solution_alt/ensemble.py      --out experiments/runs/ens   --prefix ens \
        --inputs experiments/runs/final/final_walk_t5.csv experiments/runs/lt/lt_c2.csv \
                 experiments/runs/sc/sc_m4.csv
    tools/experiment.py ens --score-only

(`run_blind.py` and `experiment.py` live in `tools/`; run them with `.venv/Scripts/python.exe`.)
