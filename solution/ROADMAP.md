# Improvement roadmap

Worked through autonomously; the Stop hook (`.claude/hooks/keep_working.py`) keeps the session
going while any box below is unchecked and `.claude/keep_working.on` exists.

Accept a change only if it stays plausible on the full match (D 38-48% of graded frames,
possessions 11-17 s) and does not lower public macroF1 or spellF1 beyond noise. Never choose
between plausible variants by public spellF1 alone. Run each attempt with `tools/experiment.py`
and record the verdict here, one line, with the numbers.

Reference: `hmm_walk_t5` -- D 44.3%, 13.7 s, macroF1 0.796, spellF1 0.475, D F1 0.70, 624 wrong-team frames.

- [x] Touch detection: strong nearest-player evidence at ball-velocity changes, weak in between
  - ACCEPTED, now default. temper 5: D 44.7%, 13.0 s, macroF1 0.831 (+0.035), spellF1 0.542 (+0.067), D F1 0.77, wrong-team 554 (-70). Better at every plausible temper (4, 5, 6), so not a lucky setting.
- [x] Boundary snapping: move each possession change to the nearest touch / ball-out / stop event within ~1 s
  - ACCEPTED, now default at the pre-committed +/- 25 frames: macroF1 0.834 (+0.003), spellF1 0.542 (=), wrong-team 526 (-28), counts and plausibility unchanged by construction. +/- 50 was a bit better (spellF1 0.568, wrong-team 516) but choosing it would be picking by public score.
- [x] Learned local frame model: self-supervised logistic regression on local features (k nearest per team, ball speed, goal-side), balanced by team, no territory
  - REJECTED (kept as `--learned`, off). Trained on frames between same-side touches. Worse at every temper: replace t5 macroF1 0.746 / spellF1 0.503 / wrong-team 655 / D F1 0.59; blend t5 0.796 / 0.525 / 612 / 0.70; reference 0.834 / 0.542 / 526 / 0.77. Its probabilities are confident wherever the ball is seen, overriding the distance fade during flight and crowding out D.
- [x] Duration model: explicit possession-length prior (semi-Markov or a minimum-duration constraint)
  - NEUTRAL, not adopted (kept as `--chain k`, Erlang durations via k sub-states; k=1 reproduces the reference exactly). Longer chains lengthen possessions, so temper must drop to stay plausible; at matched length (chain 2 t3 13.0 s, chain 3 t2.5 14.3 s) the public result is identical to the reference in every number. No measurable benefit for the added complexity.
- [x] Dead ball near lines: ball last seen near a touchline or goal line, then lost
  - REJECTED (kept as `--dead _walkline`). D share rises to 47.4% and D F1 falls to 0.71 (reference 0.77), macroF1 0.801 vs 0.834 at t5; spellF1 and wrong-team unchanged. D was already near target, so the extra evidence only adds false stoppages.
- [x] Final: pick the default from accepted changes, regenerate, check reproducibility, update docs
  - Default = reference + touches + snap 25.

## Round 3: with 10 hand-labelled minutes (validation/README.md)

Judged on the combined set (public 17 + validation 31 = 48 possessions), among candidates fixed in
advance, each also passing full-match plausibility.

- [x] Re-score everything on the validation minutes
  - All our models beat smooth50 (spellF1 0.210, near its official full-match 0.243) by +0.1-0.15. Touch detection and the rejections of the 5-way vote and the learned model hold up. Team assignment is the weak spot (~18% of in-play frames wrong against 9% on public); the shape model has the fewest team errors (836 against ~1900).
- [x] Pick the ensemble on the combined set
  - ADOPTED C4: HMM t6 + last touch c2 + HMM with shape (walk 1.25, t3), vote, merge blips < 1 s, snap 25. Combined spellF1 0.448 [0.34-0.56], macroF1 0.818, wrong-team 1576; validation 0.393; public 0.555 / 0.822; D 43.2%, 14.4 s. Others: C0 previous submission 0.412 / 0.799 / 2130; C1 + merge 0.438 / 0.800; C2 + t6 0.448 / 0.797 / 2071; C3 shape as 4th member 0.434 / 0.802 / 1766. C4 ties C2 on spellF1 and wins the macroF1 tie-break and team errors.

## Round 4: targeting the remaining errors

`solution_alt/diagnose.py` (aggregate counts only) on the combined set: in play, the ball is
missing on only 6% of frames; wrong-team errors peak at 13.7% with the nearest player 2-5 m away
(loose balls); dead-ball frames called in play (1521, 939 with the ball visible) now outnumber the
wrong-team frames (1576).

- [x] E1 `--sticky-out` (D evidence from the ball leaving the pitch until the next touch) and E2 `--decay-m 1` (loose balls say less), fixed in advance, through the full pipeline
  - REJECTED, both kept as off-by-default options. Combined spellF1 / macroF1 / wrong-team / validation: current 0.448 / 0.818 / 1576 / 0.393; E1 0.449 / 0.810 / 1576 / 0.380 (D share 47.3%); E2 0.426 / 0.817 / 1552 / 0.355; E3 both 0.427 / 0.809 / 1552 / 0.341.

## Broadcast tracking (tried after round 2)

- [x] Fill tactical ball gaps from the broadcast ATD (`solution/broadcast_sync.py`, `--broadcast` in the HMM)
  - REJECTED, kept as an option, off by default. The broadcast tracks a median of 15 players against 21 (partial TV view), so syncing uses anchors (30 s windows placed anywhere in the match by one-sided nearest-player distance) grown by local search, with frame-level and ball-agreement checks. Orientation unambiguous (unflipped). Synced 38.7% of broadcast frames, but tactical ball coverage only rises 56.9% -> 58.1%: the broadcast loses the ball in the same moments. Public scores identical to the reference at temper 5 in every column; +6 min per run. Synthetic test (partial view, mirroring, replay, close-up): 95% mapped, all within 2 frames, no close-up frames mapped.

## Round 2: ensembles (see solution_alt/COMPARISON.md)

Reference: `ens_vote3` (HMM + last touch c2 + segment clean m4) -- D 43.9%, 13.6 s, macroF1 0.849, spellF1 0.575, D F1 0.80, wrong-team 489.

- [x] More diverse members: 5-way vote adding HMM variants that err differently (--shape, --tags setpieces); judge the change of members as one decision, not a search
  - REJECTED, keep the 3-way vote. ens5 (HMM, HMM+setpieces, HMM+shape t3 walk 1.25, last touch c2, segment clean m4): D 44.4%, 12.8 s, macroF1 0.838 (-0.011), spellF1 0.575 (=), D F1 0.77 (-0.03), wrong-team 478 (-11). The extra HMM variants mostly agree with the HMM, so they add correlated votes, not new information.
- [x] Post-vote clean-up: snap boundaries to events, and/or merge blips under 1 s
  - Snap ACCEPTED: vote3 + snap 25 -> macroF1 0.851 (+0.002), spellF1 0.575 (=), D F1 0.80, wrong-team 473 (-16), D 43.8%, 13.6 s. Merge 1 s REJECTED: no public change at all, full-match possessions 13.6 -> 14.4 s, so it does act on the full match but nothing shows it helps.
- [x] One-command pipeline: solution/pipeline.py regenerates the chosen submission end to end; check it reproduces every metric
  - DONE. `run_blind.py solution/pipeline.py` computes HMM, last touch c2 and segment clean m4 in-process, votes and snaps. Ensemble and all three members match the separate runs in every metric; the HMM CLI (refactored into build_parser/predict) still reproduces final_walk_t5.
- [x] Final round 2: choose the submission (HMM or ensemble), regenerate evaluation/results/, update COMPARISON.md and docs
  - ENSEMBLE chosen: better than the HMM on every metric, plausible, one command. `evaluation/results/ensemble.csv` valid: D 43.8%, 13.6 s, macroF1 0.851, spellF1 0.575 [0.39-0.76], D F1 0.80, wrong-team 473. HMM kept alongside as `hmm_walk_t5.csv`. `evaluation/results/hmm_walk_t5.csv` regenerated and valid; a second run reproduces every metric. D 44.7%, 13.0 s, macroF1 0.834, spellF1 0.542 [0.36-0.72], D F1 0.77, wrong-team 526, share error 0.001.
