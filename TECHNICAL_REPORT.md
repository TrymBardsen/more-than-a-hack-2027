# Frame-level possession from automatic tracking

More than a Hack 2027 · FC Barcelona v Athletic Club · technical document

## Summary

We label each of the 199,319 tactical-video frames as `H`, `A`, `D`, or `X` (ungraded frames only).
The pipeline:
1. Turns the automatic tracking into per-frame evidence for "which team has the ball" and "is the ball dead".
2. Adds two weak priors for frames where the ball is missing.
3. Decodes the whole half with a three-state Viterbi path that charges a fixed cost for every change of state.

The code is about 300 lines of numpy and pandas. It runs in about 4 seconds on a laptop and regenerates the submission byte for byte.

No human labels were used anywhere, and no constant was fitted to the public 5 minutes.

## Why a state model

The ranking metric scores possessions as objects, matched by temporal IoU. A per-frame classifier
that is right 95% of the time can still split one long possession into many, and every split costs
a whole match. The published baselines show this: the same nearest-player rule scores spellF1 0.05
raw and 0.24 with a ±2 s majority vote.

A vote over a centred window also delays every boundary by up to half the window, and being 1 s late
already drops spellF1 to about 0.65.

A Viterbi path over {H, A, D} with a switching cost fixes both:
- It changes state only when the evidence piled up since the last change pays for the switch.
- The boundary lands at the frame where the evidence flips, not at the centre of a window.

## Assets

| asset | used | how |
|---|---|---|
| tactical ATD (players + ball) | yes | the main input; raw text parsed with pandas |
| ATD metadata XML | yes | which track slot belongs to which team (first team = home) |
| Smart Tagging | yes | set-piece starts open stoppages; "Kick Off (Start)" ends the pre-kickoff dead ball; phase team tags are a weak prior while the ball is missing |
| homography | no | the tracking is already in pitch coordinates; it is only needed for our own video detection |
| tactical video | no (watched only to understand the data) | the ball is about 5 px in a wide shot; a detector run would have cost hours for a small share of the gaps |
| broadcast video, ATD, homography | no | not synchronised to the tactical timeline, with cuts and replays; its ball is missing about half the time as well |

What the raw ATD does and does not contain shaped the design:
- **No owning-team or ball-dead field, and no ball height.** The ball's third value is mostly missing, so we compute ball speed from positions.
- **Track slots aren't people.** Every feature is team-level: nearest player per team, centroid, spread.
- **The ball is missing on 41% of graded frames**, mostly in long gaps: 98% of missing frames sit in gaps longer than 5 s. Interpolation can't fill these. We hold the current team through a gap and only switch on prior evidence.

## Evidence

All distances are in metres, using the pitch size from `games.py` (105.3 × 68.0). Every constant is in `possession.py` under `# --- priors`, with a one-line reason.

**Team evidence**, from frames where the ball is tracked:
- For each team, take the distance from its nearest player to the ball. The evidence is the difference, `d_away - d_home`. A positive value means home is closer.
- It is weighted by `exp(-nearest / 2 m)`, so a ball near a player counts and a loose ball barely counts.
- It is weighted down by a factor of 5 when the ball is moving faster than 10 m/s. A pass in flight past an opponent is not a change of control.
- It is clipped to ±3 per frame, so no single frame decides.
- Ball detections that jump more than 2 m in one frame (50 m/s) are dropped as tracker errors.

**Dead-ball evidence:**
- **Stoppages.** A small state machine treats a stoppage as an interval.
  - It opens when the ball crosses a line (seen inside the pitch within the last second, then more than 1 m beyond a line), when the ball is placed (averaging under 0.5 m/s over 1.5 s), or when a Smart Tagging set piece starts.
  - It closes at the next kick on the pitch (ball faster than 6 m/s), at least 1 s after opening. If no restart is seen, it gives up after 2 minutes.
  - Inside a stoppage every frame gets +2 dead evidence.
  - Thresholds were read off unlabeled whole-match distributions. A placed ball covers 2.2% of graded frames, consistent with set-piece placements.
- **Before kickoff.** From the first graded frame of each half until that half's tagged "Kick Off (Start)", every frame gets +2 dead evidence. A half is dead until it kicks off. The kickoff frame comes from the tagging, not from a constant.
- **Slow players.** `2 × (1.0 m/s − mean player speed over ±2 s)`, clipped to ±3. 1.0 m/s is about the 20th percentile of group speed over the match: players walk at stoppages and run in play.
- **Ball out.** A ball tracked more than 1 m beyond a line gets +3 on that frame.

**Priors while the ball is missing.** These apply only where the ball is not tracked. With the ball visible, the touch decides alone.
- **Smart Tagging phase team (0.2 per frame).**
  - Attacking phases name the team on the ball: Build Up, Progression, Creating Chances, Attacking Transition, Ball in Final Third, Ball in the Box, Players in the Box, Long Ball.
  - Defending phases name the other team: the three Defending thirds and Defensive Transition.
  - The prior is the sign of the vote over the tags that cover the frame.
- **Team-shape model (0.2 × logit per frame).**
  - Thirteen features, from the home side's point of view, with attacking direction found per half from the team centroids:
    - centroid gap between the teams
    - home centroid position
    - forward velocity of each team
    - differences and sums of team spread in x and y
    - each team's back-line height (10th/90th percentile of x) and its most advanced player
    - the difference in the share of each team's players in its attacking half
  - The model is a logistic regression (deterministic IRLS, L2 = 1), trained on the first pass's own H/A labels at frames where the ball is tracked. It agrees with those pseudo-labels on 79% of training frames without seeing the ball.
  - In the second pass it adds evidence only where the ball is missing.

## Decoding: the possession rules

The Viterbi path runs over each half separately.
- **Emission scores:** H gets `+team/2`, A gets `−team/2`, D gets the dead evidence.
- **Transition costs:** H↔A costs 15, and entering or leaving D costs 20.

The rules this encodes:
- **A change of possession needs sustained control.** About half a second of clear evidence that the other team is on the ball, or about one second of moderate evidence. A deflection, a touch without control, or a pass that brushes an opponent doesn't pay the switching cost, so possession stays with the last team. This is the challenge's "loose or contested ball stays with the last team in possession".
- **Loose and contested balls** carry little evidence (the ball is far from everyone or moving fast), so they stay with the team that last had control.
- **Restarts.** A stoppage ends at the kick that restarts play. The team that restarts is decided by who is on the ball afterwards, not by the tag. Throw-in tags have no team.
- **Dead ball has no owner.** `D` is a state of its own, never a possession, as the ground truth defines it.
- **Abstention.** We write `X` only on ungraded frames: before the match, halftime, after the final whistle. Inside the halves we always commit to a label, because on spellF1 an `X` repeats the previous label anyway, and at a hand-over that is the expensive case.

## Training data

None from humans. The only learned component is the team-shape logistic regression. It is trained on pseudo-labels that the first pass of this pipeline produces from the tracking, which the rules allow ("pseudo-labels produced by an algorithm, provided no person decided any of them"). No labels from the public 5 minutes, or from any other source, enter training or parameter choice.

## How we chose between versions without labels

The public window has 17 possessions. The organisers show that ranking variants on it picks the wrong one, so no constant, threshold or weight was fitted to it, and it took no part in the gap test below.

We did score each version against it as we built them, and that shaped where we spent effort:
- **v1** had only nearest-player team evidence and a slow-players dead-ball signal. On the public window it found about 6% of dead-ball frames. That told us the dead-ball side was the weak part, and v2 added the stoppage detector.
- **v2's thresholds** came from football rules and the unlabeled whole-match distributions, not from the public window: placed ball under 0.5 m/s, kick over 6 m/s, line margin 1 m.
- **The pre-kickoff rule** came from the published fact that the second half is graded from frame 110,808 but kicks off at 113,742 (FAQ.md). It is implemented from the Smart Tagging kickoff tag, not from those frame numbers.

Beyond that the window served as a correctness check: frames aligned, file well-formed, labels meaning what we think.

To compare ways of bridging ball gaps we built a label-free test, `gapcheck.py`:
1. Run the pipeline with the tracked ball and keep its answer as the reference.
2. Hide the ball in random 5–20 s windows where it *was* tracked. This hides 8.7% of the video, imitating the real gaps.
3. Rerun each variant and splice its answer inside the hidden windows into the reference.
4. Score with the official `spellF1`, plus plain team agreement inside the hidden windows.

Three random masks are averaged. Results:

| variant (prior weights) | spellF1 vs reference | team agreement in hidden windows |
|---|---|---|
| no priors (hold last team) | 0.9198 | 0.648 |
| shape 0.1 | 0.9224 | 0.686 |
| tags 0.1 + shape 0.1 | 0.9238 | 0.687 |
| tags 0.1 + shape 0.2 | 0.9225 | 0.691 |
| **tags 0.2 + shape 0.2 (submitted)** | **0.9246** | **0.693** |

The spellF1 differences are small. The priors mostly matter for team agreement during gaps, where they recover about 4.5 points over holding the last team.

This test measures consistency, not correctness. The reference is the pipeline's own answer with the ball tracked, and that answer has errors of its own. A variant wins by reproducing those labels when the ball is hidden. So the test can rank ways of bridging gaps, but it says nothing about how close the reference is to the truth.

The test also chose the shape features. With eight features (centroids, velocities, spreads) the best variant reached 0.687 agreement. Adding back-line height, most advanced player and attacking-half occupancy raised the shape model's own agreement from 77% to 79% and the gap-test best to 0.693.

An earlier version of this test also exposed a design flaw. A tag prior applied everywhere could force a switch on its own after about 3 s of a phase, overriding the ball. That's why both priors now act only where the ball is missing.

## Results

**Whole match** (from `validate_submission.py`):
- 322 possessions with a mean of 13.5 s
- H 66,226 frames, A 42,703, D 49,928 (31% of graded frames are dead ball)
- home share of live play 61%

**Public 5 minutes**, as a correctness check only:

| | spellF1 | macroF1 | possessions (truth 17) |
|---|---|---|---|
| this submission | 0.471 | 0.674 | 17 |
| baseline_smooth50 (published floor) | 0.326 | 0.621 | 26 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |

Where hand-overs are matched, our median boundary error is −1.5 frames. A one-frame offset is free on spellF1.

## Limitations, and what we would do with more time

- **Short possessions and hand-over timing.** On the public window only 4 of the 20 changes of state land within 1 s of the truth. The possession count matches (17 against 17), but several short possessions are merged or shifted, and the score comes mostly from long possessions. The fixed switching cost is the main cause (see below).
- **Dead ball is still the weakest class.** Many short stoppages, such as fouls where the ball stays on the pitch and play restarts quickly, produce no line crossing, no placed ball and no tag. A trained stoppage detector would help: one using referee-style cues, such as all players decelerating together or a wall forming.
- **Ball gaps.** During long gaps we recover about 69% of team frames in the hidden-ball test. Better options:
  - a richer shape model, e.g. per-player pressure features or a small sequence model trained on the same pseudo-labels
  - filling gaps with our own ball detector run on the tactical video
- **Fixed switching costs.** One cost for every hand-over. A cost that depends on context (lower after a fast ball arrives at a player, higher in a crowded box) would place short possessions better.
- **Thresholds are priors.** They were chosen from football knowledge and unlabeled distributions, not fitted. That keeps the method honest and generic, and it also means none of them is tuned. Labelled tracking from other matches would let us fit them properly.

## Reproduction

The match data goes in `data/` exactly as delivered (see `DATA.md`). It is gitignored and not part of this repository.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
.venv/bin/python possession.py <team_name>.csv      # about 4 s; prints label counts
.venv/bin/python evaluation/validate_submission.py <team_name>.csv
.venv/bin/python gapcheck.py                        # optional: the label-free gap test, about 30 s
```

`possession.py` runs its own self-tests first:
- a 3-frame deflection must not split a possession
- a stoppage between two home possessions must survive
- a throw-in must open a stoppage and close it

The output is deterministic, and two runs give the same file (SHA-1 `084e3536dbb39072668ffb5481a6957c5e4e2a50`).

Files:
- `possession.py`: the whole pipeline
- `gapcheck.py`: the label-free test
- `requirements-solution.txt`: pinned versions
- `starter/baseline.py` and `starter/load_smart_tagging.py` from the organisers' starter kit, reused for the team-slot mapping, the CSV writer and the tag parser
