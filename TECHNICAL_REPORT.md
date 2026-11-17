# Frame-level possession from automatic tracking

More than a Hack 2027 · FC Barcelona v Athletic Club · technical document

## Summary

We label each of the 199,319 tactical-video frames as `H`, `A`, `D`, or `X` (ungraded frames only).
The pipeline:
1. Turns the automatic tracking into per-frame evidence for "which team has the ball" and "is the ball dead".
2. Adds two weak priors for frames where the ball is missing.
3. Decodes each half with a three-state Viterbi path that charges a fixed cost for every change of state.

Its parameters were tuned on a labelled dev set built from **two other matches**: Metrica's open sample games, with possession labels derived from their event data. We made those matches look like ours by copying in this match's ball-missing pattern and adding player dropouts. RULES.md allows labels on other games.

No human labels on the competition match were used anywhere, and nothing was fitted to its public 5 minutes.

The code is about 600 lines of numpy and pandas across three files. Producing the submission takes about 4 seconds and is byte-for-byte reproducible.

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
| **Metrica open sample data, games 1 and 2** (other matches) | yes, for tuning only | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data): tracking plus event data, used to build the labelled dev set; see below |
| Metrica open sample game 3 | no | a different event format (EPTS with CARRY events); dropped for time |

What the raw ATD does and does not contain shaped the design:
- **No owning-team or ball-dead field, and no ball height.** The ball's third value is mostly missing, so we compute ball speed from positions.
- **Track slots aren't people.** Every feature is team-level: nearest player per team, centroid, spread.
- **The ball is missing on 41% of graded frames**, mostly in long gaps: 98% of missing frames sit in gaps longer than 5 s. Interpolation can't fill these. We hold the current team through a gap and only switch on prior evidence.

## Evidence

All distances are in metres, using the pitch size from `games.py` (105.3 × 68.0). Every constant is in `possession.py` under `# --- priors`, with a one-line reason. Values marked *(tuned)* were chosen on the dev set.

**Team evidence**, from frames where the ball is tracked:
- For each team, take the distance from its nearest player to the ball. The evidence is the difference, `d_away - d_home`. A positive value means home is closer.
- It is weighted by `exp(-nearest / 2 m)`, so a ball near a player counts and a loose ball barely counts.
- It is weighted down by a factor of 5 when the ball is moving faster than 10 m/s. A pass in flight past an opponent is not a change of control.
- It is clipped to ±3 per frame, so no single frame decides.
- **Reception** *(tuned weight 4)*: a ball arriving faster than 6 m/s that leaves slower than 3 m/s within about 0.3 s, within 1.5 m of a player, is a controlled touch. That player's team gets +4 per frame on those frames.
- Ball detections that jump more than 2 m in one frame (50 m/s) are dropped as tracker errors.

**Dead-ball evidence:**
- **Stoppages.** A small state machine treats a stoppage as an interval.
  - It opens when one of these happens:
    - the ball crosses a line (seen inside the pitch within the last second, then more than 1 m beyond a line)
    - the ball is placed: it averages under 0.3 m/s over 1.5 s *(tuned)*
    - a Smart Tagging set piece starts
  - It closes at the next kick on the pitch (ball faster than 4 m/s *(tuned)*), at least 1 s after opening. If no restart is seen, it gives up after 2 minutes.
  - Inside a stoppage every frame gets +1 dead evidence *(tuned)*.
- **Before kickoff.** From the first graded frame of each half until that half's tagged "Kick Off (Start)", every frame gets the maximum dead evidence (+3). This is a rule (a half is dead until it kicks off), not a soft cue, so it isn't tuned. The kickoff frame comes from the tagging, not from a constant.
- **Slow players.** `2 × (1.5 m/s − mean player speed over ±2 s)`, clipped to ±3 *(tuned)*. The dev set's player speeds, computed from positions, have almost the same distribution as this match's tracker speeds: the 20th percentile is 0.97–1.03 m/s there and 0.95 m/s here. So a threshold in m/s carries over.
- **Ball out.** A ball tracked more than 1 m beyond a line gets +3 on that frame.

**Priors while the ball is missing.** These apply only where the ball is not tracked. With the ball visible, the touch decides alone.
- **Smart Tagging phase team (0.2 per frame).**
  - Attacking phases name the team on the ball: Build Up, Progression, Creating Chances, Attacking Transition, Ball in Final Third, Ball in the Box, Players in the Box, Long Ball.
  - Defending phases name the other team: the three Defending thirds and Defensive Transition.
  - The prior is the sign of the vote over the tags that cover the frame.
  - The sample matches have no Smart Tagging, so this weight was chosen with the label-free gap test (below), not the dev set.
- **Team-shape model (0.2 × logit per frame)** *(tuned)*.
  - Thirteen features, from the home side's point of view, with attacking direction found per half from the team centroids:
    - centroid gap between the teams
    - home centroid position
    - forward velocity of each team
    - differences and sums of team spread in x and y
    - each team's back line (its second-deepest player, which skips the keeper) and its most advanced player
    - the difference in the share of each team's players in its attacking half
  - The model is a logistic regression (deterministic IRLS, L2 = 1), trained on the first pass's own H/A labels at frames where the ball is tracked. It agrees with those pseudo-labels on 83% of training frames without seeing the ball.
  - In the second pass it adds evidence only where the ball is missing.
  - On the dev set, turning it off costs about 0.04 spellF1.

## Decoding: the possession rules

The Viterbi path runs over each half separately.
- **Emission scores:** H gets `+team/2`, A gets `−team/2`, D gets the dead evidence.
- **Transition costs** *(tuned)*: H↔A costs 25, and entering or leaving D costs 20.

The rules this encodes:
- **A change of possession needs sustained control.** About one second of clear evidence that the other team is on the ball, or a reception (a fast ball stopped at a player), which pays for most of the switch at once. A deflection, a touch without control, or a pass that brushes an opponent doesn't pay the switching cost, so possession stays with the last team. This is the challenge's "loose or contested ball stays with the last team in possession".
- **Loose and contested balls** carry little evidence (the ball is far from everyone or moving fast), so they stay with the team that last had control.
- **Restarts.** A stoppage ends at the kick that restarts play. The team that restarts is decided by who is on the ball afterwards, not by the tag. Throw-in tags have no team.
- **Dead ball has no owner.** `D` is a state of its own, never a possession, as the ground truth defines it.
- **Abstention.** We write `X` only on ungraded frames: before the match, halftime, after the final whistle. Inside the halves we always commit to a label, because on spellF1 an `X` repeats the previous label anyway, and at a hand-over that is the expensive case.

## Training and tuning data

**Labels on the competition match:** none. The team-shape model is trained on pseudo-labels that the first pass of this pipeline produces from the tracking, which the rules allow ("pseudo-labels produced by an algorithm, provided no person decided any of them").

**Labels on other matches:** Metrica Sports' open sample games 1 and 2, which are anonymised, from [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
- **Who labelled them:** the event data was produced by Metrica for those sample releases. We labelled nothing by hand.
- **How much:** 3,680 events across two full matches.
- **How the events become frame labels:** `devset.py`, `event_labels()`, applies the challenge's own rules:
  - A team is in possession from its PASS, RECOVERY, SET PIECE or SHOT until the other team has one. BALL LOST and CHALLENGE alone don't change possession, because control has to be gained.
  - The ball is dead from BALL OUT, FAULT RECEIVED, or a shot that ends out or in the goal, until the next SET PIECE.
  - Frames before each period's kickoff, and after its last event, are not scored.
- **Size of the result:** 282,188 graded frames and 665 possessions, which is 3.5 possessions per minute. The public window of our match has 3.4 per minute, so the two label definitions seem to agree reasonably well.
- **What it is used for:** only to choose the pipeline's constants. No model is trained on it.

## How the dev set imitates this match

The sample tracking is clean, and ours is not, so `devset.py` scores every candidate under two conditions and averages the four results:
- **Natural:** the sample's own ball gaps (38–41% of graded frames, the same overall rate as ours), plus player dropouts. Each player vanishes for about 2 s every 25 s, which gives this match's roughly 7% missing players.
- **Harsh:** additionally, this match's real ball-missing pattern copied frame by frame onto the sample (62–64% missing in total).

There is one known difference. The sample drops the ball while it is out of play: it's missing on 85–96% of dead-ball frames there, against about a third in our public window. So the dev set gives our "ball crossed a line" and "placed ball" cues fewer chances to fire than our match does.

## How we chose the settings

**On the dev set.** A coordinate-descent sweep (`python devset.py --sweep`) tried each constant on a small grid. It kept a change only if the mean spellF1 over the four dev runs improved, and repeated over two rounds.

| setting | before | after | dev spellF1 after the change |
|---|---|---|---|
| starting point (priors) | | | 0.418 |
| slow-players threshold | 1.0 m/s | 1.5 m/s | largest single gain (+0.03 to +0.04) |
| dead evidence in a stoppage | 2 | 1 | +0.004 |
| kick speed that ends a stoppage | 6 m/s | 4 m/s | +0.002 |
| placed-ball speed | 0.5 m/s | 0.3 m/s | +0.0003 |
| H↔A switch cost | 15 | 25 | +0.006 |
| dead switch cost | 20 | 20 | moved to 45, then 15, then back to 20 as other settings changed |
| reception weight | — | 4 | +0.001 to +0.004 |
| control radius, margin scale, fast ball, shape weight | 2 m, 1 m, 10 m/s, 0.2 | unchanged | |
| **final** | | | **0.481** (macroF1 0.657 → 0.714) |

The gains after the slow-players threshold are each near the noise level. Together they are consistent across the two noise conditions.

**What the dev set says the system does wrong:**
- **It predicts too few possessions:** 866 against 1,330 labelled across the four runs. Lowering the switching cost to catch more of them lowers spellF1, so the missed ones are mostly too short to see reliably in noisy tracking.
- **It over-predicts dead ball:** 39–43% predicted against 26–27% labelled, with dead-ball recall about 0.87 and precision about 0.55. That is what maximises spellF1 there: a short stretch of `D` around an uncertain hand-over costs less than a wrong team.

**With the label-free gap test.** Smart Tagging doesn't exist for the sample matches, so the tag weight was chosen with `gapcheck.py`:
1. Run the pipeline on our match with the tracked ball and keep its answer as the reference.
2. Hide the ball in random 5–20 s windows where it *was* tracked (8.7% of the video).
3. Rerun each variant and splice its answer inside the hidden windows into the reference.
4. Score with the official spellF1, plus plain team agreement inside the hidden windows. Three random masks are averaged.

With the final settings:

| variant (prior weights) | spellF1 vs reference | team agreement in hidden windows |
|---|---|---|
| no priors (hold last team) | 0.9025 | 0.604 |
| shape 0.1 | 0.9156 | 0.674 |
| tags 0.1 + shape 0.1 | 0.9212 | 0.695 |
| tags 0.1 + shape 0.2 | 0.9214 | 0.704 |
| **tags 0.2 + shape 0.2 (submitted)** | **0.9225** | **0.706** |

This test measures consistency, not correctness. The reference is the pipeline's own answer with the ball tracked, and that answer has errors of its own. So it can rank ways of bridging gaps, but it can't say how close the reference is to the truth.

An earlier version of this test also exposed a design flaw. A tag prior applied everywhere could force a switch on its own after about 3 s of a phase, overriding the ball. That's why both priors now act only where the ball is missing.

**The public 5 minutes.** These were never used to fit or choose anything. The organisers show that ranking variants on 17 possessions picks the wrong one. We did score each version against the window as we built it, and that shaped where we spent effort:
- **v1** found about 6% of dead-ball frames there. That pointed to dead ball as the weak side, and v2 added the stoppage detector.
- **The pre-kickoff rule** came from the published fact that the second half is graded from frame 110,808 but kicks off at 113,742 (FAQ.md). It is implemented from the Smart Tagging kickoff tag, not from those frame numbers.
- **The dev-set tuning was finished before** the tuned version was scored on the window. Its result there (below) is an independent check, not a target.

## Results

**Whole match** (from `validate_submission.py`):
- 244 possessions with a mean of 14.9 s
- H 55,160 frames, A 35,689, D 68,008 (43% of graded frames are dead ball, in line with what the dev set predicts for its matches)
- home share of live play 61%

**Dev set (two other matches, four noise runs):** spellF1 0.481, macroF1 0.714.

**Public 5 minutes, as a correctness check:**

| | spellF1 | macroF1 | dead-ball F1 | possessions (truth 17) |
|---|---|---|---|---|
| this submission | **0.566** | **0.799** | 0.70 | 12 |
| the version before dev-set tuning | 0.471 | 0.674 | 0.41 | 17 |
| baseline_smooth50 (published floor) | 0.326 | 0.621 | | 26 |
| baseline_smooth125 | 0.424 | 0.581 | | 16 |

The public window improved by about as much as the dev set did, even though it took no part in the tuning. Where hand-overs are matched, our median boundary error is −1.5 frames. A one-frame offset is free on spellF1.

## Limitations, and what we would do with more time

- **Short possessions.** We predict about 70% of the possessions the dev labels contain (and 12 of 17 in the public window). Short possessions inside noisy tracking are merged with their neighbours. Lowering the switching cost doesn't fix this on the dev set; it only adds false possessions. Two things might:
  - better evidence at receptions, such as using the ball's direction change and the player's speed
  - a learned model of the evidence, trained on the dev set
- **Dead ball is over-predicted** (about 43% against a likely 25–30%). It pays on spellF1 in the dev set, but it costs macroF1 and might not transfer exactly. A stoppage detector trained on the dev labels would be the next step.
- **The dev set is only an approximation of our data.**
  - Its labels come from events, not from corrected tracking.
  - Its ball gaps lean towards dead ball more than ours.
  - Smart Tagging can't be tested on it.
  - It has two matches. Adding Metrica's sample game 3, which is in a different format, would make it larger.
- **Ball gaps.** During gaps we recover about 71% of team frames in the hidden-ball test. Options:
  - a richer shape model, e.g. per-player pressure features or a small sequence model
  - our own ball detector on the tactical video

## Reproduction

The match data goes in `data/` exactly as delivered (see `DATA.md`). It is gitignored and not part of this repository.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
.venv/bin/python possession.py <team_name>.csv      # about 4 s; prints label counts
.venv/bin/python evaluation/validate_submission.py <team_name>.csv
```

Optional, to reproduce how the settings were chosen:

```bash
# dev set: put Metrica sample games 1 and 2 in data/external/ (see devset.py), then
.venv/bin/python devset.py            # score the current settings on the dev set, about 30 s
.venv/bin/python devset.py --sweep    # the coordinate-descent sweep, about 10 min
.venv/bin/python gapcheck.py          # the label-free gap test, about 40 s
```

`possession.py` runs its own self-tests first:
- a 3-frame deflection must not split a possession
- a stoppage between two home possessions must survive
- a throw-in must open a stoppage and close it

The output is deterministic, and two runs give the same file (SHA-1 `7e7fa955c7b145802eb28c29b74f20bdbd661564`).

Files:
- `possession.py`: the whole pipeline
- `devset.py`: the labelled dev set from other matches, and the sweep
- `gapcheck.py`: the label-free gap test
- `requirements-solution.txt`: pinned versions
- `starter/baseline.py` and `starter/load_smart_tagging.py` from the organisers' starter kit, reused for the team-slot mapping, the CSV writer and the tag parser
