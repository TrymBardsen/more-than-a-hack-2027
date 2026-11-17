# Frame-level possession from automatic tracking

More than a Hack 2027 · FC Barcelona v Athletic Club · technical document

## Summary

We label each of the 199,319 tactical-video frames as `H`, `A`, `D`, or `X` (ungraded frames only).
The pipeline:
1. Turns the automatic tracking into per-frame features: who is near the ball, how the ball moves, who touched it last and next, the shape of both teams, and whether play has stopped.
2. Scores "which team has the ball" with a **learned evidence model**: two small logistic regressions, one for frames where the ball is tracked and one for frames where it is missing.
3. Scores "is the ball dead" with a hand-built stoppage detector whose thresholds were tuned on labelled data.
4. Decodes each half with a three-state Viterbi path that charges a fixed cost for every change of state.

Everything learned or tuned used a labelled dev set built from **two other matches**: Metrica's open sample games, with possession labels derived from their event data. RULES.md allows labels on other games. We made those matches look like ours by copying in this match's ball-missing pattern and adding player dropouts.

No human labels on the competition match were used anywhere, and nothing was fitted to its public 5 minutes.

Measured on a sample match the model was **not** trained on, spellF1 is **0.545**. The hand-made version scores 0.485, even though it was tuned on that same match. The code is about 850 lines of numpy and pandas across three files, plus a 128-number model file. Producing the submission takes about 3 seconds and is byte-for-byte reproducible.

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
| Smart Tagging | yes | set-piece starts open stoppages; "Kick Off (Start)" anchors the pre-kickoff dead ball; phase team tags are a prior while the ball is missing |
| homography | no (tested, dropped) | we tested the camera's aim, the image centre mapped onto the pitch, as a rough ball position. It is a median 24 m from the ball, and as a team-shape feature it lowered held-out accuracy (0.805 → 0.789) |
| tactical video, audio track | no (tested, dropped) | we tested referee-whistle detection on the audio. Tonal bursts land near stoppage starts no more often than random times (5–7% against 10%), because the camera microphone mostly records the crowd |
| tactical video, images | no | the ball is about 5 px in a wide shot; a detector run would have cost hours for a share of the gaps |
| broadcast video, ATD, homography | no | not synchronised to the tactical timeline, with cuts and replays; its ball is missing about half the time as well |
| **Metrica open sample data, games 1 and 2** (other matches) | yes, for training and tuning | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data): tracking plus event data, used to build the labelled dev set |
| Metrica open sample game 3 | no | a different event format (EPTS with CARRY events); dropped for time |

What the raw ATD does and does not contain shaped the design:
- **No owning-team or ball-dead field, and no ball height.** The ball's third value is mostly missing, so we compute ball speed from positions.
- **Track slots aren't people.** Every feature is team-level: nearest player per team, centroid, spread.
- **The ball is missing on 41% of graded frames**, mostly in long gaps: 98% of missing frames sit in gaps longer than 5 s.

## Features

All distances are in metres, using the pitch size from `games.py` (105.3 × 68.0). Every constant is in `possession.py` under `# --- priors`, with a one-line reason.

**Hand-made team score** (`features()`). This was the team evidence before the learned model, and it is now one of the model's inputs:
- For each team, take the distance from its nearest player to the ball, and use the difference `d_away − d_home`.
- It is weighted by `exp(−nearest / 2 m)`, and weighted down by a factor of 5 when the ball moves faster than 10 m/s.
- It is clipped to ±3.
- Add +4 per frame for a **reception**: a ball arriving faster than 6 m/s that leaves slower than 3 m/s within about 0.3 s, within 1.5 m of a player.
- Ball detections that jump more than 2 m in one frame are dropped as tracker errors.

**Learned-model inputs** (`evidence_features()`), 21 per frame, each either pointing towards home or neutral between the sides:
- the hand-made team score
- the distance margin between the teams' nearest players, and the margin weighted by closeness
- the nearest player's distance to the ball, and the ball speed
- the reception flag, signed by team
- **touch context:** the side of the last clear touch and of the next one (+1 home, −1 away), each fading over about 5 s. This is what carries "who had it" into and out of a ball gap.
- 13 team-shape features from the home side's point of view, with attacking direction found per half from the team centroids:
  - centroid gap, and home centroid position
  - forward velocity of each team
  - differences and sums of team spread in x and y
  - each team's back line (second-deepest player, which skips the keeper) and its most advanced player
  - the difference in attacking-half occupancy

## The learned evidence model

There are two logistic regressions (deterministic IRLS, L2 = 1, numpy only):
- `seen`, for frames where the ball is tracked
- `gap`, for frames where it is missing

Each predicts the probability that home has the ball. Its log-odds become the team evidence in the decoder: weight 1 where the ball is tracked and 0.25 where it is missing, capped at ±6. The coefficients live in `evidence_model.json` (per model: 22 weights, plus 21 means and 21 scales to standardise the inputs), so producing the submission needs no external data. `python devset.py --train` regenerates the file exactly.

**Training data:**
- every labelled live frame of both sample matches, under both noise conditions (below)
- each frame used twice: once as recorded, and once **mirrored**, with home and away swapped and the label flipped

The mirroring makes the model side-neutral. Without it, the model would learn the sample matches' home possession share as a lean towards home, and that lean has nothing to do with Barcelona and Athletic. Mirroring cost 0.008 spellF1 on the dev set, which is within noise. We accept that cost for a model whose decisions don't depend on which side is at home.

**Per-frame accuracy on a match the model never saw** (train on one sample match, test on the other):

| | ball tracked | ball missing |
|---|---|---|
| hand-made team score (sign) | 0.77–0.79 | (no evidence) |
| learned model | **0.92–0.93** | **0.63–0.64** |

## Dead-ball evidence

- **Stoppages.** A small state machine treats a stoppage as an interval.
  - It opens when one of these happens:
    - the ball crosses a line (seen inside the pitch within the last second, then more than 1 m beyond a line)
    - the ball is placed: it averages under 0.3 m/s over 1.5 s
    - a Smart Tagging set piece starts
  - It closes at the next kick on the pitch (ball faster than 4 m/s), at least 1 s after opening, or after 2 minutes.
  - Inside a stoppage every frame gets +1 dead evidence.
- **Before kickoff.** A half is dead until it kicks off. From the first graded frame of each half through the stoppage that the half's "Kick Off (Start)" tag opens, up to the kick that closes it, dead evidence is at full strength (+3) and team evidence is zero. The kickoff comes from the tagging, not from a constant.
- **Slow players.** `2 × (1.5 m/s − mean player speed over ±2 s)`, clipped to ±3. The dev set's player speeds, computed from positions, have almost the same distribution as this match's tracker speeds: the 20th percentile is 0.97–1.03 m/s there and 0.95 m/s here. So the threshold carries over.
- **Ball out.** A ball tracked more than 1 m beyond a line gets +3 on that frame.

**Smart Tagging phase prior (0.4 per frame, only where the ball is missing).**
- Attacking phases (Build Up, Progression, Creating Chances, Attacking Transition, Ball in Final Third, Ball in the Box, Players in the Box, Long Ball) name the team on the ball.
- Defending phases (the three Defending thirds, Defensive Transition) name the other team.

## Decoding: the possession rules

The Viterbi path runs over each half separately.
- **Emission scores:** H gets `+team/2`, A gets `−team/2`, D gets the dead evidence.
- **Transition costs:** H↔A costs 15, and entering or leaving D costs 30. Both were tuned across matches.

The rules this encodes:
- **A change of possession needs sustained control.** The other team has to be clearly on the ball for long enough to pay the switching cost. A clean reception with good learned evidence does that in well under a second. A deflection or a brushing touch does not, so possession stays with the last team. This is the challenge's "loose or contested ball stays with the last team in possession".
- **Restarts.** A stoppage ends at the kick that restarts play. Who restarts is decided by who is on the ball afterwards, not by the tag. Throw-in tags have no team.
- **Dead ball has no owner.** `D` is a state of its own, never a possession.
- **Abstention.** We write `X` only on ungraded frames. Inside the halves we always commit to a label, because on spellF1 an `X` repeats the previous label anyway, and at a hand-over that is the expensive case.

The hand-made path from the previous version is still in the code (`USE_MODEL = False`). It uses its own tuned costs (25 and 20) and boundary snapping. The learned path doesn't snap boundaries, because the cross-match test showed no gain from it there.

## Training and tuning data

**Labels on the competition match:** none.

**Labels on other matches:** Metrica Sports' open sample games 1 and 2, which are anonymised, from [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
- **Who labelled them:** the event data was produced by Metrica for those sample releases. We labelled nothing by hand.
- **How much:** 3,680 events across two full matches.
- **How the events become frame labels:** `devset.py`, `event_labels()`, applies the challenge's own rules:
  - A team is in possession from its PASS, RECOVERY, SET PIECE or SHOT until the other team has one. BALL LOST and CHALLENGE alone don't change possession, because control has to be gained.
  - The ball is dead from BALL OUT, FAULT RECEIVED, or a shot that ends out or in the goal, until the next SET PIECE.
  - Frames before each period's kickoff, and after its last event, are not scored.
- **Size of the result:** 282,188 graded frames and 665 possessions, which is 3.5 possessions per minute. The public window of our match has 3.4 per minute, so the two label definitions seem to agree reasonably well.
- **What it is used for:** training the evidence model, and tuning the stoppage thresholds and decoder costs.

**How the dev set imitates this match.** Every candidate is scored under two conditions:
- **Natural:** the sample's own ball gaps (38–41% of graded frames, the same overall rate as ours), plus player dropouts. Each player vanishes for about 2 s every 25 s, which gives this match's roughly 7% missing players.
- **Harsh:** additionally, this match's real ball-missing pattern copied frame by frame onto the sample (62–64% missing in total).

There is one known difference: the sample drops the ball while it is out of play, so its ball gaps lean towards dead ball more than ours do.

## How we chose the settings

**1. Stoppage thresholds and hand-path costs** (`python devset.py --sweep`). A coordinate-descent sweep over the four dev runs:

| setting | before | after | effect |
|---|---|---|---|
| slow-players threshold | 1.0 m/s | 1.5 m/s | largest single gain (+0.034) |
| dead evidence in a stoppage | 2 | 1 | +0.004 |
| kick speed that ends a stoppage | 6 m/s | 4 m/s | +0.002 |
| placed-ball speed | 0.5 m/s | 0.3 m/s | +0.0003 |
| hand path: H↔A cost, dead cost | 15, 20 | 25, 20 | +0.006 |
| hand path: reception weight, snapping | 0, off | 4, ±2 s | +0.004, +0.0035 |
| **hand path total** | 0.418 | **0.485** | |

**2. Do these settings carry across matches?** (`python devset.py --lomo`). Starting from the hand-set values, we tuned on one sample match only and scored the other:

| tune on → test on | hand-set settings | tuned on the other match | tuned on both |
|---|---|---|---|
| game 1 → game 2 | 0.427 | **0.492** (+0.065) | 0.502 |
| game 2 → game 1 | 0.409 | **0.458** (+0.049) | 0.461 |

Both runs settle on nearly the same settings, so tuning generalises across the two matches.

**3. The learned model** (`python devset.py --tune-learned`). Every candidate was scored **across matches**: train the model on one sample match, score the other, both directions, and average. So the choice can't reward memorising a match.

| step | cross-match spellF1 |
|---|---|
| hand-made pipeline, for reference (tuned on both matches, so it is flattered here) | 0.485 |
| learned model, default weights | 0.529 |
| tuned weights and costs: tracked 1, gap 0.5, H↔A 15, dead 30, no snapping | 0.549 |
| with mirrored training (side-neutral) | 0.541 |
| gap weight 0.25 (submitted) | **0.545** |
| the same with the ball-missing model switched off (gap weight 0) | 0.525 |

**4. The tag prior** can't be tested on the dev set, which has no Smart Tagging. We tuned it with the label-free gap test, `gapcheck.py`:
1. Run the pipeline on our match with the tracked ball and keep its answer as the reference.
2. Hide the ball in random 5–20 s windows where it *was* tracked (8.7% of the video).
3. Rerun and splice the answer inside the hidden windows into the reference.
4. Score with the official spellF1, plus team agreement inside the hidden windows. Three masks are averaged.

| tag prior weight | spellF1 vs reference | team agreement in hidden windows |
|---|---|---|
| 0 | 0.894 | 0.569 |
| 0.1 | 0.904 | 0.616 |
| 0.2 | 0.911 | 0.652 |
| **0.4 (submitted)** | **0.923** | **0.700** |
| 0.8 | 0.915 | 0.698 |
| 1.2 | 0.911 | 0.695 |

This test measures consistency, not correctness. The reference is the pipeline's own answer with the ball tracked, but that answer is now 92% right per frame on unseen dev data. An earlier version of the test also exposed a design flaw: a tag prior applied everywhere could force a switch on its own after a few seconds of a phase, overriding the ball. That's why the priors only act where the ball is missing.

**5. The public 5 minutes.** These were never used to fit or choose anything. We scored each version on the window as we built it:
- **v1** found about 6% of dead-ball frames there. That pointed to dead ball as the weak side, and v2 added the stoppage detector.
- **The pre-kickoff rule** comes from the published fact that the second half is graded from frame 110,808 but kicks off at 113,742 (FAQ.md). It is implemented from the Smart Tagging kickoff tag, not from those frame numbers.
- **All later changes were decided on the dev set first** and only then scored on the window, as a check.

## Results

**Whole match** (from `validate_submission.py`):
- 266 possessions with a mean of 14.3 s
- H 55,797 frames, A 39,579, D 63,481 (40% of graded frames are dead ball)
- home share of live play 58%
- pre-kickoff stretch of the second half: 2,886 of 2,934 frames dead

**Dev set, scored on a match the model never saw:** spellF1 **0.545**.

**Public 5 minutes (17 possessions, as a correctness check):**

| version | spellF1 | macroF1 | team accuracy while live | possessions (truth 17) |
|---|---|---|---|---|
| **this submission (learned evidence)** | 0.483 | **0.816** | **0.914** | 12 |
| hand-made, tuned, with snapping | 0.538 | 0.795 | | 12 |
| hand-made, tuned, no snapping | 0.566 | 0.799 | | 12 |
| hand-made, before dev tuning | 0.471 | 0.674 | | 17 |
| baseline_smooth50 (published floor) | 0.326 | 0.621 | | 26 |

On this window the per-frame measures improved with the learned model and spellF1 dropped. The scorer's own bootstrap (`run_evaluation.py --bootstrap 1000`) puts the noise band for 17 possessions at **±0.36**, so the spellF1 differences between these versions are within it. The dev-set result is about 40 times larger and was measured on a match the model never saw. That is the basis of every choice.

## Limitations, and what we would do with more time

- **Short possessions.** On the dev set we match few possessions under 5 s, and they make up about 40% of all possessions. The learned evidence improved per-frame accuracy a lot, but the decoder still needs about one solid touch to switch. A model that predicts *the moment of a change* directly, rather than who has the ball, would target this best.
- **Ball gaps.** To see what goes wrong while the ball is missing, we took live play where the sample *did* track the ball and hid it with this match's gap pattern (63,805 frames, scored on a match the model never saw):
  - 64.6% get the right team, 17.4% the wrong team, and 18.0% are labelled dead.
  - Most of the false dead ball comes from the slow-players signal (76%, including frames where the stoppage detector also fired). Players stroll during slow build-up, as they do at stoppages.
  - We tried counting slow players in full only when a stoppage was confirmed (stoppage detector active, or the ball vanished within 3 m of a line). At weight 0 for unconfirmed slowness this cut false dead on live play from 18% to 8%, but it also cut dead-ball recall from 86% to 47% and spellF1 from 0.545 to 0.512. Intermediate weights were in between, so we dropped the idea: slow players are the main way real stoppages get caught.
  - For the wrong-team share we tried giving the model the structure of each stretch between clear touches:
    - the side of the touch before and after, weighted by how near this frame is to each
    - whether both ends are the same team
    - how long the stretch is
  - Scored across matches, spellF1 stayed at 0.544–0.546 (0.545 without), and the wrong-team rate on live play with the ball hidden rose from 17% to 21–22%. The model already sees the last and next touch through its touch-context inputs, so we dropped it. A real fix probably needs information the tracking doesn't give while the ball is missing: our own ball detection, or a sequence model trained on far more labelled matches.
- **Dead ball** is still hand-built and over-predicted: 39% predicted on the dev set against 26.5% in its labels, and 40% on our match. The dev labels themselves miss stoppages that have no event, such as offside or fouls without a FAULT RECEIVED, so part of the gap is in the labels. A learned dead-ball model was left out because the sample data drops the ball when it goes out: it would learn "ball missing means dead", which is much less true in our tracking.
- **The dev set is an approximation of our data.**
  - Its labels come from events, not corrected tracking.
  - Its ball gaps lean towards dead ball.
  - It has no Smart Tagging.
  - It has two matches. Adding Metrica's game 3, or other open tracking-plus-event datasets, would give the model more to learn from.
- **The public window disagrees, within noise.** Each of the last three changes improved the dev set and scored slightly lower on the 17 public possessions. Every gap is far inside the ±0.36 band, but it is the one signal from the real match, and the reason we can't claim more than an estimate for the full match.

## Reproduction

The match data goes in `data/` exactly as delivered (see `DATA.md`). It is gitignored and not part of this repository.

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
.venv/bin/python possession.py <team_name>.csv      # about 3 s; uses the committed evidence_model.json
.venv/bin/python evaluation/validate_submission.py <team_name>.csv
```

Optional, to reproduce how the model and settings were made (put Metrica sample games 1 and 2 in `data/external/`, see `devset.py`):

```bash
.venv/bin/python devset.py --train          # retrain evidence_model.json (identical file), about 30 s
.venv/bin/python devset.py --tune-learned   # cross-match tuning of the learned path, about 15 min
.venv/bin/python devset.py --sweep          # stoppage thresholds and hand-path costs, about 10 min
.venv/bin/python devset.py --lomo           # leave one match out, about 10 min
.venv/bin/python gapcheck.py                # label-free gap test for the tag prior, about 1 min
```

`possession.py` runs its own self-tests first:
- a 3-frame deflection must not split a possession
- a stoppage between two home possessions must survive
- a throw-in must open a stoppage and close it
- snapping must move a late boundary to the new team's first touch, and never past a later touch by the old team

The output is deterministic, and two runs give the same file (SHA-1 `7e796660105036eed4523e728eafa5eb7a9e1004`).

Files:
- `possession.py`: the whole pipeline
- `evidence_model.json`: the learned coefficients
- `devset.py`: the labelled dev set from other matches, training and tuning
- `gapcheck.py`: the label-free gap test
- `requirements-solution.txt`: pinned versions
- `starter/baseline.py` and `starter/load_smart_tagging.py` from the organisers' starter kit, reused for the team-slot mapping, the CSV writer and the tag parser
