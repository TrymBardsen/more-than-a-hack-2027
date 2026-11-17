# Who has the ball? Frame-level possession from automatic tracking

More than a Hack 2027 · FC Barcelona v Athletic Club

This is the short report. Every number, experiment and design decision is documented in
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## In brief

For each of the 199,319 frames of the tactical video we say whether Barcelona (`H`) or Athletic (`A`) has the ball, or whether the ball is dead (`D`). The method:
- turns the automatic tracking into evidence of who is on the ball and whether play has stopped
- weighs that evidence with a small model trained on **other matches**
- decodes it into possessions

On a labelled match the model had never seen, it scores **spellF1 0.545**, against 0.243 for the organisers' baseline on the full match. It runs in about 3 seconds and reproduces the submission byte for byte.

## The approach

The ranking metric scores whole possessions, not frames. A system that is right on most frames but flickers between teams splits real possessions into many small ones and scores badly. So we built the method around possessions:

1. **Evidence per frame.** Which team's nearest player is closest to the ball, how fast the ball moves, and whether a fast ball has just been stopped at a player's feet (a controlled touch). We also record who touched the ball last and who touches it next, and how both teams are shaped on the pitch.
2. **Who has the ball.** Two small learned models weigh that evidence: one for frames where the ball is tracked and one for frames where it is missing. They are trained on labelled data from other matches (see *Training data*).
3. **Is the ball dead.** A stoppage starts when the ball crosses a line, sits still for a set piece, or a set-piece tag begins, and it ends at the next kick. Players slowing down as a group also points to a stoppage, and each half is dead until its kickoff.
4. **From frames to possessions.** A Viterbi decoder picks the most likely sequence of `H`, `A` and `D`. Every change of state has a cost, so possession only changes when the evidence for it is strong and lasts. A deflection or a single noisy frame can't split a possession.

### Key numbers

| Part | Setting |
|---|---|
| Team model | Two logistic regressions with 21 inputs each: one for when the ball is tracked, one for when it's missing. Trained on every frame twice, once as recorded and once with home and away swapped, so they can't learn a lean towards either side. |
| Team evidence | The models' log-odds that home has the ball: full weight where the ball is tracked, 0.25 where it's missing, capped at ±6 per frame |
| Controlled touch | A ball arriving faster than 6 m/s that leaves slower than 3 m/s within about 0.3 s, within 1.5 m of a player |
| Decoder costs | 15 for a change of team, 30 for entering or leaving dead ball, in the same log-odds units. One frame can never pay for a change on its own. |
| Stoppage start | Ball more than 1 m past a line; or ball still (under 0.3 m/s over 1.5 s); or a set-piece tag |
| Stoppage end | The next kick on the pitch (ball faster than 4 m/s), at least 1 s after the stoppage began |
| Slow players | Group mean speed over ±2 s below 1.5 m/s counts towards dead ball |
| Before kickoff | Dead at full strength, with no team evidence, until the kick that the kickoff tag announces |
| Tag hint while the ball is missing | ±0.4 per frame, from the team named by Smart Tagging's attacking and defending phases |

## What we used, and what we dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (players and ball) | Yes | The main input |
| Smart Tagging | Yes | Set pieces mark restarts, kickoffs mark the start of each half, and phase tags hint at the team in possession while the ball is missing |
| Metrica's open sample matches (other games) | Yes | Training and tuning data |
| Homography | Tested, dropped | The camera's aim was too far from the ball (24 m median) to help |
| Audio track | Tested, dropped | Referee whistles were too faint to separate from the crowd |
| Video images, broadcast feed | No | Our own ball detection would take days for an uncertain gain; the broadcast isn't synchronised to the tactical video |

## Missing and noisy tracking

- **The ball is missing on 41% of graded frames,** mostly in gaps longer than 5 seconds, so interpolation can't fill them. A separate model handles those frames. It knows who touched the ball last and who touches it next (fading over about 5 s) and how both teams are shaped, and the Smart Tagging hint adds to it.
- **Tracker jumps.** Ball detections that move more than 2 m in one frame (over 50 m/s) are dropped as errors, and ball speed is computed from positions.
- **Track slots aren't people.** Players appear, vanish and swap slots, so every feature is team-level: each team's nearest player to the ball, centroid, spread, back line (its second-deepest player, which skips the keeper) and most advanced player.
- **Noise-hardened training.** The sample matches we trained on were given this match's ball-missing pattern and random player dropouts (about 7% of players missing), so the models learned from tracking as noisy as ours.

## Training data

- **No labels on the competition match were used**, by hand or otherwise.
- **The public 5 minutes of ground truth were used only as a check:** that our output was aligned and well-formed, and how each version behaved. Nothing was trained or tuned on them, and no version was chosen by its score there. An early version's weak dead-ball result there is what pointed us towards the stoppage detector.
- **The learned model, and all labelled tuning, used Metrica's open sample games 1 and 2** ([github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)). RULES.md explicitly allows labels from other games.
  - Their event data (3,680 events, produced by Metrica) was turned into frame labels with the challenge's own possession rules. We labelled nothing by hand.
  - To make those cleaner matches resemble ours, we copied in this match's real pattern of missing-ball frames and added player dropouts.
  - The learned model's settings were chosen by training on one sample match and scoring the other.
  - The stoppage thresholds and decoder costs were tuned on both sample matches, then checked by leaving one out: tuning on one match improved the unseen one by +0.065 and +0.049 spellF1.
- **One setting was chosen on our own match, without labels.** The sample matches have no Smart Tagging, so the weight of the tag hint was set with a label-free test: hide the ball where it *was* tracked, and measure how consistently the pipeline recovers its own answer from when the ball was visible.

## Possession rules we implemented

- **A change of possession needs control.** The other team has to be clearly on the ball. A touch, a deflection or a won header without follow-up control doesn't change possession.
- **Loose and contested balls stay with the last team in possession.**
- **Restarts.** A stoppage ends at the kick that restarts play. The team that restarts is whoever is on the ball afterwards.
- **Dead ball has no owner.** When the ball is dead the answer is `D`.
- **No abstaining during play.** We commit to a label on every graded frame and write `X` only before the match, at half-time and after the final whistle.

## Results

| Measure | Score |
|---|---|
| spellF1 on a labelled match the model never saw | **0.545** |
| The same with hand-set rules instead of the learned model | 0.485 |
| Public 5 minutes (17 possessions): spellF1 / macroF1 | 0.483 / 0.816 |
| Organisers' baseline, full match | 0.243 |

The public window is too small to rank versions: the official scorer's bootstrap puts its noise band at ±0.36. We treat it only as a correctness check.

## Limitations

- **Short possessions are often missed.** Possessions under 5 seconds are about 40% of all possessions, and many are merged into their neighbours.
- **Ball gaps.** The tracking loses the ball on 41% of graded frames. During live play with the ball missing, we pick the right team about 65% of the time.
- **Too much dead ball.** We predict more stoppage time than the labelled matches contain, because the signals that catch real stoppages also fire during slow play.
- **The training matches differ from ours.** Their labels come from event data, not corrected tracking, and their tracking is cleaner.

With more time, we would add our own ball detection from the video to shrink the ball gaps, and train on more labelled matches.

## Reproduction

With the match data in `data/` as delivered:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-solution.txt
.venv/bin/python possession.py <team_name>.csv
.venv/bin/python evaluation/validate_submission.py <team_name>.csv
```

Two runs produce the same file. Retraining the model (`devset.py --train`, which needs the Metrica sample data) reproduces `evidence_model.json` exactly.

*The code was developed with an AI coding assistant.*
