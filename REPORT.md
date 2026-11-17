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

## What we used, and what we dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (players and ball) | Yes | The main input |
| Smart Tagging | Yes | Set pieces mark restarts, kickoffs mark the start of each half, and phase tags hint at the team in possession while the ball is missing |
| Metrica's open sample matches (other games) | Yes | Training and tuning data |
| Homography | Tested, dropped | The camera's aim was too far from the ball (24 m median) to help |
| Audio track | Tested, dropped | Referee whistles were too faint to separate from the crowd |
| Video images, broadcast feed | No | Our own ball detection would take days for an uncertain gain; the broadcast isn't synchronised to the tactical video |

## Training data

- **No labels on the competition match were used**, by hand or otherwise.
- **The public 5 minutes of ground truth were used only to check** that our output was aligned and well-formed, never to train or tune.
- **Everything learned or tuned came from Metrica's open sample games 1 and 2** ([github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)). RULES.md explicitly allows labels from other games.
  - Their event data (3,680 events, produced by Metrica) was turned into frame labels with the challenge's own possession rules. We labelled nothing by hand.
  - To make those cleaner matches resemble ours, we copied in this match's real pattern of missing-ball frames and added player dropouts.
  - Every setting was chosen by training on one sample match and testing on the other, so it couldn't simply memorise a match.

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
