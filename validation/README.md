# Hand-labelled validation set

The public window has only 17 possessions, too few to tell good model versions from lucky ones.
A few minutes of your own labels elsewhere in the match fixes that. The rules allow small
hand-labelled sets for **validating or training** a model; the submitted labels must still come
from the model, never from hand-labelling or hand-correcting its output (`RULES.md` §1).

Your label files stay out of git (they are derived from the video). Only this README and
`intervals_template.csv` are committed.

## Setup (once)

    winget install shinchiro.mpv

Not the "MPV.NET" one. The launcher finds mpv on your PATH or in `C:\Program Files\MPV Player`.

## How labelling works

You press a key only **when the situation changes**, never per frame. Each press means "from this
moment it is ...", and that label lasts until the next press.

| key | meaning |
|---|---|
| `h` | Barcelona (home) has the ball from here |
| `a` | Athletic (away) has the ball from here |
| `d` | dead ball from here |
| `x` | close the last stretch -- press before quitting |
| `space` | pause / play |
| `,` / `.` | one frame back / forward |
| `[` / `]` | slower / faster (half speed works well) |
| `←` / `→` | 5 s back / forward |

Each press shows a confirmation on screen. Five minutes of play is typically 20-40 presses. When
you see a change: pause, step a few frames with `,` `.` to the moment, press the key, resume.
About ±0.2 s is precise enough.

## The rules to apply (`CHALLENGE.md`)

- **Possession changes only when the other team gains *controlled* possession** -- the player keeps
  the ball for a moment, or the team completes passes. Date it to the first controlled touch.
- **Not a change:** a touch without control, a deflection that does not intercept the pass, an
  aerial duel won without controlling the ball afterwards, a loose or contested ball. Possession
  stays with the last team that had it.
- **Dead ball (`d`):** from the ball crossing a line, or the whistle for a foul or offside (no `d`
  if advantage is played), a goal, an injury -- until the restart is taken.
- **Restarts:** the team taking the throw-in, free kick, corner, goal kick or kick-off has the ball
  from the moment it is taken. Never label who "owns" a dead ball: it is just `d`.
- **Label from the video only.** Do not look at the model's output while labelling, or your
  labels will agree with it more than they should.

Times are tactical-video times from the player, not the match clock burned into the picture
(the video starts 7:27 before kick-off; the second half kicks off at 75:49.6).

## Step 1: calibrate (about 30 min)

Label the public window and score yourself against the official labels, to check you judge
"control" and "dead ball" the way the organisers did:

    .\tools\label.bat 7:27.4 validation/calibration.csv

Press the first key at 7:27.4 and `x` at 12:27.4. Then:

    .venv/Scripts/python.exe tools/make_validation_gt.py --intervals validation/calibration.csv --out validation/calibration/mine.csv
    .venv/Scripts/python.exe evaluation/run_evaluation.py --gt public --results validation/calibration --out validation/calibration_score

macroF1 above about 0.9 and close to 17 possessions means your conventions match. If not, look
at where you differ (usually when `d` starts, or when control begins) and adjust.

## Step 2: label new stretches

Two stretches of 4-5 minutes, one per half, well away from the public window:

    .\tools\label.bat 25:00
    .\tools\label.bat 95:00

Both go to `validation/intervals.csv`. Contested midfield play and a few restarts are the most
useful to have in there.

## Step 3: score the models on your labels

    .venv/Scripts/python.exe tools/make_validation_gt.py
    .venv/Scripts/python.exe evaluation/run_evaluation.py --gt validation/validation_gt.csv

## Files

| file | what |
|---|---|
| `validation/intervals.csv` | your labels (`start,end,label`, in frames: divide by 25 for seconds) |
| `validation/calibration.csv` | your labels of the public window |
| `validation/validation_gt.csv` | built by `make_validation_gt.py`; what the scorer reads |
| `validation/intervals_template.csv` | example of the format |

- **Saved as you go:** each stretch is written when you press the next key; only the open stretch
  is lost if mpv closes without `x`.
- **Never overwritten:** new presses are appended. Labelling a second stretch adds to the file.
- **To redo:** delete the file to start over, or delete the rows of one stretch in Notepad/Excel and
  label it again. Overlapping rows are not silently accepted: `make_validation_gt.py` stops and
  names the line.

## If training on these labels later

Keep a separate set that is only ever used for validation, keep the amount small, and ask the
organisers before committing labels to the repo (it must regenerate the submission). Describe
what was labelled and how it was used in the technical document.
