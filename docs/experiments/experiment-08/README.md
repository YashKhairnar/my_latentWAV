# Experiment 08: LeWM fine-tune on 205 episodes

## Goal

Test whether a much larger WAV dataset improves LeWM without making it forget
the original expert data.

## Setup

```text
Episodes:             205
Training sequences:   15,725
Validation sequences:  1,700
Training steps:       500
Learning rate:        0.000001
Visual encoder:       frozen
Dropout:              disabled
Dataset: /models/rollouts/latentwav_20260903T003251Z.h5
```

## Result

| Test data | Expert LeWM | Fine-tuned LeWM | Change |
|---|---:|---:|---:|
| Held-out WAV data | `0.035102` | `0.017632` | 49.77% better |
| Expert data | `0.007762` | `0.009159` | 18.00% worse |

## Decision

Keep the candidate for research, but do not replace the expert model yet.

```text
Active:    /models/checkpoints/lewm-pusht
Candidate: /models/checkpoints/lewm-wav-20260903-205ep-v2/best
```

Next, mix expert and WAV data during fine-tuning. Then compare both checkpoints
with the same goal-directed CEM-MPC controller.
