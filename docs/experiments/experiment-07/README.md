# Experiment 07: conservative LeWM fine-tune

## Goal

Try a safer update using more WAV data, a smaller learning rate, and a frozen
visual encoder.

## Setup

```text
Episodes:             50
Training sequences:   3,825
Validation sequences:   425
Training steps:       200
Learning rate:        0.000005
Visual encoder:       frozen
Dataset: /models/rollouts/latentwav_20260902T194103Z.h5
```

## Result

| Model | Prediction MSE |
|---|---:|
| Expert LeWM | `0.043756` |
| Fine-tuned LeWM | `0.054387` |

The fine-tuned model was `24.3%` worse.

## Decision

Reject `/models/checkpoints/lewm-wav-20260902-conservative/best`. The safer
update reduced the damage but still did not beat the expert model.
