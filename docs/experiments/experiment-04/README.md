# Experiment 04: collect WAV data

## Goal

Collect high-disagreement PushT experience for fine-tuning LeWM.

## Result

```text
Episodes:                10
Steps per episode:       100
Actions before replanning: 5
Saved rows:            1,000
LeWM sequences:          850
Dataset: /models/rollouts/latentwav_20260812T235844Z.h5
```

## Decision

Use nine episodes for training and keep one episode for validation.
