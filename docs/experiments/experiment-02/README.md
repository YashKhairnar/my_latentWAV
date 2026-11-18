# Experiment 02: train the IDM

## Goal

Train the IDM to predict five PushT actions from a current image and a future
image.

## Result

Training for 15 epochs improved validation MSE from `0.09551` to `0.09098`.

```text
Dataset:             full PushT expert dataset
Training examples:   2,020,690
Validation examples:   222,621
Best epoch:          15
Best validation MSE: 0.09098
Checkpoint:          /models/checkpoints/idm/idm_pusht_best.pt
```

## Decision

Use the epoch-15 checkpoint in the LatentWAV pipeline.
