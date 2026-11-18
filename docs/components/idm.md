# IDM: the action guesser

## What does it do?

The IDM looks at two pictures:

1. what PushT looks like now;
2. what we want it to look like next.

It guesses five small actions that could move from the first picture toward the
second picture.

```text
current picture + future picture -> five actions
```

Each action has two numbers: movement in the X and Y directions.

Main code:

- `src/latentwav/models/idm.py`
- `scripts/train_idm.py`
- `scripts/modal_train_idm.py`

## How was it trained?

We used the full PushT expert dataset. For each training example, the model
received:

```text
picture at step t
picture at step t+5
the five real actions taken between them
```

The model learned to predict those five actions. Complete episodes were kept
together so pictures from one episode could not leak into both training and
validation data.

Initial training settings:

```text
Epochs:        5
Batch size:    512
Learning rate: 0.0003
GPU:           Modal H100
```

An epoch means one pass through the training data.

## How well did it work?

The first run's best model came from epoch 5. We then resumed training through
epoch 15 with a new learning-rate schedule. The final epoch produced the best
result.

```text
Epoch 5 validation error:  0.09551
Epoch 15 training error:   0.07857
Epoch 15 validation error: 0.09098
Validation improvement:    about 4.7%
```

Lower error is better. The two numbers are close, so there is no clear sign
that the model memorized the training data.

In the environment's original action units, the model's typical error was
about `0.064` per predicted action value.

The fifth action was slightly harder to predict than the earlier actions. This
makes sense because predictions become less certain farther into the future.

The epoch-15 best checkpoint is stored on Modal at:

```text
/models/checkpoints/idm/idm_pusht_best.pt
```

## Why did we train for longer?

The first run's best result happened at its final epoch. This suggested the
model might still be improving.

We continued from epoch 5 through epoch 15 without throwing away what the model
had learned. Validation error improved from `0.09551` to `0.09098`.

```text
Existing training:  epochs 1–5
Continued training: epochs 6–15
```

## What was wrong with `T_max=5`?

The learning rate controls how large each training update is. The cosine
scheduler gradually reduced that learning rate over five epochs because it was
created with `T_max=5`.

By the end of epoch 5, that schedule was finished. Reusing it for epochs 6–15
could make the learning rate change in an unintended way.

The updated resume code now:

1. keeps the trained model;
2. keeps the optimizer and training history;
3. resets the learning rate;
4. creates a new schedule for the 10 remaining epochs.

## How do I resume training?

Run:

```bash
.venv-nanowm/bin/modal run scripts/modal_train_idm.py \
  --full \
  --epochs 15
```

Here, `15` means 15 epochs in total. Because five epochs are already complete,
the job should run epochs 6 through 15.

The script automatically resumes from:

```text
/models/checkpoints/idm/idm_pusht_best_progress.pt
```

## Is the new model better?

It is better on held-out action prediction because its validation error is
lower. One important check remains:

1. test whether its actions work better when trying to reach subgoals in PushT.

A lower numeric error is useful, but actual behavior in PushT is the more
important test.

## Current limitation

This IDM is a normal dense action predictor. It does not yet include the WAV
paper's learnable sparsity mechanism.
