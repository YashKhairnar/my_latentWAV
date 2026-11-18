# Collection and evaluation

## What does this part do?

Collection runs the policy in PushT and saves what really happened. Evaluation
checks whether a newly trained model is better than the old one.

Main code:

- `src/latentwav/loop.py`
- `src/latentwav/visualization.py`
- `scripts/modal_collect_pusht.py`
- `scripts/compare_lewm.py`

## What is saved?

For every environment step, we save the picture, executed action, simulator
state, reward, disagreement score, selected subgoal, and planning-step number.
The saved file can be loaded by the training code.

## What have we collected?

Small test:

```text
1 episode
50 steps
45 usable IDM examples
35 usable LeWM examples
```

Larger collection:

```text
10 episodes
1,000 total steps
950 usable IDM examples
850 usable LeWM examples
```

The larger file is stored at:

```text
/models/rollouts/latentwav_20260812T235844Z.h5
```

GIFs and plots are under `artifacts/rollouts/`.

## How do we evaluate a new model fairly?

The old and new models must predict the same untouched examples. We compare
their errors and keep the new model only if it improves.

The real future picture may be used for evaluation after an action happens. It
must not replace the verifier's disagreement rule when choosing the action.

## What is missing?

We need an automated experiment that repeats these steps:

```text
collect -> train IDM and LeWM -> validate -> accept or reject -> repeat
```

We also need to compare WAV with random action selection using the same number
of environment interactions.
