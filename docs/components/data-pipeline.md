# Data pipeline: the training-example builder

## What does it do?

The data pipeline reads PushT experience and turns it into examples that the
IDM and LeWM can learn from.

Main code: `src/latentwav/data.py`

## What data do we have?

The expert dataset contains:

```text
2,336,736 rows
18,685 episodes
```

Each row can contain a picture, action, simulator state, episode number, and
step number.

Modal location:

```text
/models/datasets/lewm-pusht/pusht_expert_train.h5
```

## How are IDM examples made?

For a five-action example:

```text
picture at step t
picture at step t+5
actions from t through t+4
```

The loader checks that all of these belong to the same episode. It never joins
the end of one episode to the start of another.

## How is validation kept fair?

Whole episodes go into either training or validation. An episode is never
split between both groups. This prevents nearly identical neighboring pictures
from leaking into validation.

## Why are actions normalized?

The raw X and Y action values are shifted and scaled into a range that is
easier for a neural network to learn. The saved mean and standard deviation
are used to convert predictions back to real action values later.

## Does it load everything into memory?

No. The dataset is about 46 GB, so the loader opens it lazily and reads only
the pictures and actions needed for the current batch.

