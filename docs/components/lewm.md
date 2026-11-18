# LeWM: the future guesser

## What does it do?

LeWM predicts what the PushT scene will look like after the robot performs an
action block.

It does not predict a full picture directly. It predicts a short hidden
description of the picture called a **latent**.

```text
recent pictures + recent actions + next actions -> future picture latent
```

Main code:

- `src/latentwav/models/lewm_core.py`
- `src/latentwav/models/world_model.py`
- `scripts/train_lewm.py`

## Did we train it?

We use a public LeWM that was already trained on expert PushT data. That model
is our safe baseline. We never overwrite it; experiments train separate copies.

Baseline checkpoint:

```text
/models/checkpoints/lewm-pusht
```

## What worked?

The checkpoint loads with every weight matching correctly. LeWM produces the
expected 192-number future latent and works inside the online policy.

A small test trained a new LeWM until it memorized a tiny dataset. Its error
fell from `0.06681` to `0.00494`. This proved that the training code works.

## What failed?

We fine-tuned a copy of the expert model for 500 steps using only 10 newly
collected episodes.

```text
Original model error: 0.00478
Updated model error:  0.03833
Result:               701.5% worse
```

We rejected the updated model and kept the original expert model.

## Why did it fail?

The original model was already strong. Training all of its weights on only 10
similar episodes pushed it away from what it had learned before.

Future attempts should mix old and new data, make smaller updates, validate on
untouched episodes, and keep a new model only when it improves.

