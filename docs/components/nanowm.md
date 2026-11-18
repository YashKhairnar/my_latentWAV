# NanoWM: the dreamer

## What does it do?

NanoWM looks at the current PushT picture and dreams several pictures that
could happen next. We call these future pictures **subgoals**.

```text
recent pictures -> NanoWM -> possible future pictures
```

The rest of LatentWAV uses these pictures to explore different actions.

Main code: `src/latentwav/models/nanowm.py`

## Did we train it?

No. We downloaded a pretrained PushT checkpoint and keep it frozen. Its image
encoder and decoder are also frozen.

Local checkpoint:

```text
checkpoints/nanowm-pusht/
```

## What goes in and comes out?

NanoWM receives one or more recent RGB pictures. It returns as many candidate
future pictures as we request. The final pictures are `224 x 224` RGB images.

Internally, it resizes pictures, compresses them into smaller hidden values,
generates future hidden values, and turns them back into pictures.

## Does it work?

Yes. It generates PushT-like future pictures and has been used successfully in
real PushT rollout collection.

## Current limitation

The checkpoint was trained using actions as extra information. LatentWAV does
not have those actions yet when it asks NanoWM for subgoals, so it calls the
model with no actions.

The code works because the action path is skipped when no action is supplied.
However, this is different from how the checkpoint was trained. Changing
`use_action` in its YAML would not fix that and would break strict weight
loading. A truly action-free model must be trained with `use_action: false`.

