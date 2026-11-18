# Online policy: the coordinator

## What does it do?

The online policy connects NanoWM, the IDM, LeWM, and the verifier. It decides
which actions the PushT agent should execute.

Main code: `src/latentwav/policy.py`

## What happens during planning?

1. NanoWM dreams several possible future pictures.
2. The IDM guesses five actions for each picture.
3. LeWM predicts what each five-action block will do.
4. The verifier finds the largest disagreement.
5. The policy executes those five actions one at a time.
6. After five actions, it plans again.

```text
dream futures -> guess actions -> predict results -> choose -> execute
```

## Why does it remember old pictures and actions?

LeWM needs a short history to understand movement. The policy keeps pictures
and actions aligned at five-step boundaries:

```text
picture 0 ---- picture 5 ---- picture 10
         actions 0–4   actions 5–9
```

## Does it work?

Yes. It has executed real PushT episodes, replanned after exactly five actions,
and kept its history aligned.

## What is missing?

The policy handles one environment. We still need one automated program that
repeats collection, training, validation, and model acceptance over many WAV
rounds.

