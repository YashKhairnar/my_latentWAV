# Experiment 05: first LeWM fine-tune

## Goal

Fine-tune the expert LeWM on the 10 WAV episodes from Experiment 04.

## Result

Both models were tested on the same 85 held-out sequences.

| Model | Prediction MSE |
|---|---:|
| Expert LeWM | `0.004782` |
| Fine-tuned LeWM | `0.038332` |

The fine-tuned model was `701.5%` worse.

## Decision

Reject the fine-tuned model. Ten narrow episodes were not enough and updating
the whole model caused severe forgetting. Keep using
`/models/checkpoints/lewm-pusht`.
