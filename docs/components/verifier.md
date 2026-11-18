# Verifier: the teacher

## What does it do?

The verifier finds an action that LeWM does not understand well yet.

For each NanoWM subgoal:

1. the IDM guesses actions for reaching it;
2. LeWM predicts what those actions will do;
3. the verifier compares LeWM's prediction with the subgoal.

```text
large difference = useful situation to practise
```

Main code: `src/latentwav/verifier.py`

## How is the score calculated?

```python
score = ((predicted_latent - subgoal_latent) ** 2).mean()
```

The verifier chooses the candidate with the largest score.

## Does it need training?

No. It is only a mathematical comparison and has no learned weights.

## What does the score mean?

A high score means LeWM's prediction is far from NanoWM's proposed future. It
does not guarantee a high PushT reward, and it does not prove that LeWM will be
wrong about the real result.

We can compare LeWM with the real result afterward for evaluation. We cannot
use that future information to choose the action, because it is unavailable
before execution. Doing so would be an Oracle experiment rather than WAV.

## Does it work?

Yes. Maximum-disagreement selection was used in both completed rollout
collections.

