# Experiment 09: Online control comparison

## Question

Does the 205-episode fine-tuned LeWM control PushT better than the original
expert LeWM?

## Fair comparison

Both models used the same goal-directed CEM-MPC controller. Every paired trial
used the same initial state, future-frame goal, random seed, and planning
budget. Only the LeWM checkpoint changed.

```text
Trials:              50 matched pairs
Goal:                expert frame 25 environment steps in the future
Control budget:      50 environment steps
Planning horizon:    5 blocks of 5 actions
CEM:                 300 candidates, 30 iterations, 30 elites
Expert checkpoint:   /models/checkpoints/lewm-pusht
Fine-tuned checkpoint:
  /models/checkpoints/lewm-wav-20260903-205ep-v2/best
```

## Result

Lower error and distance are better.

| Online-control metric | Expert | Fine-tuned |
|---|---:|---:|
| Success rate | **64%** | 28% |
| Mean goal-state distance | **20.21** | 46.51 |
| Median goal-state distance | **17.34** | 31.34 |
| Mean block-position error | **9.11** | 24.58 |
| Mean block-angle error | **0.107** | 0.234 |

The expert had lower goal distance in 37 of 50 trials. The fine-tuned model
won 13. The expert alone succeeded in 22 trials, while the fine-tuned model
alone succeeded in 4.

## Conclusion

Reject the 205-episode model for online control. Keep the original expert LeWM
as the active model.

The important lesson is that better offline prediction did not produce better
online planning. The fine-tuned model improved prediction on held-out WAV data
in Experiment 08, but its errors compounded when CEM used it repeatedly to
choose real actions.

Do not use `final_environment_reward` as the main metric here. That reward
measures overlap with PushT's fixed green target, while this experiment asks
the controller to reach sampled future-frame goals. Use success rate and
goal-state distance.

## Executed-action distribution check

We repeated the same 50-pair comparison while saving all 2,500 executed
actions from each CEM controller. The expert reference contains 2,104,775
finite action rows from the 16,817 episodes in the IDM training split. All
actions are in the same raw relative-action units.

| Measure | Expert training | Expert LeWM CEM | Fine-tuned LeWM CEM |
|---|---:|---:|---:|
| Mean action magnitude | 0.2425 | 0.2328 | 0.2545 |
| Mean step-to-step action change | 0.1058 | 0.2490 | 0.2651 |
| Actions outside expert 1st–99th percentile rectangle | 3.91% | 2.56% | 4.40% |
| 2-D action-histogram JS divergence from expert | — | 0.0397 bits | 0.0347 bits |

The online success rates again measured 64% and 28%. Both CEM controllers
produce much larger step-to-step action changes than the demonstrations.
However, the fine-tuned controller's *marginal* 2-D action distribution is
slightly closer to the expert histogram by JS divergence. Marginal action
shift alone therefore does not explain its control regression. The remaining
question is whether five-action blocks and state-conditioned actions differ,
or whether the fine-tuned LeWM misranks candidate plans.

![Shared-scale expert and CEM action densities](../../../artifacts/lewm_control_action_comparison_50_action_distributions.svg)

The full report, including each executed action and per-distribution summary,
is at `artifacts/lewm_control_action_comparison_50.json`.

## Files

- Results: `artifacts/lewm_control_comparison.json`
- Paired GIFs: `artifacts/lewm_control_comparison_pair_000.gif` through
  `artifacts/lewm_control_comparison_pair_004.gif`
- Evaluation: `scripts/modal_compare_lewm_control.py`
- Planner: `src/latentwav/control.py`
- Action analysis: `src/latentwav/action_distribution.py`
