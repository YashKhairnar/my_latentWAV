# Experiments

Only experiments that produced a reusable model, dataset, or training lesson
are kept here. The numbers preserve the original run order.

| Experiment | What happened | Decision |
|---|---|---|
| [02: Train the IDM](experiment-02/README.md) | Validation MSE reached `0.09098` | Use epoch 15 |
| [04: Collect 10 WAV episodes](experiment-04/README.md) | Collected 850 LeWM training sequences | Use for the first update |
| [05: Fine-tune on 10 episodes](experiment-05/README.md) | Prediction error became 701.5% worse | Reject |
| [07: Fine-tune on 50 episodes](experiment-07/README.md) | Prediction error became 24.3% worse | Reject |
| [08: Fine-tune on 205 episodes](experiment-08/README.md) | 49.77% better on WAV data, 18.00% worse on expert data | Keep as experimental |
| [09: Online control comparison](experiment-09/README.md) | Success fell from 64% to 28% | Reject fine-tuned model |

Experiment 09 uses:

```bash
modal run scripts/modal_compare_lewm_control.py --episodes 50
```

It runs the expert and fine-tuned checkpoints through the same goal-directed
CEM-MPC planner with identical starts, goals, seeds, and planning budgets.
