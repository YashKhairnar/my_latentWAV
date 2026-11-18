# LatentWAV

## The project in one sentence

LatentWAV teaches a computer to understand what happens when a robot takes an
action in a simple game called **PushT**.

In PushT, a blue circle acts like the robot's hand. It must move a gray block
shaped like the letter **T** toward a green goal.

For a more technical explanation, see
[`docs/COMPONENTS.md`](docs/COMPONENTS.md).
Individual component pages are indexed in
[`docs/README.md`](docs/README.md).
Completed, failed, and ongoing runs are recorded in
[`docs/experiments/README.md`](docs/experiments/README.md).

## A simple story

Imagine a robot with three little helpers:

1. **The Dreamer (NanoWM)** looks at the current picture and dreams about
   possible pictures that could happen next.
2. **The Action Guesser (IDM)** looks at the current picture and a dreamed
   picture. It guesses which five small moves could connect them.
3. **The Future Guesser (LeWM)** looks at the pictures and moves. It guesses
   what the result should be.

We compare the Future Guesser's answer with the Dreamer's picture. If they are
very different, we found something LeWM does not understand well yet. The robot
tries that action in the real PushT game. We save what really happened and use
it to teach a new LeWM.

```text
current pictures
       |
       v
NanoWM dreams several possible future pictures
       |
       v
IDM guesses five moves for each picture
       |
       v
LeWM guesses what each set of moves will do
       |
       v
Choose the action LeWM understands the least
       |
       v
Try it in PushT and save what really happened
       |
       v
Train a new LeWM using the saved experience
```

## Why choose the biggest mistake?

Suppose a child knows `1 + 1`, but does not know `7 + 8`. Giving the child more
`1 + 1` questions does not teach much. It is better to practice the difficult
question.

LatentWAV does the same thing. It chooses the action with the biggest
**disagreement**:

```text
disagreement = difference between LeWM's guess and NanoWM's subgoal
```

A large disagreement means, "LeWM should practice this kind of event."

This disagreement is the WAV paper's verification score. It is computed before
execution, so the system can choose which action is worth trying. We may compare
LeWM with the real resulting frame afterward for evaluation, but we do not use
that true error to replace WAV's selection rule or discard collected samples.

The PushT reward does not need to increase while collecting data. At this
stage, the goal is to find useful lessons for LeWM, not immediately solve the
game.

## The four main parts

### 1. NanoWM: the Dreamer

NanoWM is a pretrained video model. It receives one to three recent PushT
pictures and creates possible future pictures called **subgoals**.

```text
recent pictures -> NanoWM -> possible future pictures
```

We did not train NanoWM. We downloaded its weights and wrote code that loads
the model, converts pictures into hidden numbers, generates future states, and
turns them back into `224 x 224` RGB pictures.

Important note: our current checkpoint was trained with action conditioning,
but we call it without actions when creating subgoals. It works for building
and testing the pipeline. An action-free checkpoint would be better for final
experiments.

Main file: `src/latentwav/models/nanowm.py`

### 2. IDM: the Action Guesser

IDM means **Inverse Dynamics Model**. It receives the current picture and one
possible future picture. It returns five two-number actions:

```text
(current picture, future picture) -> action block shaped (5, 2)
```

The IDM has a picture encoder, a small Transformer, and an action head. We
wrote and trained it from scratch on the full PushT expert dataset using a
Modal H100 GPU.

Best checkpoint on Modal:

```text
/models/checkpoints/idm/idm_pusht_best.pt
```

Main files:

- `src/latentwav/models/idm.py`
- `scripts/train_idm.py`
- `scripts/modal_train_idm.py`

### 3. LeWM: the Future Guesser

LeWM means **Latent World Model**. It does not predict a full RGB picture.
First, its ViT image encoder turns a picture into 192 hidden numbers called a
**latent**. A latent is like a small secret description of a picture.

LeWM receives up to three picture latents and three five-move action blocks. It
predicts the next picture latent:

```text
3 picture latents + 3 action blocks -> next picture latent
```

We made our code compatible with the public LeWM checkpoint. It loads with
every parameter matching correctly.

For the main experiment, we use the public LeWM trained on expert PushT data as
our starting model. We keep that original checkpoint safe and train a separate
copy on the new high-disagreement experience.

We also wrote and tested an optional from-scratch trainer. It creates a
brand-new LeWM from random weights without loading pretrained LeWM weights.

The small overfit test proved that the trainer works:

```text
initial prediction error: 0.06681
best prediction error:    0.00494
error reduction:          92.6%
```

This was only a plumbing test. It proved that our trainer works, but it is not
the main experiment path anymore.

Main files:

- `src/latentwav/models/lewm_core.py`
- `src/latentwav/models/world_model.py`
- `scripts/train_lewm.py`
- `scripts/modal_train_lewm.py`

### 4. The Verifier: the Teacher

The verifier gives every candidate a disagreement score:

```python
score = ((predicted_latent - subgoal_latent) ** 2).mean()
```

It selects the candidate with the **largest** score. The verifier is only a
math rule. It has no neural network and needs no training.

Main file: `src/latentwav/verifier.py`

## The PushT data

We downloaded the official LeWM PushT expert dataset. It is about 46 GB and
contains:

```text
2,336,736 rows
18,685 episodes
```

Each row can store a picture, an action, the robot position, the simulator
state, and episode information.

Our dataset code checks the file, calculates action statistics, splits whole
episodes into training and validation groups, and never creates a transition
across an episode boundary. It loads frame pairs for IDM and four-frame
sequences for LeWM.

Dataset on Modal:

```text
/models/datasets/lewm-pusht/pusht_expert_train.h5
```

Main file: `src/latentwav/data.py`

## The online policy

The policy joins NanoWM, IDM, LeWM, and the verifier. At each planning moment,
it:

1. remembers up to three older pictures;
2. asks NanoWM for several subgoals;
3. asks IDM for one action block per subgoal;
4. asks LeWM to predict each action block's result;
5. selects the candidate with the biggest disagreement;
6. returns its five actions one at a time;
7. plans again after all five actions are used.

Its history stays aligned like this:

```text
frames:         frame 0 -------- frame 5 -------- frame 10
actions:              actions 0-4      actions 5-9
```

Main file: `src/latentwav/policy.py`

## Collecting new experience

We wrote a real PushT episode loop. For every step it saves the picture, state,
executed action, reward, disagreement score, selected subgoal, and macro-step.
The saved HDF5 file works with our training data loader.

We successfully collected a real 50-step episode:

```text
environment steps:             50
planning moments:              10
subgoals per planning moment:   4
usable frameskip-5 transitions: 45
usable four-frame LeWM sequences: 35
mean disagreement:             0.1569
```

Rollout on Modal:

```text
/models/rollouts/latentwav_20260812T224310Z.h5
```

Local GIFs and plots are stored under `artifacts/rollouts/`.

We then collected a larger H100 batch for LeWM training:

```text
episodes:                         10
steps per episode:               100
total rows:                    1,000
IDM-style transitions:           950
four-frame LeWM sequences:        850
```

The larger rollout is stored at:

```text
/models/rollouts/latentwav_20260812T235844Z.h5
```

Main files:

- `src/latentwav/loop.py`
- `src/latentwav/visualization.py`
- `scripts/modal_collect_pusht.py`

## What we train

| Part | How it starts | What we do |
|---|---|---|
| NanoWM | Pretrained weights | Keep frozen |
| Stable Diffusion VAE | Pretrained weights | Keep frozen |
| IDM | Random weights | Train from scratch |
| LeWM | Expert-trained checkpoint | Train a separate copy on collected data |
| Verifier | A math rule | No training |

## What already works

- NanoWM creates PushT-like subgoal pictures.
- The dataset loader creates correct frames and action blocks.
- IDM was trained on the full expert dataset.
- The public LeWM checkpoint loads correctly.
- LeWM predicts a 192-number future latent.
- The verifier selects the largest disagreement.
- The policy replans after exactly five actions.
- Frame and action histories stay aligned.
- The full policy generated and executed actions in real PushT.
- A 50-step rollout was saved and loaded back successfully.
- The rollout produced 45 valid training transitions.
- The optional from-scratch LeWM trainer can overfit the small rollout.
- Updated LeWM checkpoints load in the normal inference code, but the first
  updated checkpoint was rejected by validation.
- A second, versioned LeWM candidate was trained on 50 WAV episodes with the
  visual representation frozen and a 10x smaller learning rate. It reduced
  degradation substantially, but was still 24.3% worse than the expert model
  and was rejected.
- A 205-episode deterministic fine-tune produced the first candidate to beat
  the expert LeWM on held-out WAV data: prediction MSE improved by 49.77%.
  It remained experimental because expert-data MSE regressed by 18.00%.

## What the first fine-tuning experiment taught us

We trained a copy of the expert LeWM for 500 steps on the 10 collected
episodes. We then evaluated both models on the same 85 untouched sequences:

```text
expert baseline prediction MSE: 0.00478
updated candidate prediction MSE: 0.03833
change: 701.5% worse
```

The updated checkpoint was rejected. The project still uses the original
expert checkpoint at `/models/checkpoints/lewm-pusht`.

This result does not invalidate WAV's maximum-disagreement rule. The paper uses
that rule as a proxy for unknown future error and reports that its ranking is
correlated with true error. Our run instead shows that updating every parameter
of an already strong LeWM for 500 steps on only 10 narrow episodes is unstable.

## What the conservative fine-tuning experiment taught us

We collected 50 episodes and trained a versioned candidate for 200 steps on
3,825 sequences, reserving 425 sequences from five complete episodes for
validation. The visual encoder and projector were frozen, and the learning
rate was reduced to `5e-6`.

```text
expert baseline prediction MSE: 0.04376
conservative candidate MSE:     0.05439
change:                         24.3% worse
```

The candidate at
`/models/checkpoints/lewm-wav-20260902-conservative/best` was rejected. The
active model remains `/models/checkpoints/lewm-pusht`. The next update should
mix expert warm-up examples with WAV data instead of training only on the new
rollouts.

## What the 205-episode update taught us

We collected 205 episodes, yielding 15,725 training sequences and 1,700
sequences from 20 held-out episodes. A deterministic 500-step update froze the
visual encoder and projector and trained the action-conditioned components at
a `1e-6` learning rate.

```text
held-out WAV MSE:
  expert baseline: 0.03510
  new candidate:   0.01763  (49.77% better)

expert-data MSE:
  expert baseline: 0.00776
  new candidate:   0.00916  (18.00% worse)
```

The candidate is preserved at
`/models/checkpoints/lewm-wav-20260903-205ep-v2/best`, but it has not replaced
the active expert checkpoint. The next update should use mixed expert/WAV
replay and must pass both evaluations.

# Online control comparison

The existing LatentWAV policy maximizes disagreement to collect useful data.
It is not a goal-directed controller, so its PushT reward does not tell us
whether a fine-tuned LeWM plans better.

The paired control script now compares the expert and 205-episode LeWM with the
same goal-directed CEM-MPC planner. It keeps starts, future-frame goals, seeds,
and planning budget identical and changes only the LeWM checkpoint.

Run a 10-pair smoke test:

```bash
modal run scripts/modal_compare_lewm_control.py
```

Run the full 50-pair comparison:

```bash
modal run scripts/modal_compare_lewm_control.py --episodes 50
```

The local report is written to
`artifacts/lewm_control_comparison.json`. Smaller goal-state distance and higher
success rate are better. Do not interpret LeWM's latent planning cost across
the two checkpoints as a shared metric because each checkpoint defines its own
latent space.

The first matched pair is also saved as
`artifacts/lewm_control_comparison_pair_000.gif`. It shows the expert rollout,
fine-tuned rollout, and shared goal side by side. Use `--video-episodes N` to
save more paired GIFs, or `--video-episodes 0` to disable them.

The completed 50-pair comparison found:

| Online-control metric | Expert | Fine-tuned |
|---|---:|---:|
| Success rate | **64%** | 28% |
| Mean goal-state distance | **20.21** | 46.51 |
| Median goal-state distance | **17.34** | 31.34 |

The expert had lower goal distance in 37 of 50 matched trials. This rejects
the 205-episode candidate for online control: its better offline WAV prediction
error did not translate into better planning. See
[`docs/experiments/experiment-09`](docs/experiments/experiment-09/README.md).

# What still needs to be tested

To stay faithful to the paper, the next experiment should:

1. keep maximum subgoal-to-LeWM disagreement as the acquisition score;
2. execute the selected action block and add the transition to the action data;
3. update both LeWM and the IDM over repeated exploration rounds;
4. use an action-free subgoal generator and a sparse IDM;
5. evaluate true prediction error only as a held-out metric;
6. compare WAV against random and, optionally, an Oracle selector under the
   same interaction budget.

## Project folder map

```text
my_latentWAV/
├── README.md                         # this simple explanation
├── docs/COMPONENTS.md                # detailed technical explanation
├── checkpoints/nanowm-pusht/        # NanoWM weights and config
├── nano-world-model/                # NanoWM source code
├── src/latentwav/
│   ├── data.py                      # dataset loaders
│   ├── control.py                   # goal-conditioned CEM planner
│   ├── loop.py                      # collect and save episodes
│   ├── policy.py                    # the five-action policy
│   ├── verifier.py                  # disagreement scoring
│   ├── visualization.py             # rollout GIFs and plots
│   └── models/
│       ├── nanowm.py                # the Dreamer
│       ├── idm.py                   # the Action Guesser
│       ├── lewm_core.py             # LeWM building blocks
│       └── world_model.py           # LeWM loading and inference
└── scripts/
    ├── modal_setup_pusht_dataset.py # download the expert dataset
    ├── modal_setup_lewm.py          # download the expert LeWM
    ├── modal_collect_pusht.py       # collect PushT experience
    ├── compare_lewm.py              # comparison implementation
    ├── modal_compare_lewm.py        # compare models on Modal
    ├── modal_compare_lewm_control.py # compare online control on Modal
    ├── train_idm.py                 # train IDM
    ├── modal_train_idm.py           # train IDM on Modal
    ├── train_lewm.py                # train or update LeWM
    ├── modal_train_lewm.py          # train LeWM on Modal
```

## Useful commands

Collect one 50-step PushT episode:

```bash
.venv-nanowm/bin/modal run scripts/modal_collect_pusht.py \
  --episodes 1 \
  --max-steps 50 \
  --num-subgoals 4 \
  --sampling-steps 50
```

Fine-tune a copy of the expert LeWM on the new 10-episode dataset:

```bash
.venv-nanowm/bin/modal run scripts/modal_train_lewm.py \
  --full \
  --rollout-path /models/rollouts/latentwav_20260812T235844Z.h5 \
  --max-steps 500
```

This 500-step experiment was completed, evaluated, and rejected because it
performed worse than the baseline. Do not use its checkpoint as the policy's
world model.

After training finishes, compare the original and updated models on the
untouched collected episode:

```bash
.venv-nanowm/bin/modal run scripts/modal_compare_lewm.py
```

Then check for forgetting on held-out expert episodes:

```bash
.venv-nanowm/bin/modal run scripts/modal_compare_lewm.py \
  --dataset /models/datasets/lewm-pusht/pusht_expert_train.h5 \
  --max-sequences 512
```

The important output is `mse_change_percent`. A negative number means the
updated model makes a smaller prediction error than the original model.

## What remains

1. Add repeated WAV exploration rounds instead of a single collection/update.
2. Retrain the IDM together with LeWM after each round.
3. Add the paper's learnable sparsity mechanism to the IDM.
4. Use an action-free NanoWM checkpoint as the transition prior.
5. Preserve warm-up data when adding newly acquired transitions.
6. Validate true LeWM prediction error on a separate held-out dataset.
7. Compare WAV with random acquisition under an equal interaction budget.
8. Keep the original expert LeWM as a safety baseline during development.

## The big idea to remember

LatentWAV is not only asking, "Which action gets the biggest reward now?"

It is asking:

> Which plausible and reachable subgoal disagrees most with what my world model
> predicts for its inferred action?

That real experience becomes the lesson used to train a better LeWM.
