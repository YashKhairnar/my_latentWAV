# LatentWAV Component Documentation

This document explains the LatentWAV components implemented so far, how data
moves through them, and what remains to be built.

Detailed contracts and results are maintained in separate pages:

- [NanoWM](components/nanowm.md)
- [Inverse Dynamics Model](components/idm.md)
- [Latent World Model](components/lewm.md)
- [Verifier](components/verifier.md)
- [Data pipeline](components/data-pipeline.md)
- [Online policy](components/online-policy.md)
- [Collection and evaluation](components/collection-evaluation.md)

## 1. Project objective

LatentWAV is a self-improving world-model system for the PushT manipulation
task. It combines a generative video model, an inverse dynamics model, and a
latent forward world model.

At a high level:

```text
current visual context
        |
        v
NanoWM proposes future subgoals
        |
        v
IDM predicts actions that may reach them
        |
        v
LeWM predicts the outcomes of those actions
        |
        v
Verifier finds large LeWM-to-subgoal disagreements
        |
        v
Execute, collect data, and improve LeWM
```

## 2. Current implementation status

| Component | Status | Main file |
|---|---|---|
| NanoWM subgoal wrapper | Implemented and smoke-tested | `src/latentwav/models/nanowm.py` |
| PushT dataset download | Implemented and completed | `scripts/modal_setup_pusht_dataset.py` |
| PushT dataset inspection | Implemented | `src/latentwav/data.py` |
| Action normalization | Implemented | `src/latentwav/data.py` |
| Episode train/validation split | Implemented | `src/latentwav/data.py` |
| Valid transition indexing | Implemented | `src/latentwav/data.py` |
| PyTorch transition dataset | Implemented and tested | `src/latentwav/data.py` |
| Pixel IDM architecture | Implemented, trained, and tested | `src/latentwav/models/idm.py` |
| IDM training program | Implemented and completed | `scripts/train_idm.py` |
| LeWM wrapper | Implemented and smoke-tested | `src/latentwav/models/world_model.py` |
| Action verifier | Implemented and smoke-tested | `src/latentwav/verifier.py` |
| Single-environment online policy | Implemented and end-to-end tested | `src/latentwav/policy.py` |
| PushT rollout collection/recording | Implemented and smoke-tested | `src/latentwav/loop.py` |
| LeWM training/fine-tuning | Implemented and exercised | `scripts/train_lewm.py` |
| Baseline/candidate comparison | Implemented and exercised | `scripts/compare_lewm.py` |
| Iterative collection/training loop | Not implemented yet | `src/latentwav/loop.py` |
| Sparse IDM mechanism | Not implemented yet | `src/latentwav/models/idm.py` |

## 3. What is trained

Not every model is trained from scratch.

| Component | Starting point | Project behavior |
|---|---|---|
| NanoWM | Pretrained PushT checkpoint | Frozen during initial integration |
| SD-VAE | Pretrained Stable Diffusion VAE | Frozen |
| IDM | Random initialization | Trained from scratch on PushT transitions |
| LeWM | Expert-trained checkpoint | Keep the expert model active; train and validate separate experimental copies |
| Verifier | Mathematical scoring rule | Has no trainable parameters |

The current NanoWM checkpoint is action-conditioned. It works for engineering
tests, but the final paper-style subgoal generator should use an action-free
checkpoint. Otherwise NanoWM expects an action before the IDM has predicted it.

## 4. NanoWM subgoal generator

### Purpose

NanoWM proposes one or more plausible future images from recent context:

```text
context frames -> NanoWM -> candidate future frames
```

The wrapper is implemented by `NanoWMSubgoalGenerator` in
`src/latentwav/models/nanowm.py`.

### Model configuration

The downloaded checkpoint uses approximately the following configuration:

```text
Architecture:        NanoWM-B/2
Image size:          256 x 256 internally
Latent shape:        4 x 32 x 32 per frame
Number of frames:    4
Context frames:      1
Default sample steps: 250
Action conditioning: enabled
```

### Input contract

The public wrapper accepts context frames as NumPy arrays:

```text
(L, H, W, 3), uint8, values 0 to 255
```

`L` is the number of context frames and must be smaller than the checkpoint's
total frame count.

### Internal preprocessing

The wrapper:

1. Checks the context shape and type.
2. Resizes frames to `256 x 256`.
3. Changes layout from `LHWC` to `LCHW`.
4. Converts pixels from `[0, 255]` to `[-1, 1]`.
5. Encodes frames with the frozen SD-VAE.

```text
(L,H,W,3) uint8
       |
       v
(L,3,256,256) float in [-1,1]
       |
       v
(L,4,32,32) VAE latents
```

### Sampling

For `K` requested subgoals, the encoded context is duplicated `K` times. The
diffusion sampler generates one future latent per candidate.

```text
context latents -> K diffusion samples -> K future latents
```

The SD-VAE decoder converts them back to RGB images, which are resized to the
project output size:

```text
(K,4,32,32) -> (K,224,224,3) uint8
```

### Important limitation

When the synthetic test image was supplied, the model generated PushT-looking
frames. This is expected: the checkpoint learned the PushT visual distribution
and pulls unfamiliar inputs toward that distribution.

The real PushT context test is a more meaningful validation than a synthetic
out-of-distribution image.

## 5. PushT dataset

### Storage

The official LeWM PushT dataset is stored in the `latentwav-models` Modal
volume:

```text
/models/datasets/lewm-pusht/pusht_expert_train.h5
```

The compressed archive is approximately 13.14 GB. The decompressed HDF5 file
is approximately 46.30 GB.

### Dataset summary

```text
Episodes:          18,685
Total rows:        2,336,736
Minimum episode:   49 steps
Maximum episode:   246 steps
Average episode:   125.06 steps
```

### HDF5 schema

```text
action       (2,336,736, 2)            float32
pixels       (2,336,736, 224, 224, 3)  uint8
proprio      (2,336,736, 4)            float32
state        (2,336,736, 7)            float32
episode_idx  (2,336,736)               int64
step_idx     (2,336,736)               int64
ep_len       (18,685)                  int32
ep_offset    (18,685)                  int64
```

All episodes are concatenated along the first dimension:

```text
episode 0 rows | episode 1 rows | episode 2 rows | ...
```

For episode `i`:

```python
start = ep_offset[i]
length = ep_len[i]
end = start + length
```

## 6. Data pipeline

The data pipeline is implemented in `src/latentwav/data.py`.

### 6.1 Dataset inspection

`inspect_dataset(path)`:

- Verifies that the path exists.
- Opens the HDF5 file read-only.
- Validates required keys.
- Prints every dataset's shape and type.
- Reports episode-length statistics.
- Reads the first frame and action.

HDF5 access is lazy. This returns a handle and does not load all images:

```python
pixels = dataset["pixels"]
```

This reads only one frame:

```python
frame = dataset["pixels"][0]
```

### 6.2 Action statistics

Every PushT action has two values:

```text
action = [delta_x, delta_y]
```

`compute_action_stats(path)` removes invalid rows containing `NaN`, then
computes a mean and standard deviation for each action dimension.

Training normalization:

```python
normalized_action = (raw_action - mean) / std
```

Inference de-normalization:

```python
raw_action = normalized_prediction * std + mean
```

Small standard deviations are replaced with `1.0` to avoid division by zero.

### 6.3 Episode-level splitting

`split_episode_ids(num_episodes, val_fraction, seed)` creates reproducible,
disjoint training and validation episode sets.

The split is performed by complete episode, not individual transition. This
prevents nearly identical neighboring frames from appearing in both sets.

```text
90% complete episodes -> training
10% complete episodes -> validation
```

### 6.4 Transition indexing

The IDM learns an action block connecting two frames. With `frameskip=5`:

```text
frame[t] + frame[t+5] -> actions[t:t+5]
```

`build_transition_indices()` finds valid starting rows without crossing an
episode boundary.

For an episode beginning at row 100 with length 8:

```text
100 -> 105  valid
101 -> 106  valid
102 -> 107  valid
103 -> 108  invalid: row 108 is outside the episode
```

The function can accept selected episode IDs, allowing separate train and
validation transition arrays.

### 6.5 PyTorch transition dataset

`PushTTransitionDataset` turns one valid row into an IDM training example.

For `row=100` and `frameskip=5`:

```python
current_frame = pixels[100]
future_frame = pixels[105]
actions = action[100:105]
```

The frames change from HDF5 layout to PyTorch layout:

```text
(224,224,3) HWC -> (3,224,224) CHW
```

Five two-dimensional actions are normalized and flattened:

```text
(5,2) -> (10,)
```

Each item returns:

```text
current frame: (3,224,224) uint8
future frame:  (3,224,224) uint8
action block:  (10,)       float32
```

The HDF5 file is opened lazily. Each PyTorch DataLoader worker receives its own
file handle, and each item reads only two frames and five actions.

### Complete data flow

```text
PushT HDF5
    |
    +--> inspect and validate schema
    |
    +--> calculate action mean/std
    |
    +--> split complete episodes
    |
    +--> calculate valid transition rows
                     |
                     v
        PushTTransitionDataset[index]
                     |
          +----------+----------+
          |          |          |
          v          v          v
       frame[t]  frame[t+5]  actions[t:t+5]
          |          |          |
       HWC->CHW   HWC->CHW   normalize + flatten
          |          |          |
          v          v          v
    (3,224,224) (3,224,224)    (10,)
          +----------+----------+
                     |
                     v
                 IDM training
```

## 7. Inverse dynamics model

The IDM is implemented in `src/latentwav/models/idm.py`.

### Purpose

During supervised training:

```text
(real current frame, real future frame) -> real action block
```

During LatentWAV inference:

```text
(current frame, NanoWM subgoal) -> predicted action block
```

### 7.1 BasicBlock

`BasicBlock` is a residual convolutional block:

```text
input -> Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm --+
  |                                                               |
  +---------------- shortcut -------------------------------------+
                                                                  |
                                                                ReLU
```

If channels or spatial resolution change, a `1 x 1` convolution projects the
shortcut to the correct shape. Otherwise, the shortcut is an identity mapping.

### 7.2 PixelEncoder

The shared `PixelEncoder` converts an image into a 256-dimensional vector:

```text
(B,3,224,224)
      |
      v
7x7 convolution + max pooling
      |
      v
residual blocks: 64 -> 128 -> 256 channels
      |
      v
global average pooling
      |
      v
linear projection
      |
      v
(B,256)
```

The same encoder processes current and future frames so their features occupy
the same representation space.

### 7.3 Transition representation

The IDM creates three tokens:

```text
current feature
future feature - current feature
future feature
```

For batch size `B` and embedding dimension 256:

```text
tokens: (B,3,256)
```

Learned type embeddings tell the model which token is current, change, or
future.

### 7.4 Transformer and action head

A three-layer Transformer allows the three tokens to exchange information.
The output tokens are averaged and passed through a regression head:

```text
(B,3,256) -> Transformer -> mean over tokens -> MLP -> (B,10)
```

For PushT:

```text
raw action dimension = 2
frameskip = 5
action block dimension = 2 * 5 = 10
```

### Pixel preprocessing

The dataset returns `uint8` images. The IDM converts them to `[0,1]` and applies
ImageNet normalization internally:

```python
image = image.float() / 255.0
image = (image - pixel_mean) / pixel_std
```

### Model contract

```text
current:    (B,3,224,224) uint8
future:     (B,3,224,224) uint8
prediction: (B,10)        float32
```

The default model has approximately 4.05 million parameters. Forward,
backward, residual-shape, and configuration-reconstruction tests passed.

## 8. Full inference and online collection flow

### Step 1: Observe

PushT returns the current RGB frame. The policy maintains a short frame and
action history for NanoWM and LeWM.

### Step 2: Propose subgoals

NanoWM samples `K` plausible future images:

```text
context -> NanoWM -> K subgoals
```

### Step 3: Infer candidate actions

The IDM processes the current frame with every subgoal:

```text
(current, subgoal[k]) -> normalized action block[k]
```

Each prediction is de-normalized, reshaped to `(5,2)`, and clipped to the
environment's valid action range.

### Step 4: Predict candidate outcomes

LeWM predicts the future latent caused by each action block:

```text
(latent history, action history + candidate action) -> predicted latent[k]
```

LeWM's image encoder also converts every NanoWM subgoal into the same latent
space.

### Step 5: Verify

The verifier computes:

```python
score[k] = ((predicted_latent[k] - subgoal_latent[k]) ** 2).mean()
```

The candidate with the largest disagreement is selected. This identifies a
large gap between LeWM's predicted outcome and NanoWM's proposed subgoal. It is
an acquisition heuristic; it does not prove that LeWM will predict the real
environment outcome poorly.

### Step 6: Execute and record

The five selected actions are executed one at a time. Frames, actions, states,
and episode metadata are written to a new HDF5 rollout dataset.

### Step 7: Improve LeWM

The expert-trained LeWM checkpoint is copied and updated using the accumulated
action-labeled experience. The paper's Algorithm 1 also updates the IDM from
that accumulated data. NanoWM remains fixed as the action-free transition
prior. Our current one-round experiment kept the IDM frozen, so that part still
needs to be implemented for paper fidelity.

```text
frame history
    |
    v
NanoWM: K subgoals
    |
    v
IDM: K action blocks
    |
    v
LeWM: K predicted outcome latents
    |                              NanoWM subgoals
    |                                    |
    |                              LeWM image encoder
    |                                    |
    +----------------+-------------------+
                     |
                     v
              K disagreement scores
                     |
                     v
             select maximum score
                     |
                     v
             execute action block
                     |
                     v
              record transition
                     |
                     v
          update a copy of LeWM
```

The maximum-disagreement rule is the WAV acquisition rule and does not
necessarily choose the action with the highest task reward. True error against
the actual successor frame is useful for held-out evaluation and for measuring
the verifier's ranking quality. It must not replace the WAV selection rule;
selecting directly by true successor error is the paper's Oracle baseline.

## 9. IDM training

The training CLI is implemented in `scripts/train_idm.py`.

It performs the following sequence:

1. Validates command-line settings and resolves CUDA, MPS, or CPU.
2. Computes action normalization statistics.
3. Splits complete episodes into training and validation sets.
4. Builds boundary-safe transition indices for both sets.
5. Creates lazy HDF5 PyTorch DataLoaders.
6. Trains the IDM using mean-squared error on normalized action blocks.
7. Clips gradients and applies AdamW optimization.
8. Evaluates total and per-dimension validation MSE.
9. Saves both the best and most recent checkpoints.

Every checkpoint contains:

```text
model weights
optimizer state
model configuration
action mean and standard deviation
training arguments
epoch number
training MSE
validation metrics
```

The transition-limit arguments support inexpensive smoke and overfit tests:

```bash
python scripts/train_idm.py \
  --dataset /path/to/pusht_expert_train.h5 \
  --output /path/to/idm_smoke.pt \
  --epochs 3 \
  --batch-size 16 \
  --num-workers 2 \
  --max-train-transitions 128 \
  --max-val-transitions 128
```

The best checkpoint is written to the exact `--output` path. The most recent
checkpoint adds `_latest` before the file extension.

## 10. Modal commands

### Download the dataset

```bash
.venv-nanowm/bin/modal run scripts/modal_setup_pusht_dataset.py
```

This script downloads, decompresses, validates, and stores the PushT dataset in
the persistent Modal volume.

### Install the expert LeWM checkpoint

```bash
.venv-nanowm/bin/modal run scripts/modal_setup_lewm.py
```

### Collect new experience

```bash
.venv-nanowm/bin/modal run scripts/modal_collect_pusht.py \
  --episodes 10 \
  --max-steps 100 \
  --num-subgoals 4 \
  --sampling-steps 50
```

### Train a copy of the expert checkpoint

```bash
.venv-nanowm/bin/modal run scripts/modal_train_lewm.py \
  --full \
  --rollout-path /models/rollouts/latentwav_20260812T235844Z.h5 \
  --max-steps 500
```

### Compare the baseline and candidate

```bash
.venv-nanowm/bin/modal run scripts/modal_compare_lewm.py
```

## 11. Completed online experiment

### Acquisition dataset

The expert LeWM, trained IDM, NanoWM, and maximum-disagreement verifier were
used to collect:

```text
10 episodes
100 environment steps per episode
1,000 total rows
950 frame-pair transitions
850 four-frame LeWM sequences
```

The dataset is stored at:

```text
/models/rollouts/latentwav_20260812T235844Z.h5
```

### First checkpoint update

A copy of the expert LeWM was trained for 500 steps. Nine episodes supplied
765 training sequences; one untouched episode supplied 85 validation
sequences. Expert action statistics were used for normalization.

Both checkpoints were then evaluated on exactly the same 85 sequences:

| Metric | Expert baseline | Updated candidate |
|---|---:|---:|
| Prediction MSE | 0.004782 | 0.038332 |
| Normalized MSE | 0.004020 | 0.034054 |
| Cosine error | 0.001948 | 0.017175 |
| Target latent standard deviation | 1.090644 | 1.060958 |
| Prediction latent standard deviation | 1.080561 | 1.051459 |

The candidate prediction MSE was `701.5%` worse. The candidate checkpoint was
therefore rejected, and `/models/checkpoints/lewm-pusht` remains the active
baseline.

### Conservative checkpoint update

A later run used 50 WAV episodes, froze the LeWM visual encoder and projector,
reduced the learning rate to `5e-6`, and trained a versioned candidate for 200
steps. Forty-five episodes supplied 3,825 training sequences and five complete
episodes supplied 425 validation sequences.

| Metric | Expert baseline | Conservative candidate |
|---|---:|---:|
| Prediction MSE | 0.043756 | 0.054387 |
| Normalized MSE | 0.044767 | 0.056258 |
| Cosine error | 0.022943 | 0.025638 |

The conservative candidate was `24.3%` worse and was also rejected. Freezing
the representation limited the damage relative to the first update, but WAV
data alone did not improve the expert model. The next candidate should retain
expert warm-up data while adding WAV transitions.

### 205-episode checkpoint update

Scaling collection to 205 episodes yielded 15,725 training sequences and
1,700 sequences from 20 held-out episodes. With the visual representation
frozen, dropout disabled, and a `1e-6` learning rate, a 500-step update produced
the first improvement over the expert baseline on held-out WAV data.

| Evaluation set | Expert baseline MSE | Candidate MSE | Change |
|---|---:|---:|---:|
| 1,700 held-out WAV sequences | 0.035102 | 0.017632 | 49.77% better |
| 512 expert sequences | 0.007762 | 0.009159 | 18.00% worse |

The checkpoint at
`/models/checkpoints/lewm-wav-20260903-205ep-v2/best` is the leading
experimental candidate, but `/models/checkpoints/lewm-pusht` remains active
because the expert-data forgetting check regressed. Mixed expert/WAV replay is
required before promotion.

### Interpretation

The acquisition score compares:

```text
LeWM predicted latent <-> NanoWM subgoal latent
```

The evaluation score compares:

```text
LeWM predicted latent <-> actual environment outcome latent
```

The baseline's actual-outcome MSE was already only `0.004782`. Thus, large
subgoal disagreement did not mean that these real outcomes were difficult for
LeWM. Training the full representation on a small and narrow dataset moved a
strong model away from its good solution.

## 12. Paper-faithful next steps

1. Keep maximum subgoal-to-prediction disagreement as the online acquisition
   score.
2. Append every selected real transition to the accumulated action dataset.
3. Implement multiple exploration rounds and update both LeWM and the IDM after
   each round, as in Algorithm 1.
4. Add the paper's learnable sparse input/latent mechanism to the IDM.
5. Use an action-free NanoWM checkpoint as the fixed subgoal prior.
6. Retain the warm-up data while adding each round's newly acquired data.
7. Measure actual successor prediction error only on held-out evaluation data,
   and measure rank correlation between WAV scores and Oracle errors.
8. Run random acquisition and, where simulator look-ahead permits it, Oracle
   acquisition as equal-budget experimental baselines.
9. Preserve the expert checkpoint and report both prediction MSE and
   action-following behavior after every round.
