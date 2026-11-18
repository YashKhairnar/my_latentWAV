"""Collect real PushT episodes with the complete LatentWAV policy on Modal."""

from __future__ import annotations

from datetime import datetime, timezone

import modal


app = modal.App("latentwav-pusht-collection")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "Pillow",
        "timm",
        "einops",
        "diffusers==0.24.0",
        "huggingface-hub==0.25.2",
        "transformers==4.45.2",
        "accelerate",
        "omegaconf",
        "hydra-core",
        "safetensors",
        "h5py",
        "hdf5plugin",
        "gym==0.26.2",
        "pygame==2.6.1",
        "pymunk==6.6.0",
        "opencv-python-headless",
        "shapely",
        "scikit-image",
        "matplotlib",
    )
    .env(
        {
            "HF_HOME": "/models/huggingface",
            "PYTHONPATH": "/root/latentwav-src:/root/nano-world-model/src",
            "SDL_VIDEODRIVER": "dummy",
        }
    )
    .add_local_dir("nano-world-model", remote_path="/root/nano-world-model")
    .add_local_dir("src", remote_path="/root/latentwav-src")
)

models_volume = modal.Volume.from_name("latentwav-models")


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=6 * 60 * 60,
    memory=32768,
)
def collect(
    episodes: int = 1,
    max_steps: int = 5,
    num_subgoals: int = 2,
    sampling_steps: int = 50,
    lewm_checkpoint: str = "/models/checkpoints/lewm-pusht",
    base_seed: int = 3407,
    run_label: str = "",
) -> dict:
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import random
    import torch

    from environments.pusht.pusht_env import PushTEnv
    from latentwav.loop import collect_episode, save_rollouts
    from latentwav.models.idm import load_idm_checkpoint
    from latentwav.models.nanowm import NanoWMSubgoalGenerator
    from latentwav.models.world_model import LeWMWorldModel
    from latentwav.policy import LatentWAVPolicy
    from latentwav.verifier import ActionVerifier
    from latentwav.visualization import (
        rollout_gif_bytes,
        rollout_metrics_png_bytes,
    )

    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")

    dataset_path = "/models/datasets/lewm-pusht/pusht_expert_train.h5"
    idm, action_mean, action_std = load_idm_checkpoint(
        "/models/checkpoints/idm/idm_pusht_best.pt", device="cuda"
    )
    world_model = LeWMWorldModel(
        checkpoint_dir=lewm_checkpoint,
        device="cuda",
        action_mean=action_mean,
        action_std=action_std,
    )
    generator = NanoWMSubgoalGenerator(
        repo_path="/root/nano-world-model",
        checkpoint_dir="/models/nanowm-pusht",
        device="cuda",
        num_sampling_steps=sampling_steps,
    )

    # PushT declares absolute [0, 512] bounds even in relative-action mode.
    # The models were trained on scaled relative actions, so keep collection
    # inside the finite action range observed in their training dataset.
    with h5py.File(dataset_path, "r") as dataset:
        training_actions = dataset["action"][:]
        action_low = np.nanmin(training_actions, axis=0)
        action_high = np.nanmax(training_actions, axis=0)

    verifier = ActionVerifier(
        idm=idm,
        world_model=world_model,
        idm_action_mean=action_mean,
        idm_action_std=action_std,
        action_low=action_low,
        action_high=action_high,
    )
    policy = LatentWAVPolicy(generator, verifier, num_subgoals=num_subgoals)
    environment = PushTEnv(
        render_size=224,
        relative=True,
        action_scale=100,
        with_velocity=True,
        with_target=True,
    )

    try:
        rollouts = []
        for episode_index in range(episodes):
            episode_seed = base_seed + episode_index
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(episode_seed)
            rollout = collect_episode(
                environment,
                policy,
                max_steps=max_steps,
                seed=episode_seed,
            )
            rollouts.append(rollout)
            print(
                f"[collect] episode={episode_index + 1}/{episodes} "
                f"steps={len(rollout)} "
                f"mean_reward={float(rollout.rewards.mean()):.4f} "
                f"mean_disagreement={float(rollout.disagreements.mean()):.4f}",
                flush=True,
            )
    finally:
        environment.close()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(
        character for character in run_label if character.isalnum() or character in "-_"
    )
    name_prefix = f"latentwav_{safe_label}" if safe_label else "latentwav"
    output_path = f"/models/rollouts/{name_prefix}_{timestamp}.h5"
    save_rollouts(output_path, rollouts)
    models_volume.commit()

    with h5py.File(output_path, "r") as dataset:
        schema = {
            key: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
            for key, value in dataset.items()
        }
    return {
        "run_name": f"{name_prefix}_{timestamp}",
        "lewm_checkpoint": lewm_checkpoint,
        "base_seed": base_seed,
        "output_path": output_path,
        "episode_lengths": [len(rollout) for rollout in rollouts],
        "valid_frameskip_5_transitions": sum(
            max(0, len(rollout) - 5) for rollout in rollouts
        ),
        "terminated": [rollout.terminated for rollout in rollouts],
        "mean_rewards": [float(rollout.rewards.mean()) for rollout in rollouts],
        "mean_disagreements": [
            float(rollout.disagreements.mean()) for rollout in rollouts
        ],
        "schema": schema,
        "rollout_gif": rollout_gif_bytes(rollouts[0]),
        "metrics_png": rollout_metrics_png_bytes(rollouts[0]),
    }


@app.local_entrypoint()
def main(
    episodes: int = 1,
    max_steps: int = 5,
    num_subgoals: int = 2,
    sampling_steps: int = 50,
    lewm_checkpoint: str = "/models/checkpoints/lewm-pusht",
    base_seed: int = 3407,
    run_label: str = "",
):
    result = collect.remote(
        episodes=episodes,
        max_steps=max_steps,
        num_subgoals=num_subgoals,
        sampling_steps=sampling_steps,
        lewm_checkpoint=lewm_checkpoint,
        base_seed=base_seed,
        run_label=run_label,
    )
    from pathlib import Path

    output_directory = Path("artifacts/rollouts") / result["run_name"]
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "episode_000.gif").write_bytes(result.pop("rollout_gif"))
    (output_directory / "episode_000_metrics.png").write_bytes(
        result.pop("metrics_png")
    )
    print("PushT rollout collection passed.")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"local_artifacts: {output_directory.resolve()}")
