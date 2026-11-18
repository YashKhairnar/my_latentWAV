"""Launch from-scratch LeWM training on Modal."""

from __future__ import annotations

import subprocess

import modal


app = modal.App("latentwav-lewm-from-scratch")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.45.2",
        "hydra-core==1.3.2",
        "h5py",
        "hdf5plugin",
        "numpy",
    )
    .env({"PYTHONPATH": "/root/latentwav-src"})
    .add_local_dir("src", remote_path="/root/latentwav-src")
    .add_local_file("scripts/train_lewm.py", remote_path="/root/train_lewm.py")
)
models_volume = modal.Volume.from_name("latentwav-models")


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=6 * 60 * 60,
    cpu=16,
    memory=65536,
)
def train(
    smoke_test: bool = True,
    rollout_path: str = "",
    max_steps: int = 5_000,
    output_name: str = "lewm-wav-finetuned",
    learning_rate: float = 5e-5,
    warmup_steps: int = 1_000,
    val_every: int = 500,
    freeze_visual_encoder: bool = False,
    sigreg_weight: float = 0.09,
    disable_dropout: bool = False,
):
    expert_dataset = "/models/datasets/lewm-pusht/pusht_expert_train.h5"
    if rollout_path:
        dataset = rollout_path
        output = f"/models/checkpoints/{output_name}"
    elif smoke_test:
        dataset = "/models/rollouts/latentwav_20260812T224310Z.h5"
        output = "/models/checkpoints/lewm-scratch-smoke"
    else:
        dataset = expert_dataset
        output = "/models/checkpoints/lewm-scratch"
    command = [
        "python",
        "/root/train_lewm.py",
        "--dataset", dataset,
        "--config", "/models/checkpoints/lewm-pusht/config.json",
        "--output-dir", output,
        "--device", "cuda",
        "--max-steps", str(150 if smoke_test else max_steps),
        "--batch-size", "35" if smoke_test else "128",
        "--num-workers", "2" if smoke_test else "12",
        "--learning-rate", str(learning_rate),
        "--sigreg-weight", str(sigreg_weight),
        "--warmup-steps", "15" if smoke_test else str(warmup_steps),
        "--val-every", "25" if smoke_test else str(val_every),
        "--save-every", "25" if smoke_test else "500",
        "--log-every", "10" if smoke_test else "25",
        "--mixed-precision",
    ]
    if rollout_path:
        command.extend(
            [
                "--action-stats-dataset", expert_dataset,
                "--initial-checkpoint-dir", "/models/checkpoints/lewm-pusht",
            ]
        )
    if freeze_visual_encoder:
        command.append("--freeze-visual-encoder")
    if disable_dropout:
        command.append("--disable-dropout")
    if smoke_test:
        command.extend(
            [
                "--overfit",
                "--sigreg-weight", "0",
                "--sigreg-projections", "128",
            ]
        )
    subprocess.run(command, check=True)
    models_volume.commit()
    return output


@app.local_entrypoint()
def main(
    full: bool = False,
    rollout_path: str = "",
    max_steps: int = 5_000,
    output_name: str = "lewm-wav-finetuned",
    learning_rate: float = 5e-5,
    warmup_steps: int = 1_000,
    val_every: int = 500,
    freeze_visual_encoder: bool = False,
    sigreg_weight: float = 0.09,
    disable_dropout: bool = False,
):
    if rollout_path and not full:
        raise ValueError("Pass --full when fine-tuning an expert checkpoint")
    output = train.remote(
        smoke_test=not full,
        rollout_path=rollout_path,
        max_steps=max_steps,
        output_name=output_name,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        val_every=val_every,
        freeze_visual_encoder=freeze_visual_encoder,
        sigreg_weight=sigreg_weight,
        disable_dropout=disable_dropout,
    )
    print(f"LeWM training completed: {output}")
