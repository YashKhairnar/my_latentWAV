"""Run baseline-versus-fine-tuned LeWM evaluation on Modal."""

from __future__ import annotations

import subprocess

import modal


app = modal.App("latentwav-lewm-comparison")
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
    .add_local_file("scripts/compare_lewm.py", remote_path="/root/compare_lewm.py")
)
models_volume = modal.Volume.from_name("latentwav-models")


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=1800,
    cpu=16,
    memory=65536,
)
def compare(
    dataset: str,
    max_sequences: int = 512,
    candidate: str = "/models/checkpoints/lewm-wav-finetuned/best",
):
    command = [
        "python", "/root/compare_lewm.py",
        "--dataset", dataset,
        "--action-stats-dataset", "/models/datasets/lewm-pusht/pusht_expert_train.h5",
        "--baseline", "/models/checkpoints/lewm-pusht",
        "--candidate", candidate,
        "--device", "cuda",
        "--batch-size", "128",
        "--num-workers", "12",
        "--max-sequences", str(max_sequences),
    ]
    subprocess.run(command, check=True)


@app.local_entrypoint()
def main(
    dataset: str = "/models/rollouts/latentwav_20260812T235844Z.h5",
    max_sequences: int = 512,
    candidate: str = "/models/checkpoints/lewm-wav-finetuned/best",
):
    compare.remote(dataset, max_sequences, candidate)
