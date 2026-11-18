from __future__ import annotations

import subprocess
import time

import modal


app = modal.App("latentwav-idm-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
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
    )
    .uv_pip_install("h5py", "hdf5plugin")
    .env({"PYTHONPATH": "/root/latentwav-src"})
    .add_local_dir("src", remote_path="/root/latentwav-src")
    .add_local_dir("scripts", remote_path="/root/latentwav-scripts")
)

models_volume = modal.Volume.from_name("latentwav-models")

DATASET_PATH = "/models/datasets/lewm-pusht/pusht_expert_train.h5"
CHECKPOINT_DIR = "/models/checkpoints/idm"


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=6 * 60 * 60,
    cpu=16,
    memory=65536,
)
def train_idm(smoke_test: bool = True, epochs: int | None = None) -> str:
    """Run a small overfit test or full IDM training on Modal."""
    from pathlib import Path

    output_dir = Path(CHECKPOINT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if smoke_test:
        output_path = output_dir / "idm_smoke.pt"
        selected_epochs = 10 if epochs is None else epochs
        batch_size = 16
        workers = 2
    else:
        output_path = output_dir / "idm_pusht_best.pt"
        selected_epochs = 5 if epochs is None else epochs
        batch_size = 512
        workers = 16

    command = [
        "python",
        "/root/latentwav-scripts/train_idm.py",
        "--dataset",
        DATASET_PATH,
        "--output",
        str(output_path),
        "--device",
        "cuda",
        "--frameskip",
        "5",
        "--epochs",
        str(selected_epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
        "--log-every",
        "10" if smoke_test else "100",
        "--checkpoint-every",
        "100" if smoke_test else "500",
    ]

    if not smoke_test:
        command.append("--mixed-precision")
        progress_path = output_path.with_name(
            f"{output_path.stem}_progress{output_path.suffix}"
        )
        if progress_path.exists():
            command.extend(["--resume", str(progress_path)])
            print(f"Resuming from {progress_path}", flush=True)

    if smoke_test:
        command.extend(
            [
                "--max-train-transitions",
                "128",
                "--max-val-transitions",
                "128",
            ]
        )

    print(f"Running {'smoke' if smoke_test else 'full'} IDM training", flush=True)
    print(" ".join(command), flush=True)
    process = subprocess.Popen(command)
    while process.poll() is None:
        time.sleep(30)
        # Persist any atomic progress checkpoint while training is still active.
        models_volume.commit()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    models_volume.commit()
    return str(output_path)


@app.local_entrypoint()
def main(full: bool = False, epochs: int | None = None):
    output_path = train_idm.remote(smoke_test=not full, epochs=epochs)
    print(f"Training completed. Checkpoint saved to {output_path}")
