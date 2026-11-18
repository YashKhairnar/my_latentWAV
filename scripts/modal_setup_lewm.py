from __future__ import annotations

import modal


app = modal.App("latentwav-lewm-setup")

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==0.25.2",
    "hf-xet",
)

models_volume = modal.Volume.from_name("latentwav-models")

REPO_ID = "quentinll/lewm-pusht"
CHECKPOINT_DIR = "/models/checkpoints/lewm-pusht"
FILES = ("config.json", "weights.pt")


@app.function(
    image=image,
    volumes={"/models": models_volume},
    timeout=1800,
)
def setup_lewm() -> dict:
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for filename in FILES:
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="model",
                filename=filename,
                local_dir=checkpoint_dir,
            )
        )
        downloaded[filename] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }

    with (checkpoint_dir / "config.json").open() as file:
        config = json.load(file)

    required_sections = {
        "encoder",
        "predictor",
        "action_encoder",
        "projector",
        "pred_proj",
    }
    missing = required_sections - set(config.keys())
    if missing:
        raise ValueError(f"LeWM config is missing sections: {sorted(missing)}")

    if config["predictor"]["num_frames"] != 3:
        raise ValueError("Expected the PushT LeWM checkpoint to use 3 history frames")
    if config["action_encoder"]["input_dim"] != 10:
        raise ValueError("Expected the PushT LeWM checkpoint action dimension to be 10")
    if config["predictor"]["output_dim"] != 192:
        raise ValueError("Expected the PushT LeWM checkpoint latent dimension to be 192")
    if downloaded["weights.pt"]["size_bytes"] < 1_000_000:
        raise ValueError("Downloaded weights.pt is unexpectedly small")

    models_volume.commit()
    return {
        "repo_id": REPO_ID,
        "checkpoint_dir": CHECKPOINT_DIR,
        "history_size": config["predictor"]["num_frames"],
        "action_block_dim": config["action_encoder"]["input_dim"],
        "latent_dim": config["predictor"]["output_dim"],
        "image_size": config["encoder"]["image_size"],
        "files": downloaded,
    }


@app.local_entrypoint()
def main():
    result = setup_lewm.remote()
    print("LeWM checkpoint installed on Modal.")
    print(f"Repository: {result['repo_id']}")
    print(f"Directory: {result['checkpoint_dir']}")
    print(
        f"history={result['history_size']}, "
        f"action_block_dim={result['action_block_dim']}, "
        f"latent_dim={result['latent_dim']}, "
        f"image_size={result['image_size']}"
    )
    for filename, metadata in result["files"].items():
        size_mb = metadata["size_bytes"] / 1_000_000
        print(f"  {filename}: {size_mb:.2f} MB")
