from __future__ import annotations

import modal


app = modal.App("latentwav-pusht-dataset-setup")

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==0.25.2",
    "hf-xet",
    "zstandard",
    "h5py",
    "hdf5plugin",
    "numpy",
)

models_volume = modal.Volume.from_name("latentwav-models")

DATASET_REPO = "quentinll/lewm-pusht"
ARCHIVE_NAME = "pusht_expert_train.h5.zst"
DATASET_NAME = "pusht_expert_train.h5"
REQUIRED_KEYS = {"action", "ep_len", "ep_offset", "pixels"}
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


@app.function(
    image=image,
    volumes={"/models": models_volume},
    timeout=6 * 60 * 60,
    cpu=4,
    memory=8192,
)
def setup_pusht_dataset() -> dict:
    """Download, decompress, and validate the official LeWM PushT dataset."""
    import os
    from pathlib import Path

    import h5py
    import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
    import numpy as np
    import zstandard as zstd
    from huggingface_hub import hf_hub_download

    dataset_dir = Path("/models/datasets/lewm-pusht")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / ARCHIVE_NAME
    dataset_path = dataset_dir / DATASET_NAME
    partial_path = dataset_dir / f"{DATASET_NAME}.partial"

    if not dataset_path.exists():
        print(f"Downloading {DATASET_REPO}/{ARCHIVE_NAME} ...", flush=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=ARCHIVE_NAME,
                local_dir=dataset_dir,
            )
        )
        if downloaded.resolve() != archive_path.resolve():
            raise RuntimeError(f"Unexpected download location: {downloaded}")

        print(
            f"Downloaded {archive_path.stat().st_size / 1e9:.2f} GB; decompressing ...",
            flush=True,
        )
        with archive_path.open("rb") as compressed, partial_path.open("wb") as output:
            decompressor = zstd.ZstdDecompressor()
            decompressor.copy_stream(compressed, output)

        with partial_path.open("rb") as output:
            signature = output.read(len(HDF5_SIGNATURE))
        if signature != HDF5_SIGNATURE:
            raise ValueError(
                "The decompressed file is not an HDF5 file: "
                f"unexpected signature {signature!r}."
            )

        os.replace(partial_path, dataset_path)
        models_volume.commit()
    else:
        print(f"Dataset already exists; skipping download: {dataset_path}", flush=True)

    with h5py.File(dataset_path, "r") as dataset:
        keys = sorted(dataset.keys())
        missing = sorted(REQUIRED_KEYS - set(keys))
        if missing:
            raise ValueError(f"Dataset is missing required keys: {missing}")

        schema = {
            key: {
                "shape": tuple(dataset[key].shape),
                "dtype": str(dataset[key].dtype),
            }
            for key in keys
            if isinstance(dataset[key], h5py.Dataset)
        }
        episode_lengths = dataset["ep_len"][:]
        first_frame = dataset["pixels"][0]
        first_action = dataset["action"][0]

    result = {
        "dataset_path": str(dataset_path),
        "dataset_size_gb": round(dataset_path.stat().st_size / 1e9, 2),
        "num_rows": int(schema["pixels"]["shape"][0]),
        "num_episodes": int(episode_lengths.shape[0]),
        "episode_length_min": int(np.min(episode_lengths)),
        "episode_length_max": int(np.max(episode_lengths)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "first_frame_shape": tuple(first_frame.shape),
        "first_action": np.asarray(first_action).tolist(),
        "schema": schema,
    }
    print(result, flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = setup_pusht_dataset.remote()
    print("\nPushT dataset is ready on Modal.")
    print(f"Path: {result['dataset_path']}")
    print(
        f"Rows: {result['num_rows']:,}; episodes: {result['num_episodes']:,}; "
        f"uncompressed size: {result['dataset_size_gb']:.2f} GB"
    )
    print("Schema:")
    for key, metadata in result["schema"].items():
        print(f"  {key}: shape={metadata['shape']}, dtype={metadata['dtype']}")
