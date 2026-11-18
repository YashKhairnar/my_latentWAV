"""Compare two LeWM checkpoints on exactly the same held-out sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from latentwav.data import (
    LeWMSequenceDataset,
    build_lewm_sequence_indices,
    compute_action_stats,
    split_episode_ids,
)
from latentwav.models.world_model import LeWMWorldModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action-stats-dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def held_out_indices(args):
    with h5py.File(args.dataset, "r") as dataset:
        num_episodes = len(dataset["ep_len"])
    if num_episodes < 2:
        indices = build_lewm_sequence_indices(args.dataset)
    else:
        _, validation_episodes = split_episode_ids(
            num_episodes, args.val_fraction, args.seed
        )
        indices = build_lewm_sequence_indices(
            args.dataset, episode_ids=validation_episodes
        )
    if len(indices) > args.max_sequences:
        generator = np.random.default_rng(args.seed)
        indices = generator.choice(indices, args.max_sequences, replace=False)
    if len(indices) == 0:
        raise ValueError("No held-out LeWM sequences were found")
    return indices


@torch.no_grad()
def evaluate(model, loader):
    squared_error_sum = 0.0
    cosine_error_sum = 0.0
    target_sum = 0.0
    target_square_sum = 0.0
    prediction_sum = 0.0
    prediction_square_sum = 0.0
    element_count = 0
    example_count = 0

    for frames, normalized_actions in loader:
        frames = frames.permute(0, 1, 3, 4, 2).contiguous()
        normalized_actions = normalized_actions.to(model.device, non_blocking=True)
        history = model.encode_frames(frames[:, :-1])
        targets = model.encode_frames(frames[:, -1:])[:, 0]
        predictions = model.predict_next(
            history,
            normalized_actions,
            actions_normalized=True,
        )

        difference = predictions - targets
        squared_error_sum += float(difference.square().sum())
        cosine_error_sum += float(
            (1 - F.cosine_similarity(predictions, targets, dim=-1)).sum()
        )
        target_sum += float(targets.sum())
        target_square_sum += float(targets.square().sum())
        prediction_sum += float(predictions.sum())
        prediction_square_sum += float(predictions.square().sum())
        element_count += targets.numel()
        example_count += targets.shape[0]

    mse = squared_error_sum / element_count
    target_mean = target_sum / element_count
    target_variance = max(
        target_square_sum / element_count - target_mean**2,
        0.0,
    )
    prediction_mean = prediction_sum / element_count
    prediction_variance = max(
        prediction_square_sum / element_count - prediction_mean**2,
        0.0,
    )
    return {
        "sequences": example_count,
        "prediction_mse": mse,
        "normalized_mse": mse / max(target_variance, 1e-12),
        "cosine_error": cosine_error_sum / example_count,
        "target_latent_std": target_variance**0.5,
        "prediction_latent_std": prediction_variance**0.5,
    }


def main():
    args = parse_args()
    action_mean, action_std = compute_action_stats(args.action_stats_dataset)
    indices = held_out_indices(args)
    dataset = LeWMSequenceDataset(
        args.dataset,
        sequence_indices=indices,
        action_mean=action_mean,
        action_std=action_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    results = {}
    for name, checkpoint in (
        ("baseline", args.baseline),
        ("candidate", args.candidate),
    ):
        model = LeWMWorldModel(
            checkpoint,
            device=args.device,
            action_mean=action_mean,
            action_std=action_std,
        )
        results[name] = evaluate(model, loader)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_mse = results["baseline"]["prediction_mse"]
    candidate_mse = results["candidate"]["prediction_mse"]
    results["comparison"] = {
        "mse_change_percent": 100 * (candidate_mse - baseline_mse) / baseline_mse,
        "candidate_is_better": candidate_mse < baseline_mse,
    }
    text = json.dumps(results, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    dataset.close()


if __name__ == "__main__":
    main()
