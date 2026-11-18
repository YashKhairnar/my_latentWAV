"""Train the PushT LeWM architecture from random initialization."""

from __future__ import annotations

import argparse
import json
import math
import random
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
from latentwav.models.lewm_core import SIGReg
from latentwav.models.world_model import build_lewm_from_config


PIXEL_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
PIXEL_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--action-stats-dataset",
        type=Path,
        help="Optional dataset providing the action normalization statistics.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint-dir",
        type=Path,
        help="Load config.json and weights.pt from this LeWM before training.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-projections", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-train-sequences", type=int)
    parser.add_argument("--max-val-sequences", type=int)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument(
        "--freeze-visual-encoder",
        action="store_true",
        help="Keep the pretrained image encoder and projector fixed during fine-tuning.",
    )
    parser.add_argument(
        "--disable-dropout",
        action="store_true",
        help="Fine-tune deterministically while retaining gradient computation.",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.dataset.exists() or not args.config.exists():
        raise FileNotFoundError("Dataset and config paths must exist")
    if args.action_stats_dataset is not None and not args.action_stats_dataset.exists():
        raise FileNotFoundError("Action-statistics dataset does not exist")
    if args.initial_checkpoint_dir is not None:
        weights = args.initial_checkpoint_dir / "weights.pt"
        if not weights.exists():
            raise FileNotFoundError(f"Initial LeWM weights not found: {weights}")
    for name in ("max_steps", "batch_size", "val_every", "save_every", "log_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_indices(indices, maximum, seed):
    if maximum is None or len(indices) <= maximum:
        return indices
    generator = np.random.default_rng(seed)
    return generator.choice(indices, size=maximum, replace=False)


def build_loaders(args, action_mean, action_std):
    with h5py.File(args.dataset, "r") as dataset:
        num_episodes = len(dataset["ep_len"])

    if args.overfit or num_episodes == 1:
        all_indices = build_lewm_sequence_indices(args.dataset)
        train_indices = limit_indices(
            all_indices, args.max_train_sequences, args.seed
        )
        val_indices = train_indices
    else:
        train_episodes, val_episodes = split_episode_ids(
            num_episodes, args.val_fraction, args.seed
        )
        train_indices = build_lewm_sequence_indices(
            args.dataset, episode_ids=train_episodes
        )
        val_indices = build_lewm_sequence_indices(
            args.dataset, episode_ids=val_episodes
        )
        train_indices = limit_indices(
            train_indices, args.max_train_sequences, args.seed
        )
        val_indices = limit_indices(
            val_indices, args.max_val_sequences, args.seed + 1
        )
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Dataset does not contain a valid LeWM sequence")

    train_dataset = LeWMSequenceDataset(
        args.dataset,
        sequence_indices=train_indices,
        action_mean=action_mean,
        action_std=action_std,
    )
    val_dataset = LeWMSequenceDataset(
        args.dataset,
        sequence_indices=val_indices,
        action_mean=action_mean,
        action_std=action_std,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    print(f"[data] train_sequences={len(train_dataset):,} val_sequences={len(val_dataset):,}")
    return train_loader, val_loader


def encode(model, frames, device):
    frames = frames.to(device, non_blocking=True).float().div_(255)
    mean = PIXEL_MEAN.to(device)
    std = PIXEL_STD.to(device)
    frames = (frames - mean) / std
    batch, length = frames.shape[:2]
    encoded = model.encoder(
        frames.reshape(batch * length, 3, frames.shape[-2], frames.shape[-1]),
        interpolate_pos_encoding=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state[:, 0]
    projected = model.projector(encoded)
    return projected.reshape(batch, length, -1)


def compute_losses(model, sigreg, frames, actions, device, sigreg_weight):
    latents = encode(model, frames, device)
    actions = actions.to(device, non_blocking=True)
    action_embeddings = model.action_encoder(actions)
    predictions = model.predict(latents[:, :-1], action_embeddings)
    targets = latents[:, 1:]
    prediction_loss = F.mse_loss(predictions, targets)
    regularization_loss = (
        sigreg(latents.transpose(0, 1))
        if sigreg_weight > 0
        else prediction_loss.new_zeros(())
    )
    total_loss = prediction_loss + sigreg_weight * regularization_loss
    return total_loss, prediction_loss, regularization_loss


@torch.no_grad()
def validate(model, sigreg, loader, device, sigreg_weight, mixed_precision):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    examples = 0
    for frames, actions in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=mixed_precision,
        ):
            losses = compute_losses(
                model, sigreg, frames, actions, device, sigreg_weight
            )
        batch_size = frames.shape[0]
        totals += np.asarray([float(loss) for loss in losses]) * batch_size
        examples += batch_size
    model.train()
    return tuple((totals / examples).tolist())


def save_model(output_dir, model, config, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".weights.pt.tmp"
    torch.save(model.state_dict(), temporary)
    temporary.replace(output_dir / "weights.pt")
    with (output_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)
    with (output_dir / "training.json").open("w") as file:
        json.dump(metadata, file, indent=2)


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    stats_dataset = args.action_stats_dataset or args.dataset
    action_mean, action_std = compute_action_stats(stats_dataset)
    train_loader, val_loader = build_loaders(args, action_mean, action_std)
    model, config = build_lewm_from_config(args.config, device=device)
    if args.initial_checkpoint_dir is not None:
        weights_path = args.initial_checkpoint_dir / "weights.pt"
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"[model] initialized from expert checkpoint: {weights_path}")
    else:
        print("[model] initialized from random weights")
    if args.freeze_visual_encoder:
        for module in (model.encoder, model.projector):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        print("[model] froze visual encoder and projector")
    def enter_training_mode():
        model.train()
        if args.disable_dropout:
            model.eval()
        elif args.freeze_visual_encoder:
            model.encoder.eval()
            model.projector.eval()

    enter_training_mode()
    sigreg = SIGReg(num_proj=args.sigreg_projections).to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    def lr_factor(step):
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            args.max_steps - args.warmup_steps, 1
        )
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    mixed_precision = args.mixed_precision and device.type == "cuda"
    iterator = iter(train_loader)
    best_validation = float("inf")
    initial_validation = validate(
        model, sigreg, val_loader, device, args.sigreg_weight, mixed_precision
    )
    enter_training_mode()
    print(
        f"[initial] total={initial_validation[0]:.6f} "
        f"prediction={initial_validation[1]:.6f} sigreg={initial_validation[2]:.6f}"
    )

    for step in range(1, args.max_steps + 1):
        try:
            frames, actions = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            frames, actions = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=mixed_precision,
        ):
            total, prediction, regularization = compute_losses(
                model, sigreg, frames, actions, device, args.sigreg_weight
            )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            print(
                f"[train] step={step}/{args.max_steps} total={float(total):.6f} "
                f"prediction={float(prediction):.6f} "
                f"sigreg={float(regularization):.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.max_steps:
            metrics = validate(
                model, sigreg, val_loader, device, args.sigreg_weight, mixed_precision
            )
            enter_training_mode()
            print(
                f"[validation] step={step} total={metrics[0]:.6f} "
                f"prediction={metrics[1]:.6f} sigreg={metrics[2]:.6f}",
                flush=True,
            )
            metadata = {
                "initialized_from_scratch": args.initial_checkpoint_dir is None,
                "initial_checkpoint_dir": (
                    str(args.initial_checkpoint_dir)
                    if args.initial_checkpoint_dir is not None
                    else None
                ),
                "step": step,
                "initial_validation": initial_validation,
                "validation": metrics,
                "action_mean": action_mean.tolist(),
                "action_std": action_std.tolist(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            }
            save_model(args.output_dir / "latest", model, config, metadata)
            if metrics[1] < best_validation:
                best_validation = metrics[1]
                save_model(args.output_dir / "best", model, config, metadata)

        elif step % args.save_every == 0:
            metadata = {
                "initialized_from_scratch": args.initial_checkpoint_dir is None,
                "initial_checkpoint_dir": (
                    str(args.initial_checkpoint_dir)
                    if args.initial_checkpoint_dir is not None
                    else None
                ),
                "step": step,
                "action_mean": action_mean.tolist(),
                "action_std": action_std.tolist(),
            }
            save_model(args.output_dir / "latest", model, config, metadata)

    print(f"[done] best validation prediction MSE={best_validation:.6f}")


if __name__ == "__main__":
    main()
