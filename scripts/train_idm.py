from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from latentwav.data import (
    PushTTransitionDataset,
    build_transition_indices,
    compute_action_stats,
    split_episode_ids,
)
from latentwav.models.idm import PixelTransformerIDM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the pixel-based inverse dynamics model on PushT."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, mps, cpu, or a specific device such as cuda:0",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA float16 autocast and gradient scaling.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an in-progress checkpoint every N training steps (0 disables).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, and scaler state from a checkpoint.",
    )
    parser.add_argument(
        "--resume-scheduler",
        choices=("auto", "continue", "restart"),
        default="auto",
        help=(
            "Scheduler behavior when resuming. 'auto' restarts cosine decay when "
            "--epochs extends the checkpoint's original total; otherwise it "
            "continues the saved scheduler."
        ),
    )

    # These limits are useful for inexpensive smoke tests and overfit tests.
    parser.add_argument("--max-train-transitions", type=int, default=None)
    parser.add_argument("--max-val-transitions", type=int, default=None)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-mlp-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if args.frameskip < 1:
        raise ValueError("frameskip must be at least 1")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay cannot be negative")
    if args.gradient_clip < 0:
        raise ValueError("gradient-clip cannot be negative")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.log_every < 1:
        raise ValueError("log-every must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every cannot be negative")
    if args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    for name in ("max_train_transitions", "max_val_transitions"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def read_dataset_metadata(path: Path) -> tuple[int, int]:
    """Return the number of episodes and raw action dimensions."""
    with h5py.File(path, "r") as dataset:
        required = {"ep_len", "ep_offset", "pixels", "action"}
        missing = required - set(dataset.keys())
        if missing:
            raise ValueError(f"Dataset is missing required keys: {sorted(missing)}")

        num_episodes = int(dataset["ep_len"].shape[0])
        if dataset["action"].ndim != 2:
            raise ValueError("action must have shape (rows, action_dim)")
        raw_action_dim = int(dataset["action"].shape[1])

    return num_episodes, raw_action_dim


def limit_indices(
    indices: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    """Select a reproducible random transition subset without replacement."""
    if maximum is None or maximum >= len(indices):
        return indices

    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(len(indices), size=maximum, replace=False)
    return indices[selected_positions]


def build_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, int]:
    num_episodes, raw_action_dim = read_dataset_metadata(args.dataset)
    action_mean, action_std = compute_action_stats(args.dataset)

    train_episode_ids, val_episode_ids = split_episode_ids(
        num_episodes,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_indices = build_transition_indices(
        args.dataset,
        frameskip=args.frameskip,
        episode_ids=train_episode_ids,
    )
    val_indices = build_transition_indices(
        args.dataset,
        frameskip=args.frameskip,
        episode_ids=val_episode_ids,
    )
    train_indices = limit_indices(
        train_indices,
        args.max_train_transitions,
        seed=args.seed,
    )
    val_indices = limit_indices(
        val_indices,
        args.max_val_transitions,
        seed=args.seed + 1,
    )

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("The train and validation splits must contain transitions")

    train_dataset = PushTTransitionDataset(
        args.dataset,
        frameskip=args.frameskip,
        transition_indices=train_indices,
        action_mean=action_mean,
        action_std=action_std,
    )
    val_dataset = PushTTransitionDataset(
        args.dataset,
        frameskip=args.frameskip,
        transition_indices=val_indices,
        action_mean=action_mean,
        action_std=action_std,
    )

    use_pinned_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    loader_generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pinned_memory,
        persistent_workers=persistent_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pinned_memory,
        persistent_workers=persistent_workers,
    )

    print(
        f"[data] episodes: train={len(train_episode_ids):,}, "
        f"validation={len(val_episode_ids):,}"
    )
    print(
        f"[data] transitions: train={len(train_indices):,}, "
        f"validation={len(val_indices):,}"
    )
    print(f"[data] action mean={action_mean}, std={action_std}")
    return train_loader, val_loader, action_mean, action_std, raw_action_dim


def move_batch(batch, device: torch.device):
    current, future, target = batch
    non_blocking = device.type == "cuda"
    return (
        current.to(device, non_blocking=non_blocking),
        future.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
    )


def train_one_epoch(
    model: PixelTransformerIDM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    log_every: int,
    epoch: int,
    scaler: torch.amp.GradScaler,
    use_mixed_precision: bool,
    progress_callback=None,
) -> float:
    model.train()
    squared_error_sum = 0.0
    scalar_count = 0

    for step, batch in enumerate(loader, start=1):
        current, future, target = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_mixed_precision,
        ):
            prediction = model(current, future)
            loss = F.mse_loss(prediction, target)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        squared_error_sum += loss.item() * target.numel()
        scalar_count += target.numel()

        if step % log_every == 0 or step == len(loader):
            running_mse = squared_error_sum / scalar_count
            print(
                f"[train] epoch={epoch} step={step:,}/{len(loader):,} "
                f"mse={running_mse:.6f}"
            )
        if progress_callback is not None:
            progress_callback(step, squared_error_sum / scalar_count)

    return squared_error_sum / scalar_count


@torch.no_grad()
def evaluate(
    model: PixelTransformerIDM,
    loader: DataLoader,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> dict[str, float | list[float]]:
    model.eval()
    squared_error_sum = 0.0
    per_dimension_error = None
    prediction_norm_sum = 0.0
    target_norm_sum = 0.0
    scalar_count = 0
    sample_count = 0

    for batch in loader:
        current, future, target = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_mixed_precision,
        ):
            prediction = model(current, future)
        squared_error = (prediction - target).square()

        squared_error_sum += squared_error.sum().item()
        dimension_sum = squared_error.sum(dim=0)
        per_dimension_error = (
            dimension_sum
            if per_dimension_error is None
            else per_dimension_error + dimension_sum
        )
        prediction_norm_sum += prediction.norm(dim=1).sum().item()
        target_norm_sum += target.norm(dim=1).sum().item()
        scalar_count += target.numel()
        sample_count += target.shape[0]

    if sample_count == 0 or per_dimension_error is None:
        raise ValueError("Validation loader is empty")

    return {
        "action_mse": squared_error_sum / scalar_count,
        "per_dimension_mse": (
            per_dimension_error / sample_count
        ).cpu().tolist(),
        "prediction_norm": prediction_norm_sum / sample_count,
        "target_norm": target_norm_sum / sample_count,
    }


def save_checkpoint(
    path: Path,
    model: PixelTransformerIDM,
    optimizer: torch.optim.Optimizer,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    args: argparse.Namespace,
    epoch: int,
    train_mse: float,
    validation_metrics: dict | None,
    scheduler=None,
    scaler=None,
    epoch_step: int = 0,
    epoch_complete: bool = True,
    best_validation_mse: float = float("inf"),
    best_epoch: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config,
        "action_mean": action_mean,
        "action_std": action_std,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "epoch_complete": epoch_complete,
        "train_mse": train_mse,
        "validation_metrics": validation_metrics,
        "training_args": vars(args),
        "best_validation_mse": best_validation_mse,
        "best_epoch": best_epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    temporary_path = path.with_name(f"{path.name}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_training_checkpoint(
    path: Path,
    model: PixelTransformerIDM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    args: argparse.Namespace,
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_config") != model.config:
        raise ValueError("Resume checkpoint model configuration does not match")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    saved_args = checkpoint.get("training_args") or {}
    saved_total_epochs = int(saved_args.get("epochs", checkpoint["epoch"]))
    extends_training = args.epochs > saved_total_epochs
    restart_scheduler = args.resume_scheduler == "restart" or (
        args.resume_scheduler == "auto" and extends_training
    )

    if restart_scheduler:
        # The optimizer checkpoint contains the LR reached by the old cosine
        # schedule. Restore the requested base LR before starting a fresh
        # cosine decay over only the newly requested epochs.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
            parameter_group["initial_lr"] = args.learning_rate
    elif "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    saved_epoch = int(checkpoint["epoch"])
    start_epoch = saved_epoch + 1 if checkpoint.get("epoch_complete", True) else saved_epoch
    if restart_scheduler:
        remaining_epochs = args.epochs - start_epoch + 1
        if remaining_epochs < 1:
            raise ValueError(
                f"Resume checkpoint starts at epoch {start_epoch}, but --epochs "
                f"is only {args.epochs}"
            )
        scheduler.T_max = remaining_epochs
        scheduler.base_lrs = [
            parameter_group["lr"] for parameter_group in optimizer.param_groups
        ]
        print(
            f"[resume] restarted cosine scheduler for {remaining_epochs} epochs "
            f"at lr={args.learning_rate:.2e}"
        )
    else:
        print(
            f"[resume] continued saved scheduler "
            f"(T_max={scheduler.T_max}, last_epoch={scheduler.last_epoch})"
        )
    best_validation_mse = float(
        checkpoint.get("best_validation_mse", float("inf"))
    )
    best_epoch = int(checkpoint.get("best_epoch", 0))
    print(
        f"[resume] loaded {path}; continuing at epoch {start_epoch} "
        f"(saved step {checkpoint.get('epoch_step', 0):,})"
    )
    return start_epoch, best_validation_mse, best_epoch


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[setup] device={device}")

    train_loader, val_loader, action_mean, action_std, raw_action_dim = (
        build_dataloaders(args, device)
    )
    model = PixelTransformerIDM(
        raw_action_dim=raw_action_dim,
        frameskip=args.frameskip,
        embed_dim=args.embed_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_mlp_dim=args.transformer_mlp_dim,
        dropout=args.dropout,
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[model] parameters={parameter_count:,}, "
        f"action_block_dim={model.action_block_dim}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    output = args.output
    latest_output = output.with_name(f"{output.stem}_latest{output.suffix}")
    progress_output = output.with_name(f"{output.stem}_progress{output.suffix}")
    use_mixed_precision = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    print(f"[setup] mixed_precision={use_mixed_precision}")

    start_epoch = 1
    best_validation_mse = float("inf")
    best_epoch = 0
    if args.resume is not None:
        start_epoch, best_validation_mse, best_epoch = load_training_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            args,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        def save_progress(step: int, running_mse: float) -> None:
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(
                    progress_output,
                    model,
                    optimizer,
                    action_mean,
                    action_std,
                    args,
                    epoch,
                    running_mse,
                    None,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch_step=step,
                    epoch_complete=False,
                    best_validation_mse=best_validation_mse,
                    best_epoch=best_epoch,
                )
                print(f"[checkpoint] progress saved at epoch={epoch}, step={step:,}")

        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.gradient_clip,
            args.log_every,
            epoch,
            scaler,
            use_mixed_precision,
            save_progress,
        )
        validation_metrics = evaluate(
            model,
            val_loader,
            device,
            use_mixed_precision,
        )
        scheduler.step()

        validation_mse = float(validation_metrics["action_mse"])
        print(
            f"[epoch {epoch}] train_mse={train_mse:.6f}, "
            f"validation_mse={validation_mse:.6f}, "
            f"prediction_norm={validation_metrics['prediction_norm']:.4f}, "
            f"target_norm={validation_metrics['target_norm']:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        save_checkpoint(
            latest_output,
            model,
            optimizer,
            action_mean,
            action_std,
            args,
            epoch,
            train_mse,
            validation_metrics,
            scheduler=scheduler,
            scaler=scaler,
            epoch_complete=True,
            best_validation_mse=min(best_validation_mse, validation_mse),
            best_epoch=epoch if validation_mse < best_validation_mse else best_epoch,
        )

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            save_checkpoint(
                output,
                model,
                optimizer,
                action_mean,
                action_std,
                args,
                epoch,
                train_mse,
                validation_metrics,
                scheduler=scheduler,
                scaler=scaler,
                epoch_complete=True,
                best_validation_mse=validation_mse,
                best_epoch=epoch,
            )
            print(f"[checkpoint] saved new best model to {output}")

        save_checkpoint(
            progress_output,
            model,
            optimizer,
            action_mean,
            action_std,
            args,
            epoch,
            train_mse,
            validation_metrics,
            scheduler=scheduler,
            scaler=scaler,
            epoch_complete=True,
            best_validation_mse=best_validation_mse,
            best_epoch=best_epoch,
        )

    print(
        f"[done] best validation MSE={best_validation_mse:.6f} "
        f"at epoch {best_epoch}"
    )
    print(f"[done] best checkpoint: {output}")
    print(f"[done] latest checkpoint: {latest_output}")


if __name__ == "__main__":
    main()
