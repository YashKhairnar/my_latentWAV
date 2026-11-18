"""PushT episode collection and HDF5 rollout persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class EpisodeRollout:
    """One complete sequence of pre-action observations and executed actions."""

    pixels: np.ndarray
    actions: np.ndarray
    proprio: np.ndarray
    states: np.ndarray
    rewards: np.ndarray
    disagreements: np.ndarray
    selected_subgoal_indices: np.ndarray
    macro_step_indices: np.ndarray
    terminated: bool

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def _extract_observation(observation, state=None):
    if not isinstance(observation, dict):
        raise TypeError("PushT observation must be a dictionary")

    pixel_key = "visual" if "visual" in observation else "pixels"
    if pixel_key not in observation:
        raise KeyError("Observation must contain 'visual' or 'pixels'")
    pixels = np.asarray(observation[pixel_key])
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("Observation pixels must be uint8 with shape (H,W,3)")

    if "proprio" not in observation:
        raise KeyError("Observation must contain 'proprio'")
    proprio = np.asarray(observation["proprio"], dtype=np.float32)
    if proprio.ndim != 1 or not np.isfinite(proprio).all():
        raise ValueError("Observation proprio must be a finite vector")

    if state is None:
        state = observation.get("state")
    if state is None:
        raise KeyError("PushT state was not returned by reset/step")
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("PushT state must be a finite vector")

    return np.ascontiguousarray(pixels), proprio, state


def _reset_environment(environment, seed):
    if seed is not None and hasattr(environment, "seed"):
        environment.seed(int(seed))
    result = environment.reset()
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Environment reset must return (observation, state/info)")
    observation, auxiliary = result
    state = auxiliary.get("state") if isinstance(auxiliary, dict) else auxiliary
    return observation, state


def _step_environment(environment, action):
    result = environment.step(action)
    if not isinstance(result, tuple):
        raise TypeError("Environment step must return a tuple")
    if len(result) == 4:
        observation, reward, done, info = result
        terminated = bool(done)
    elif len(result) == 5:
        observation, reward, terminated, truncated, info = result
        terminated = bool(terminated or truncated)
    else:
        raise ValueError("Environment step must return 4 or 5 values")
    if not isinstance(info, dict):
        raise TypeError("Environment step info must be a dictionary")
    return observation, float(reward), terminated, info


def collect_episode(environment, policy, max_steps, seed=None) -> EpisodeRollout:
    """Run one policy-controlled episode and retain every executed transition."""
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    policy.reset()
    observation, current_state = _reset_environment(environment, seed)
    pixel_rows = []
    action_rows = []
    proprio_rows = []
    state_rows = []
    reward_rows = []
    disagreement_rows = []
    selected_index_rows = []
    macro_step_rows = []
    terminated = False

    for _ in range(max_steps):
        pixels, proprio, state = _extract_observation(observation, current_state)
        action = np.asarray(policy.act(pixels), dtype=np.float32)
        if action.shape != (policy.raw_action_dim,):
            raise RuntimeError(
                f"Policy action must have shape {(policy.raw_action_dim,)}"
            )
        if not np.isfinite(action).all():
            raise RuntimeError("Policy action contains NaN or infinity")

        decision = policy.last_decision
        if decision is None:
            raise RuntimeError("Policy did not expose its latest decision")

        next_observation, reward, terminated, info = _step_environment(
            environment, action
        )
        pixel_rows.append(pixels.copy())
        action_rows.append(action.copy())
        proprio_rows.append(proprio.copy())
        state_rows.append(state.copy())
        reward_rows.append(reward)
        disagreement_rows.append(decision.verification.selected_score)
        selected_index_rows.append(decision.verification.selected_index)
        macro_step_rows.append(policy.macro_steps - 1)

        observation = next_observation
        observation_state = (
            next_observation.get("state")
            if isinstance(next_observation, dict)
            else None
        )
        current_state = info.get("state", observation_state)
        if terminated:
            break

    return EpisodeRollout(
        pixels=np.stack(pixel_rows).astype(np.uint8, copy=False),
        actions=np.stack(action_rows).astype(np.float32, copy=False),
        proprio=np.stack(proprio_rows).astype(np.float32, copy=False),
        states=np.stack(state_rows).astype(np.float32, copy=False),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        disagreements=np.asarray(disagreement_rows, dtype=np.float32),
        selected_subgoal_indices=np.asarray(selected_index_rows, dtype=np.int64),
        macro_step_indices=np.asarray(macro_step_rows, dtype=np.int64),
        terminated=terminated,
    )


def _validate_episodes(episodes):
    episodes = list(episodes)
    if not episodes:
        raise ValueError("At least one episode is required")

    array_fields = (
        "pixels",
        "actions",
        "proprio",
        "states",
        "rewards",
        "disagreements",
        "selected_subgoal_indices",
        "macro_step_indices",
    )
    reference_tails = None
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, EpisodeRollout):
            raise TypeError("Every episode must be an EpisodeRollout")
        length = len(episode)
        if length < 1:
            raise ValueError(f"Episode {episode_index} is empty")
        for field in array_fields:
            value = getattr(episode, field)
            if value.shape[0] != length:
                raise ValueError(
                    f"Episode {episode_index} field {field!r} has wrong length"
                )

        tails = tuple(getattr(episode, field).shape[1:] for field in array_fields)
        if reference_tails is None:
            reference_tails = tails
        elif tails != reference_tails:
            raise ValueError("All episodes must use matching tensor shapes")
    return episodes


def save_rollouts(path, episodes, overwrite=False) -> Path:
    """Atomically save episodes using the LeWM/IDM-compatible HDF5 schema."""
    path = Path(path)
    episodes = _validate_episodes(episodes)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing rollout: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    lengths = np.asarray([len(episode) for episode in episodes], dtype=np.int64)
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1])]
    )
    episode_ids = np.repeat(np.arange(len(episodes), dtype=np.int64), lengths)
    step_ids = np.concatenate(
        [np.arange(length, dtype=np.int64) for length in lengths]
    )

    field_to_key = {
        "pixels": "pixels",
        "actions": "action",
        "proprio": "proprio",
        "states": "state",
        "rewards": "reward",
        "disagreements": "disagreement",
        "selected_subgoal_indices": "selected_subgoal_idx",
        "macro_step_indices": "macro_step_idx",
    }
    try:
        with h5py.File(temporary_path, "w") as dataset:
            dataset.create_dataset("ep_len", data=lengths)
            dataset.create_dataset("ep_offset", data=offsets)
            dataset.create_dataset("episode_idx", data=episode_ids)
            dataset.create_dataset("step_idx", data=step_ids)
            for field, key in field_to_key.items():
                values = np.concatenate(
                    [getattr(episode, field) for episode in episodes], axis=0
                )
                dataset.create_dataset(
                    key,
                    data=values,
                    chunks=True,
                    compression="lzf",
                )
            dataset.create_dataset(
                "terminated",
                data=np.asarray([episode.terminated for episode in episodes]),
            )
            dataset.attrs["format"] = "latentwav-rollout-v1"
            dataset.attrs["num_episodes"] = len(episodes)
            dataset.attrs["num_rows"] = int(lengths.sum())
        temporary_path.replace(path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return path
