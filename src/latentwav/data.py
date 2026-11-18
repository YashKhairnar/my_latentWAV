from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {"pixels", "action", "ep_len", "ep_offset"}


def inspect_dataset(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, "r") as dataset:
        print(f"Dataset: {path}")
        print("Schema:")

        for key in sorted(dataset.keys()):
            item = dataset[key]
            if isinstance(item, h5py.Dataset):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")

        missing_keys = REQUIRED_KEYS - set(dataset.keys())
        if missing_keys:
            raise ValueError(
                f"Dataset is missing required keys: {sorted(missing_keys)}"
            )

        episode_lengths = dataset["ep_len"][:]

        # Episode is a single, complete trial or trajectory
        # where an agent tries to push a T-shaped block onto a target zone
        print(f"Episodes: {len(episode_lengths):,}")
        print(f"Minimum episode length: {episode_lengths.min()}")
        print(f"Maximum episode length: {episode_lengths.max()}")
        print(f"Average episode length: {episode_lengths.mean():.2f}")

        first_frame = dataset["pixels"][0]
        first_action = dataset["action"][0]

        print(f"First frame: shape={first_frame.shape}, dtype={first_frame.dtype}")
        print(f"First action: {first_action}")


def compute_action_stats(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, "r") as dataset:
        if "action" not in dataset:
            raise ValueError(f"Dataset {path} does not contain 'action' key")

        actions = dataset["action"][:]
        valid_rows = ~np.isnan(actions).any(axis=1)
        valid_actions = actions[valid_rows]

        if len(valid_actions) == 0:
            raise ValueError("No valid actions found in the dataset.")

        mean = np.mean(valid_actions, axis=0).astype(np.float32)
        std = np.std(valid_actions, axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0

        return mean, std


def split_episode_ids(num_episodes, val_fraction=0.1, seed=0):
    """Split episode IDs into reproducible, disjoint train and validation sets."""
    if num_episodes < 2:
        raise ValueError("num_episodes must be at least 2")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    episode_ids = np.arange(num_episodes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    num_validation = int(round(num_episodes * val_fraction))
    num_validation = min(max(num_validation, 1), num_episodes - 1)

    validation_ids = episode_ids[:num_validation]
    train_ids = episode_ids[num_validation:]
    return train_ids, validation_ids


def build_transition_indices(path, frameskip=5, episode_ids=None):
    """Return valid global row indices for frame[t] -> frame[t + frameskip]."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if frameskip < 1:
        raise ValueError("frameskip must be at least 1")

    with h5py.File(path, "r") as dataset:
        required_keys = {"ep_offset", "ep_len"}
        missing_keys = required_keys - set(dataset.keys())
        if missing_keys:
            raise ValueError(
                f"Dataset is missing required keys: {sorted(missing_keys)}"
            )

        episode_offsets = dataset["ep_offset"][:]
        episode_lengths = dataset["ep_len"][:]

    if len(episode_offsets) != len(episode_lengths):
        raise ValueError(
            "ep_offset and ep_len must contain the same number of episodes"
        )

    if episode_ids is None:
        selected_episode_ids = np.arange(len(episode_lengths), dtype=np.int64)
    else:
        raw_episode_ids = np.asarray(episode_ids)
        if raw_episode_ids.ndim != 1:
            raise ValueError("episode_ids must be a one-dimensional array")
        if raw_episode_ids.size == 0:
            return np.empty(0, dtype=np.int64)
        if not np.issubdtype(raw_episode_ids.dtype, np.integer):
            raise ValueError("episode_ids must contain integers")

        selected_episode_ids = raw_episode_ids.astype(np.int64, copy=False)
        if np.any(selected_episode_ids < 0) or np.any(
            selected_episode_ids >= len(episode_lengths)
        ):
            raise ValueError("episode_ids contains an out-of-range episode")
        if len(np.unique(selected_episode_ids)) != len(selected_episode_ids):
            raise ValueError("episode_ids must not contain duplicates")

    index_chunks = []
    for episode_id in selected_episode_ids:
        episode_start = episode_offsets[episode_id]
        episode_length = episode_lengths[episode_id]
        valid_count = int(episode_length) - frameskip
        if valid_count <= 0:
            continue

        start = int(episode_start)
        index_chunks.append(
            np.arange(start, start + valid_count, dtype=np.int64)
        )

    if not index_chunks:
        return np.empty(0, dtype=np.int64)

    return np.concatenate(index_chunks)


def build_lewm_sequence_indices(
    path,
    frameskip=5,
    history_size=3,
    episode_ids=None,
):
    """Return starts for ``history_size + 1`` frame LeWM sequences."""
    path = Path(path)
    if frameskip < 1 or history_size < 1:
        raise ValueError("frameskip and history_size must be positive")
    with h5py.File(path, "r") as dataset:
        offsets = dataset["ep_offset"][:]
        lengths = dataset["ep_len"][:]

    selected = (
        np.arange(len(lengths), dtype=np.int64)
        if episode_ids is None
        else np.asarray(episode_ids, dtype=np.int64)
    )
    chunks = []
    required_span = frameskip * history_size
    for episode_id in selected:
        if episode_id < 0 or episode_id >= len(lengths):
            raise ValueError("episode_ids contains an out-of-range episode")
        count = int(lengths[episode_id]) - required_span
        if count > 0:
            start = int(offsets[episode_id])
            chunks.append(np.arange(start, start + count, dtype=np.int64))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


class LeWMSequenceDataset(Dataset):
    """Load macro-frame histories and normalized action blocks for LeWM."""

    def __init__(
        self,
        path,
        frameskip=5,
        history_size=3,
        sequence_indices=None,
        action_mean=None,
        action_std=None,
    ):
        self.path = Path(path)
        self.frameskip = int(frameskip)
        self.history_size = int(history_size)
        self._h5 = None
        self.sequence_indices = (
            build_lewm_sequence_indices(path, frameskip, history_size)
            if sequence_indices is None
            else np.asarray(sequence_indices, dtype=np.int64)
        )
        if action_mean is None or action_std is None:
            action_mean, action_std = compute_action_stats(path)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        if self.action_mean.shape != self.action_std.shape:
            raise ValueError("Action statistics must have matching shapes")
        if np.any(self.action_std <= 0):
            raise ValueError("Action standard deviation must be positive")

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, index):
        start = int(self.sequence_indices[index])
        frame_rows = start + np.arange(self.history_size + 1) * self.frameskip
        frames = np.asarray(self._dataset["pixels"][frame_rows])
        action_rows = start + np.arange(self.history_size)[:, None] * self.frameskip
        action_rows = action_rows + np.arange(self.frameskip)[None, :]
        actions = np.asarray(
            self._dataset["action"][action_rows.reshape(-1)], dtype=np.float32
        ).reshape(self.history_size, self.frameskip, -1)
        actions = (actions - self.action_mean) / self.action_std
        return (
            torch.from_numpy(frames.copy()).permute(0, 3, 1, 2),
            torch.from_numpy(actions.reshape(self.history_size, -1).copy()),
        )

    @property
    def _dataset(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()


class PushTTransitionDataset(Dataset):
    """Load frame pairs and normalized action blocks from a PushT HDF5 file.

    For a transition starting at row ``t`` and frameskip ``F``, an item contains
    ``pixels[t]``, ``pixels[t + F]``, and flattened ``action[t:t + F]``.
    The HDF5 file is opened lazily so each DataLoader worker gets its own handle.
    """

    def __init__(
        self,
        path,
        frameskip=5,
        transition_indices=None,
        action_mean=None,
        action_std=None,
    ):
        self.path = Path(path)
        self._h5 = None
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        if frameskip < 1:
            raise ValueError("frameskip must be at least 1")

        self.frameskip = int(frameskip)
        self.transition_indices = (
            build_transition_indices(self.path, self.frameskip)
            if transition_indices is None
            else np.asarray(transition_indices, dtype=np.int64)
        )
        if self.transition_indices.ndim != 1:
            raise ValueError("transition_indices must be a one-dimensional array")

        if action_mean is None or action_std is None:
            if action_mean is not None or action_std is not None:
                raise ValueError("action_mean and action_std must be provided together")
            action_mean, action_std = compute_action_stats(self.path)

        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        if self.action_mean.ndim != 1 or self.action_std.ndim != 1:
            raise ValueError("action_mean and action_std must be one-dimensional")
        if self.action_mean.shape != self.action_std.shape:
            raise ValueError("action_mean and action_std must have matching shapes")
        if not np.isfinite(self.action_mean).all():
            raise ValueError("action_mean must contain only finite values")
        if not np.isfinite(self.action_std).all() or np.any(self.action_std <= 0):
            raise ValueError("action_std must contain finite, positive values")

        with h5py.File(self.path, "r") as dataset:
            missing_keys = {"pixels", "action"} - set(dataset.keys())
            if missing_keys:
                raise ValueError(
                    f"Dataset is missing required keys: {sorted(missing_keys)}"
                )

            action_dim = int(dataset["action"].shape[-1])
            if self.action_mean.shape != (action_dim,):
                raise ValueError(
                    f"Expected action statistics with shape {(action_dim,)}, "
                    f"got {self.action_mean.shape}"
                )

        self.action_dim = action_dim
        self.action_block_dim = self.frameskip * self.action_dim

    def __len__(self):
        return len(self.transition_indices)

    def __getitem__(self, index):
        row = int(self.transition_indices[index])
        future_row = row + self.frameskip

        current_frame = np.asarray(self._dataset["pixels"][row])
        future_frame = np.asarray(self._dataset["pixels"][future_row])
        actions = np.asarray(
            self._dataset["action"][row:future_row], dtype=np.float32
        )

        expected_action_shape = (self.frameskip, self.action_dim)
        if actions.shape != expected_action_shape:
            raise RuntimeError(
                f"Expected action block shape {expected_action_shape}, "
                f"got {actions.shape} at row {row}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"Action block at row {row} contains NaN or infinity")

        normalized_actions = (actions - self.action_mean) / self.action_std

        current_tensor = torch.from_numpy(current_frame.copy()).permute(2, 0, 1)
        future_tensor = torch.from_numpy(future_frame.copy()).permute(2, 0, 1)
        action_tensor = torch.from_numpy(
            normalized_actions.reshape(self.action_block_dim).copy()
        )

        return current_tensor, future_tensor, action_tensor

    @property
    def _dataset(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getstate__(self):
        """Drop the open file handle before spawning a DataLoader worker."""
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()
