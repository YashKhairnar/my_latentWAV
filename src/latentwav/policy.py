"""Stateful single-environment LatentWAV inference policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .verifier import VerificationResult


@dataclass(frozen=True)
class PolicyDecision:
    """Artifacts created at one LatentWAV macro-step."""

    frame_history: np.ndarray
    subgoal_frames: np.ndarray
    verification: VerificationResult

    @property
    def action_block(self) -> np.ndarray:
        return self.verification.selected_action.numpy().copy()


class LatentWAVPolicy:
    """Generate and execute one verified action block every ``frameskip`` steps.

    ``act(frame)`` is called once per environment step. It invokes NanoWM, the
    IDM, and LeWM only when the previous action block has been exhausted. The
    intervening calls return the remaining actions from the cached block.

    This first implementation intentionally handles one environment. Batching
    multiple environments can be added after the full loop is correct.
    """

    def __init__(self, subgoal_generator, verifier, num_subgoals=8):
        if not isinstance(num_subgoals, int) or num_subgoals < 1:
            raise ValueError("num_subgoals must be a positive integer")

        self.subgoal_generator = subgoal_generator
        self.verifier = verifier
        self.num_subgoals = num_subgoals
        self.history_size = int(verifier.world_model.history_size)
        self.frameskip = int(verifier.frameskip)
        self.raw_action_dim = int(verifier.raw_action_dim)

        nanowm_frames = getattr(subgoal_generator, "num_frames", None)
        if nanowm_frames is not None and self.history_size >= int(nanowm_frames):
            raise ValueError(
                "LeWM history must leave at least one NanoWM frame for generation"
            )

        self.reset()

    def reset(self) -> None:
        """Clear all episode-specific frames, actions, and cached decisions."""
        self._frames = deque(maxlen=self.history_size)
        self._previous_action_blocks = deque(maxlen=max(self.history_size - 1, 0))
        self._action_buffer = deque()
        self._completed_block = None
        self.last_decision = None
        self.macro_steps = 0
        self.environment_steps = 0

    @property
    def needs_plan(self) -> bool:
        return len(self._action_buffer) == 0

    @property
    def buffered_actions(self) -> int:
        return len(self._action_buffer)

    def _validate_frame(self, frame) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if frame.dtype != np.uint8:
            raise TypeError(f"frame must have dtype uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("frame must have shape (H,W,3)")
        return np.ascontiguousarray(frame)

    def _commit_macro_observation(self, frame: np.ndarray) -> None:
        """Add the new boundary frame and the block that led to it."""
        if self._completed_block is not None:
            self._previous_action_blocks.append(self._completed_block)
            self._completed_block = None
        self._frames.append(frame.copy())

        expected_actions = max(len(self._frames) - 1, 0)
        if len(self._previous_action_blocks) != expected_actions:
            raise RuntimeError("Frame and action histories became misaligned")

    def _plan(self, frame: np.ndarray) -> None:
        self._commit_macro_observation(frame)
        frame_history = np.stack(tuple(self._frames))
        subgoal_frames = self.subgoal_generator.sample(
            frame_history,
            self.num_subgoals,
        )
        if not isinstance(subgoal_frames, np.ndarray):
            raise TypeError("Subgoal generator must return a NumPy array")
        if subgoal_frames.dtype != np.uint8:
            raise TypeError("Subgoal generator must return uint8 frames")
        expected_subgoal_shape = (self.num_subgoals, *frame.shape)
        if subgoal_frames.shape != expected_subgoal_shape:
            raise ValueError(
                "Subgoal generator returned shape "
                f"{subgoal_frames.shape}, expected {expected_subgoal_shape}"
            )

        previous_blocks = None
        if len(self._previous_action_blocks):
            previous_blocks = np.stack(tuple(self._previous_action_blocks))
        verification = self.verifier.score_candidates(
            frame_history,
            subgoal_frames,
            previous_action_blocks=previous_blocks,
        )
        selected_block = verification.selected_action.numpy()
        expected_action_shape = (self.frameskip, self.raw_action_dim)
        if selected_block.shape != expected_action_shape:
            raise RuntimeError(
                f"Verifier selected {selected_block.shape}, "
                f"expected {expected_action_shape}"
            )
        if not np.isfinite(selected_block).all():
            raise RuntimeError("Verifier selected an action block containing NaN/inf")

        selected_block = selected_block.astype(np.float32, copy=True)
        self._action_buffer.extend(selected_block)
        self._completed_block = selected_block
        self.last_decision = PolicyDecision(
            frame_history=frame_history.copy(),
            subgoal_frames=subgoal_frames.copy(),
            verification=verification,
        )
        self.macro_steps += 1

    def act(self, frame) -> np.ndarray:
        """Return the next raw environment action for the supplied RGB frame."""
        frame = self._validate_frame(frame)
        if self.needs_plan:
            self._plan(frame)

        action = np.asarray(self._action_buffer.popleft(), dtype=np.float32).copy()
        if action.shape != (self.raw_action_dim,):
            raise RuntimeError("Internal action buffer contains an invalid action")
        self.environment_steps += 1
        return action
