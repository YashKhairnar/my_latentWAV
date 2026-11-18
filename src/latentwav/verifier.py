"""Forward-inverse disagreement scoring for LatentWAV action candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VerificationResult:
    """CPU tensors produced while choosing one candidate action block."""

    candidate_actions: torch.Tensor
    predicted_latents: torch.Tensor
    subgoal_latents: torch.Tensor
    scores: torch.Tensor
    selected_index: int

    @property
    def selected_action(self) -> torch.Tensor:
        return self.candidate_actions[self.selected_index]

    @property
    def selected_score(self) -> float:
        return float(self.scores[self.selected_index])


class ActionVerifier:
    """Use the IDM and LeWM to select a high-disagreement action candidate.

    This initial version scores one environment at a time. The frame history may
    contain between one and ``LeWMWorldModel.history_size`` frames.
    """

    def __init__(
        self,
        idm,
        world_model,
        idm_action_mean,
        idm_action_std,
        action_low,
        action_high,
    ):
        self.idm = idm
        self.world_model = world_model
        self.idm_device = next(idm.parameters()).device
        self.frameskip = int(idm.frameskip)
        self.raw_action_dim = int(idm.raw_action_dim)

        if self.frameskip * self.raw_action_dim != world_model.action_block_dim:
            raise ValueError("IDM and LeWM action-block dimensions do not match")

        self.idm_action_mean = self._action_vector(
            idm_action_mean, "idm_action_mean", positive=False
        ).to(self.idm_device)
        self.idm_action_std = self._action_vector(
            idm_action_std, "idm_action_std", positive=True
        ).to(self.idm_device)
        self.action_low = self._action_vector(
            action_low, "action_low", positive=False
        ).to(self.idm_device)
        self.action_high = self._action_vector(
            action_high, "action_high", positive=False
        ).to(self.idm_device)
        if torch.any(self.action_low > self.action_high):
            raise ValueError("Every action_low value must be <= action_high")

    def _action_vector(self, value, name, *, positive):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape != (self.raw_action_dim,):
            raise ValueError(
                f"{name} must have shape {(self.raw_action_dim,)}, "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        if positive and torch.any(tensor <= 0):
            raise ValueError(f"{name} must contain only positive values")
        return tensor

    def _prepare_idm_frames(self, frames, name):
        tensor = torch.as_tensor(frames, device=self.idm_device)
        if tensor.ndim != 4 or tensor.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (N,H,W,3)")
        return tensor.permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def score_candidates(
        self,
        frame_history,
        subgoal_frames,
        previous_action_blocks=None,
    ) -> VerificationResult:
        """Infer, score, and select candidate action blocks.

        Args:
            frame_history: RGB uint8 array with shape ``(T,H,W,3)``.
            subgoal_frames: ``K`` proposed RGB frames with shape ``(K,H,W,3)``.
            previous_action_blocks: The ``T-1`` already executed raw action
                blocks, shaped ``(T-1,frameskip,action_dim)``. It is omitted
                only when ``T == 1``.
        """
        history = torch.as_tensor(frame_history)
        subgoals = torch.as_tensor(subgoal_frames)
        if history.ndim != 4 or history.shape[-1] != 3:
            raise ValueError("frame_history must have shape (T,H,W,3)")
        if subgoals.ndim != 4 or subgoals.shape[-1] != 3:
            raise ValueError("subgoal_frames must have shape (K,H,W,3)")
        if history.shape[1:] != subgoals.shape[1:]:
            raise ValueError("History and subgoal frames must have matching sizes")

        history_length = int(history.shape[0])
        num_candidates = int(subgoals.shape[0])
        if not 1 <= history_length <= self.world_model.history_size:
            raise ValueError(
                f"History must contain 1..{self.world_model.history_size} frames"
            )
        if num_candidates < 1:
            raise ValueError("At least one subgoal candidate is required")

        current = history[-1:].expand(num_candidates, -1, -1, -1)
        current_chw = self._prepare_idm_frames(current, "current frames")
        subgoal_chw = self._prepare_idm_frames(subgoals, "subgoal_frames")
        normalized_actions = self.idm(current_chw, subgoal_chw)
        expected = (num_candidates, self.frameskip * self.raw_action_dim)
        if normalized_actions.shape != expected:
            raise RuntimeError(
                f"IDM returned {tuple(normalized_actions.shape)}, expected {expected}"
            )

        candidate_actions = normalized_actions.reshape(
            num_candidates, self.frameskip, self.raw_action_dim
        )
        candidate_actions = (
            candidate_actions * self.idm_action_std + self.idm_action_mean
        )
        candidate_actions = torch.maximum(candidate_actions, self.action_low)
        candidate_actions = torch.minimum(candidate_actions, self.action_high)

        expected_previous_shape = (
            history_length - 1,
            self.frameskip,
            self.raw_action_dim,
        )
        if previous_action_blocks is None:
            if history_length != 1:
                raise ValueError(
                    "previous_action_blocks is required when history has multiple frames"
                )
            previous = torch.empty(expected_previous_shape, dtype=torch.float32)
        else:
            previous = torch.as_tensor(previous_action_blocks, dtype=torch.float32)
            if previous.shape != expected_previous_shape:
                raise ValueError(
                    "previous_action_blocks must have shape "
                    f"{expected_previous_shape}, got {tuple(previous.shape)}"
                )
            if not torch.isfinite(previous).all():
                raise ValueError("previous_action_blocks must be finite")

        world_device = self.world_model.device
        action_histories = torch.empty(
            num_candidates,
            history_length,
            self.frameskip,
            self.raw_action_dim,
            dtype=torch.float32,
            device=world_device,
        )
        if history_length > 1:
            action_histories[:, :-1] = previous.to(world_device).unsqueeze(0)
        action_histories[:, -1] = candidate_actions.to(world_device)

        history_latents = self.world_model.encode_frames(history)
        subgoal_latents = self.world_model.encode_frames(subgoals)
        repeated_history = history_latents.unsqueeze(0).expand(
            num_candidates, -1, -1
        )
        predicted_latents = self.world_model.predict_next(
            repeated_history,
            action_histories,
        )
        scores = self.world_model.disagreement(predicted_latents, subgoal_latents)
        selected_index = int(torch.argmax(scores).item())

        return VerificationResult(
            candidate_actions=candidate_actions.detach().cpu(),
            predicted_latents=predicted_latents.detach().cpu(),
            subgoal_latents=subgoal_latents.detach().cpu(),
            scores=scores.detach().cpu(),
            selected_index=selected_index,
        )
