"""Goal-conditioned CEM planning with a LeWM latent world model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CEMConfig:
    """Planning settings matching the original PushT LeWM evaluation."""

    horizon: int = 5
    candidates: int = 300
    iterations: int = 30
    elites: int = 30
    batch_size: int = 300
    initial_std: float = 1.0
    minimum_std: float = 1e-3

    def __post_init__(self):
        for name in ("horizon", "candidates", "iterations", "elites", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.elites > self.candidates:
            raise ValueError("elites cannot exceed candidates")
        if self.initial_std <= 0 or self.minimum_std < 0:
            raise ValueError("CEM standard deviations must be non-negative")


class LeWMCEMPlanner:
    """Optimize normalized PushT action blocks against a goal-image latent."""

    def __init__(self, model, action_low, action_high, config=None):
        if model.action_mean is None or model.action_std is None:
            raise ValueError("LeWM must be constructed with action statistics")
        self.model = model
        self.config = config or CEMConfig()

        action_low = torch.as_tensor(
            action_low, dtype=torch.float32, device=model.device
        )
        action_high = torch.as_tensor(
            action_high, dtype=torch.float32, device=model.device
        )
        if action_low.shape != (2,) or action_high.shape != (2,):
            raise ValueError("PushT action bounds must have shape (2,)")
        if torch.any(action_low >= action_high):
            raise ValueError("Every action lower bound must be below its upper bound")

        normalized_low = (action_low - model.action_mean) / model.action_std
        normalized_high = (action_high - model.action_mean) / model.action_std
        repeats = model.action_block_dim // 2
        if repeats * 2 != model.action_block_dim:
            raise ValueError("LeWM action block dimension must be divisible by 2")
        self.low = normalized_low.repeat(repeats)
        self.high = normalized_high.repeat(repeats)

    def _normalize_history(self, action_blocks) -> torch.Tensor:
        blocks = np.asarray(action_blocks, dtype=np.float32)
        if blocks.size == 0:
            return torch.empty(
                (0, self.model.action_block_dim),
                dtype=torch.float32,
                device=self.model.device,
            )
        if blocks.ndim != 3 or blocks.shape[1:] != (5, 2):
            raise ValueError("Action history must have shape (T,5,2)")
        tensor = torch.as_tensor(blocks, device=self.model.device)
        normalized = (tensor - self.model.action_mean) / self.model.action_std
        return normalized.flatten(start_dim=1)

    @torch.no_grad()
    def _costs(
        self,
        candidates: torch.Tensor,
        latent_history: torch.Tensor,
        action_history: torch.Tensor,
        goal_latent: torch.Tensor,
    ) -> torch.Tensor:
        costs = []
        for start in range(0, len(candidates), self.config.batch_size):
            actions = candidates[start : start + self.config.batch_size]
            batch = len(actions)
            latents = latent_history.unsqueeze(0).expand(batch, -1, -1)
            past_actions = action_history.unsqueeze(0).expand(batch, -1, -1)

            for step in range(self.config.horizon):
                action_inputs = torch.cat(
                    (past_actions, actions[:, step : step + 1]), dim=1
                )
                prediction = self.model.predict_next(
                    latents, action_inputs, actions_normalized=True
                )
                latents = torch.cat((latents, prediction.unsqueeze(1)), dim=1)
                latents = latents[:, -self.model.history_size :]
                past_actions = torch.cat(
                    (past_actions, actions[:, step : step + 1]), dim=1
                )
                past_actions = past_actions[:, -(latents.shape[1] - 1) :]

            cost = (prediction - goal_latent).square().mean(dim=-1)
            costs.append(cost)
        return torch.cat(costs)

    @torch.no_grad()
    def plan(self, frames, action_history, goal_frame, seed=0):
        """Return raw actions `(horizon,5,2)` and their predicted goal cost."""
        frame_array = np.asarray(frames)
        if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
            raise ValueError("Frame history must have shape (T,H,W,3)")
        if not 1 <= len(frame_array) <= self.model.history_size:
            raise ValueError("Frame history length is incompatible with LeWM")
        if len(action_history) != len(frame_array) - 1:
            raise ValueError("There must be one action block between adjacent frames")

        latent_history = self.model.encode_frames(frame_array)
        normalized_history = self._normalize_history(action_history)
        goal_latent = self.model.encode_frames(np.asarray(goal_frame)[None])[0]

        shape = (
            self.config.candidates,
            self.config.horizon,
            self.model.action_block_dim,
        )
        mean = torch.zeros(shape[1:], device=self.model.device)
        std = torch.full_like(mean, self.config.initial_std)
        generator = torch.Generator(device=self.model.device)
        generator.manual_seed(int(seed))

        for _ in range(self.config.iterations):
            noise = torch.randn(shape, generator=generator, device=self.model.device)
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidates.clamp_(self.low, self.high)
            candidates[0] = mean.clamp(self.low, self.high)
            costs = self._costs(
                candidates, latent_history, normalized_history, goal_latent
            )
            elite_indices = torch.topk(
                costs, k=self.config.elites, largest=False
            ).indices
            elites = candidates[elite_indices]
            mean = elites.mean(dim=0).clamp(self.low, self.high)
            std = elites.std(dim=0, unbiased=False).clamp_min(
                self.config.minimum_std
            )

        # The returned plan is the final distribution mean, so report its cost.
        best_cost = self._costs(
            mean.unsqueeze(0), latent_history, normalized_history, goal_latent
        )[0]

        raw = mean.reshape(self.config.horizon, 5, 2)
        raw = raw * self.model.action_std + self.model.action_mean
        raw_low = self.low[:2] * self.model.action_std + self.model.action_mean
        raw_high = self.high[:2] * self.model.action_std + self.model.action_mean
        raw.clamp_(raw_low, raw_high)
        return raw.cpu().numpy(), float(best_cost)
