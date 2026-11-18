"""Inference wrapper for the official pretrained PushT LeWM checkpoint."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from .lewm_core import ARPredictor, Embedder, JEPA, MLP


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _legacy_vit_hf(
    size="tiny",
    patch_size=14,
    image_size=224,
    pretrained=False,
    use_mask_token=False,
    **_,
):
    """Build the Hugging Face ViT layout used by the released checkpoint."""
    if pretrained:
        raise ValueError("The LeWM checkpoint config must set pretrained=false")
    sizes = {
        "tiny": {
            "hidden_size": 192,
            "num_hidden_layers": 12,
            "num_attention_heads": 3,
            "intermediate_size": 768,
        }
    }
    if size not in sizes:
        raise ValueError(f"Unsupported LeWM ViT size: {size!r}")

    from transformers import ViTConfig, ViTModel

    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        qkv_bias=True,
        **sizes[size],
    )
    return ViTModel(
        config,
        add_pooling_layer=False,
        use_mask_token=use_mask_token,
    )


def _register_checkpoint_targets() -> None:
    """Register the target names stored in the official Hydra config."""
    stable_worldmodel = sys.modules.setdefault(
        "stable_worldmodel", types.ModuleType("stable_worldmodel")
    )
    wm_package = sys.modules.setdefault(
        "stable_worldmodel.wm", types.ModuleType("stable_worldmodel.wm")
    )
    lewm_package = types.ModuleType("stable_worldmodel.wm.lewm")
    lewm_package.__path__ = []
    lewm_package.LeWM = JEPA
    module_package = types.ModuleType("stable_worldmodel.wm.lewm.module")
    module_package.Predictor = ARPredictor
    module_package.ARPredictor = ARPredictor
    module_package.ActionEncoder = Embedder
    module_package.Embedder = Embedder
    module_package.Projector = MLP
    module_package.MLP = MLP

    stable_worldmodel.wm = wm_package
    wm_package.lewm = lewm_package
    lewm_package.module = module_package
    sys.modules["stable_worldmodel.wm.lewm"] = lewm_package
    sys.modules["stable_worldmodel.wm.lewm.module"] = module_package

    # The published weights use the classic transformers.ViTModel module names.
    # Newer stable-pretraining releases build a different ViT implementation,
    # so resolving the saved Hydra target through this compatibility shim is
    # necessary for a strict (and therefore safe) checkpoint load.
    stable_pretraining = types.ModuleType("stable_pretraining")
    stable_pretraining.__path__ = []
    backbone_package = types.ModuleType("stable_pretraining.backbone")
    backbone_package.__path__ = []
    backbone_utils = types.ModuleType("stable_pretraining.backbone.utils")
    backbone_utils.vit_hf = _legacy_vit_hf
    stable_pretraining.backbone = backbone_package
    backbone_package.utils = backbone_utils
    sys.modules["stable_pretraining"] = stable_pretraining
    sys.modules["stable_pretraining.backbone"] = backbone_package
    sys.modules["stable_pretraining.backbone.utils"] = backbone_utils


def build_lewm_from_config(config_path, device="cpu"):
    """Construct a randomly initialized LeWM from an architecture config."""
    from hydra.utils import instantiate

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"LeWM config not found: {config_path}")
    _register_checkpoint_targets()
    with config_path.open() as file:
        config = json.load(file)
    model = instantiate(config).to(device)
    return model, config


class LeWMWorldModel:
    """Load LeWM and expose frame encoding and next-latent prediction."""

    def __init__(
        self,
        checkpoint_dir,
        device="cuda",
        action_mean=None,
        action_std=None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.action_mean = self._optional_stat(action_mean)
        self.action_std = self._optional_stat(action_std)
        if (self.action_mean is None) != (self.action_std is None):
            raise ValueError("action_mean and action_std must be provided together")
        if self.action_std is not None and torch.any(self.action_std <= 0):
            raise ValueError("action_std must be positive")

        self.model, self.config = self._load_model()
        self.history_size = int(self.config["predictor"]["num_frames"])
        self.action_block_dim = int(self.config["action_encoder"]["input_dim"])
        self.latent_dim = int(self.config["predictor"]["output_dim"])
        self.image_size = int(self.config["encoder"]["image_size"])

    def _optional_stat(self, value):
        if value is None:
            return None
        result = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if result.shape != (2,) or not torch.isfinite(result).all():
            raise ValueError("PushT action statistics must be finite arrays of shape (2,)")
        return result

    def _load_model(self):
        config_path = self.checkpoint_dir / "config.json"
        weights_path = self.checkpoint_dir / "weights.pt"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Expected config.json and weights.pt in {self.checkpoint_dir}"
            )

        model, config = build_lewm_from_config(config_path, device="cpu")
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device).requires_grad_(False)
        return model, config

    def _prepare_frames(self, frames) -> tuple[torch.Tensor, bool]:
        frames = torch.as_tensor(frames, device=self.device)
        unbatched = frames.ndim == 4
        if unbatched:
            frames = frames.unsqueeze(0)
        if frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError("Frames must have shape (T,H,W,3) or (B,T,H,W,3)")
        if tuple(frames.shape[-3:-1]) != (self.image_size, self.image_size):
            raise ValueError(
                f"LeWM expects {self.image_size}x{self.image_size} frames, "
                f"got {tuple(frames.shape[-3:-1])}"
            )
        frames = frames.float().div_(255.0).permute(0, 1, 4, 2, 3).contiguous()
        mean = IMAGENET_MEAN.to(self.device)
        std = IMAGENET_STD.to(self.device)
        return (frames - mean) / std, unbatched

    @torch.no_grad()
    def encode_frames(self, frames) -> torch.Tensor:
        """Encode RGB frames into `(B,T,192)` LeWM latents."""
        frames, unbatched = self._prepare_frames(frames)
        batch, length = frames.shape[:2]
        flat_frames = frames.reshape(batch * length, 3, self.image_size, self.image_size)
        output = self.model.encoder(
            flat_frames,
            interpolate_pos_encoding=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        latents = self.model.projector(output.last_hidden_state[:, 0])
        latents = latents.reshape(batch, length, -1)
        return latents[0] if unbatched else latents

    def normalize_action_blocks(self, action_blocks) -> torch.Tensor:
        actions = torch.as_tensor(
            action_blocks, dtype=torch.float32, device=self.device
        )
        if self.action_mean is None or self.action_std is None:
            raise ValueError("Action statistics are required for raw action blocks")
        if actions.ndim != 4 or actions.shape[-2:] != (5, 2):
            raise ValueError("Raw action blocks must have shape (B,T,5,2)")
        normalized = (actions - self.action_mean) / self.action_std
        return normalized.flatten(start_dim=2)

    @torch.no_grad()
    def predict_next(
        self,
        latent_history: torch.Tensor,
        action_blocks,
        *,
        actions_normalized: bool = False,
    ) -> torch.Tensor:
        """Predict the next latent from `(B,T,D)` history and action blocks."""
        latents = torch.as_tensor(
            latent_history, dtype=torch.float32, device=self.device
        )
        if latents.ndim != 3:
            raise ValueError("latent_history must have shape (B,T,D)")
        batch, length, dimension = latents.shape
        if not 1 <= length <= self.history_size or dimension != self.latent_dim:
            raise ValueError(
                f"Expected latent history (B,1..{self.history_size},{self.latent_dim})"
            )

        if actions_normalized:
            action_tensor = torch.as_tensor(
                action_blocks, dtype=torch.float32, device=self.device
            )
            if action_tensor.shape != (batch, length, self.action_block_dim):
                raise ValueError(
                    f"Normalized actions must have shape "
                    f"{(batch, length, self.action_block_dim)}"
                )
        else:
            action_tensor = self.normalize_action_blocks(action_blocks)
            if action_tensor.shape[:2] != (batch, length):
                raise ValueError("Action history must match latent batch and time axes")

        action_embeddings = self.model.action_encoder(action_tensor)
        predictions = self.model.predict(latents, action_embeddings)
        return predictions[:, -1]

    @staticmethod
    def disagreement(predicted_latents, subgoal_latents) -> torch.Tensor:
        predicted = torch.as_tensor(predicted_latents)
        subgoals = torch.as_tensor(subgoal_latents, device=predicted.device)
        if predicted.shape != subgoals.shape:
            raise ValueError("Predicted and subgoal latent shapes must match")
        return (predicted - subgoals).square().mean(dim=-1)
