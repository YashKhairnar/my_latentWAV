from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class NanoWMSubgoalGenerator:
    """Generate plausible future RGB frames with a pretrained NanoWM model."""

    def __init__(
        self,
        repo_path,
        checkpoint_dir,
        device,
        num_sampling_steps,
    ):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.output_size = 224
        self.num_sampling_steps = int(num_sampling_steps)

        if self.num_sampling_steps <= 0:
            raise ValueError("num_sampling_steps must be greater than zero.")

        self._import_nanowm()
        self._load_config()
        self._load_model()
        self._load_codec()
        self._load_diffusion()

    def _import_nanowm(self):
        """Import the NanoWM inference API from the cloned repository."""
        nanowm_src = self.repo_path / "src"
        if not nanowm_src.is_dir():
            raise FileNotFoundError(
                f"NanoWM source directory not found: {nanowm_src}"
            )

        src_string = str(nanowm_src)
        if src_string not in sys.path:
            sys.path.insert(0, src_string)

        try:
            from diffusion import create_diffusion
            from diffusion.df_sample import dfot_sample
            from latent_codecs import (
                SDVAELatentCodec,
                get_model_latent_channels,
                get_model_latent_size,
                load_autoencoder_kl,
                resolve_latent_codec_config,
            )
            from models import get_models
        except ImportError as exc:
            raise ImportError(f"Failed to import NanoWM from {nanowm_src}.") from exc

        self.get_models = get_models
        self.create_diffusion = create_diffusion
        self.dfot_sample = dfot_sample
        self.SDVAELatentCodec = SDVAELatentCodec
        self.get_model_latent_channels = get_model_latent_channels
        self.get_model_latent_size = get_model_latent_size
        self.load_autoencoder_kl = load_autoencoder_kl
        self.resolve_latent_codec_config = resolve_latent_codec_config

    def _load_config(self):
        """Load the checkpoint configuration and resolve model dimensions."""
        from omegaconf import OmegaConf

        config_path = self.checkpoint_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"NanoWM config not found: {config_path}")

        self.config = OmegaConf.load(config_path)
        self.config.experiment.infra.compile = False

        self.num_frames = int(self.config.model.num_frames)
        self.image_size = int(self.config.model.image_size)
        self.latent_size = int(self.get_model_latent_size(self.config))
        self.latent_channels = int(self.get_model_latent_channels(self.config))
        self.codec_config = self.resolve_latent_codec_config(self.config)
        self.scheduling_mode = str(self.config.model.scheduling_mode)
        self.history_stabilization_level = float(
            self.config.experiment.diffusion.history_stabilization_level
        )

        if not self.codec_config.has_decoder:
            raise ValueError(
                f"Codec {self.codec_config.kind!r} cannot generate RGB subgoals."
            )

        if bool(self.config.model.use_action):
            warnings.warn(
                "This NanoWM checkpoint was trained with action conditioning, but "
                "subgoal generation runs without actions. Generated futures may "
                "therefore differ from the checkpoint's training distribution.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _load_model(self):
        """Load the NanoWM architecture and pretrained parameters."""
        from safetensors.torch import load_file

        weights_path = self.checkpoint_dir / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"NanoWM weights not found: {weights_path}")

        model = self.get_models(self.config)
        state_dict = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        self.model = model

    def _load_codec(self):
        """Load the SD-VAE used to encode and decode NanoWM latents."""
        vae = self.load_autoencoder_kl(self.codec_config.model_path)
        vae = vae.to(self.device)
        vae.eval()
        vae.requires_grad_(False)

        self.codec = self.SDVAELatentCodec(
            vae=vae,
            precision=self.codec_config.precision,
            latent_shape=self.codec_config.latent_shape,
        )

    def _load_diffusion(self):
        """Create the respaced diffusion process used during inference."""
        diffusion_config = self.config.experiment.diffusion
        self.diffusion = self.create_diffusion(
            timestep_respacing=str(self.num_sampling_steps),
            noise_schedule=str(diffusion_config.noise_schedule),
            pred_name=str(diffusion_config.pred_name),
            diffusion_steps=int(diffusion_config.diffusion_steps),
            snr_gamma=float(diffusion_config.snr_gamma),
            zero_terminal_snr=bool(diffusion_config.zero_terminal_snr),
        )

    def _validate_context(self, context):
        """Validate an ``(L, H, W, 3)`` uint8 context array."""
        if not isinstance(context, np.ndarray):
            raise TypeError("context must be a NumPy array.")
        if context.dtype != np.uint8:
            raise TypeError(f"context must have dtype uint8, received {context.dtype}.")
        if context.ndim != 4:
            raise ValueError(
                f"Expected context shape (L, H, W, 3), received {context.shape}."
            )
        if context.shape[0] == 0:
            raise ValueError("context must contain at least one frame.")
        if context.shape[-1] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, received {context.shape[-1]}."
            )

    def _validate_num_subgoals(self, num_subgoals):
        """Validate the requested number of candidate futures."""
        if type(num_subgoals) is not int:
            raise TypeError("num_subgoals must be an integer.")
        if num_subgoals <= 0:
            raise ValueError("num_subgoals must be greater than zero.")

    def _prepare_context(self, context):
        """Convert uint8 RGB context frames to normalized model inputs."""
        self._validate_context(context)

        prepared = torch.from_numpy(np.ascontiguousarray(context))
        prepared = prepared.to(self.device)
        prepared = prepared.permute(0, 3, 1, 2).float()
        prepared = prepared.div(255.0).mul(2.0).sub(1.0)
        return F.interpolate(
            prepared,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def _encode_context(self, context):
        """Encode RGB context frames into SD-VAE latents."""
        latent = self.codec.encode(self._prepare_context(context))
        expected_shape = (
            context.shape[0],
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        if tuple(latent.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected latent shape {tuple(latent.shape)}; "
                f"expected {expected_shape}."
            )
        return latent

    @torch.no_grad()
    def _decode_subgoals(self, subgoals):
        """Decode NanoWM latents into 224x224 uint8 RGB subgoal frames."""
        if not isinstance(subgoals, torch.Tensor):
            raise TypeError("subgoals must be a PyTorch tensor.")
        if subgoals.ndim != 4:
            raise ValueError(
                "Expected subgoal latent shape (K, C, H, W), "
                f"received {tuple(subgoals.shape)}."
            )

        expected_tail = (
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        if tuple(subgoals.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected subgoal latent dimensions {expected_tail}, "
                f"received {tuple(subgoals.shape[1:])}."
            )

        frames = self.codec.decode(subgoals.to(self.device))
        frames = frames.add(1.0).div(2.0).clamp(0.0, 1.0)
        frames = F.interpolate(
            frames,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        frames = frames.mul(255.0).round().clamp(0, 255).to(torch.uint8)
        return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    @torch.no_grad()
    def sample(self, context, num_subgoals):
        """Generate ``num_subgoals`` plausible next-frame candidates.

        Args:
            context: Context frames shaped ``(L, H, W, 3)`` as uint8 RGB.
            num_subgoals: Number of candidate future frames to generate.

        Returns:
            A uint8 NumPy array shaped ``(K, 224, 224, 3)``.
        """
        self._validate_context(context)
        self._validate_num_subgoals(num_subgoals)

        num_context_frames = int(context.shape[0])
        if num_context_frames >= self.num_frames:
            raise ValueError(
                f"Context has {num_context_frames} frames, but NanoWM supports "
                f"at most {self.num_frames - 1} when generating a future frame."
            )

        context_latents = self._encode_context(context)
        batched_context = (
            context_latents.unsqueeze(0)
            .expand(num_subgoals, *context_latents.shape)
            .contiguous()
        )
        output_shape = (
            num_subgoals,
            self.num_frames,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )

        sampled_video = self.dfot_sample(
            diffusion=self.diffusion,
            model=self.model,
            shape=output_shape,
            context=batched_context,
            n_context_frames=num_context_frames,
            scheduling_mode=self.scheduling_mode,
            num_sampling_steps=self.num_sampling_steps,
            model_kwargs={"y": None},
            device=self.device,
            progress=False,
            eta=0.0,
            clip_denoised=False,
            n_generate_frames=1,
            history_stabilization_level=self.history_stabilization_level,
        )

        expected_video_shape = output_shape
        if tuple(sampled_video.shape) != expected_video_shape:
            raise RuntimeError(
                f"Unexpected sampled video shape {tuple(sampled_video.shape)}; "
                f"expected {expected_video_shape}."
            )

        first_future_latent = sampled_video[:, num_context_frames]
        return self._decode_subgoals(first_future_latent)
