from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ENCODER_CHANNELS = (64, 128, 256)


class BasicBlock(nn.Module):
    """Two convolutional layers with a residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = self.relu(output + identity)
        return output


class PixelEncoder(nn.Module):
    """Convert a batch of RGB images into fixed-size feature vectors."""

    def __init__(self, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be positive")

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        blocks = []
        previous_channels = 32
        for output_channels in ENCODER_CHANNELS:
            blocks.append(
                BasicBlock(previous_channels, output_channels, stride=2)
            )
            previous_channels = output_channels

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.projection = nn.Linear(previous_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(start_dim=1)
        return self.projection(x)


class PixelTransformerIDM(nn.Module):
    """Infer a normalized action block from current and future RGB frames."""

    def __init__(
        self,
        raw_action_dim: int,
        frameskip: int,
        image_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 256,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_mlp_dim: int = 1024,
        dropout: float = 0.1,
        head_hidden_dims: tuple[int, ...] = (512, 512),
    ):
        super().__init__()

        if raw_action_dim < 1:
            raise ValueError("raw_action_dim must be positive")
        if frameskip < 1:
            raise ValueError("frameskip must be positive")
        if image_size < 1:
            raise ValueError("image_size must be positive")
        if in_channels != 3:
            raise ValueError("PixelTransformerIDM currently expects three RGB channels")
        if embed_dim < 1:
            raise ValueError("embed_dim must be positive")
        if transformer_layers < 1 or transformer_heads < 1:
            raise ValueError("Transformer layers and heads must be positive")
        if embed_dim % transformer_heads != 0:
            raise ValueError("embed_dim must be divisible by transformer_heads")
        if transformer_mlp_dim < 1:
            raise ValueError("transformer_mlp_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.raw_action_dim = int(raw_action_dim)
        self.frameskip = int(frameskip)
        self.action_block_dim = self.raw_action_dim * self.frameskip
        self.in_channels = int(in_channels)

        self.config = {
            "model_type": "PixelTransformerIDM",
            "raw_action_dim": self.raw_action_dim,
            "frameskip": self.frameskip,
            "action_block_dim": self.action_block_dim,
            "image_size": int(image_size),
            "in_channels": self.in_channels,
            "embed_dim": int(embed_dim),
            "transformer_layers": int(transformer_layers),
            "transformer_heads": int(transformer_heads),
            "transformer_mlp_dim": int(transformer_mlp_dim),
            "dropout": float(dropout),
            "head_hidden_dims": list(head_hidden_dims),
        }

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        self.encoder = PixelEncoder(embed_dim, in_channels=in_channels)

        # The Transformer receives current, change, and future feature tokens.
        self.type_embedding = nn.Parameter(torch.zeros(1, 3, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers,
        )

        head_layers: list[nn.Module] = [nn.LayerNorm(embed_dim)]
        previous_dim = embed_dim
        for hidden_dim in head_hidden_dims:
            if hidden_dim < 1:
                raise ValueError("Every head hidden dimension must be positive")
            head_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        head_layers.append(nn.Linear(previous_dim, self.action_block_dim))
        self.head = nn.Sequential(*head_layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)

        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for module in (self.encoder.projection, *self.head.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config: dict) -> "PixelTransformerIDM":
        if config.get("model_type") != "PixelTransformerIDM":
            raise ValueError("Config does not describe a PixelTransformerIDM")

        return cls(
            raw_action_dim=config["raw_action_dim"],
            frameskip=config["frameskip"],
            image_size=config.get("image_size", 224),
            in_channels=config.get("in_channels", 3),
            embed_dim=config.get("embed_dim", 256),
            transformer_layers=config.get("transformer_layers", 3),
            transformer_heads=config.get("transformer_heads", 4),
            transformer_mlp_dim=config.get("transformer_mlp_dim", 1024),
            dropout=config.get("dropout", 0.1),
            head_hidden_dims=tuple(config.get("head_hidden_dims", (512, 512))),
        )

    def _validate_frames(
        self,
        current_frame: torch.Tensor,
        future_frame: torch.Tensor,
    ) -> None:
        if current_frame.ndim != 4 or future_frame.ndim != 4:
            raise ValueError("Frames must have shape (batch, channels, height, width)")
        if current_frame.shape != future_frame.shape:
            raise ValueError("Current and future frames must have matching shapes")
        if current_frame.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} image channels, "
                f"got {current_frame.shape[1]}"
            )

    def _encode(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.float() / 255.0
        frame = (frame - self.pixel_mean) / self.pixel_std
        return self.encoder(frame)

    def forward(
        self,
        current_frame: torch.Tensor,
        future_frame: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_frames(current_frame, future_frame)

        current_features = self._encode(current_frame)
        future_features = self._encode(future_frame)
        feature_change = future_features - current_features

        tokens = torch.stack(
            [current_features, feature_change, future_features],
            dim=1,
        )
        tokens = self.embedding_dropout(tokens + self.type_embedding)
        encoded_tokens = self.transformer(tokens)
        transition_features = encoded_tokens.mean(dim=1)
        return self.head(transition_features)


def load_idm_checkpoint(checkpoint_path, device="cuda"):
    """Load a trained IDM and the action statistics saved with it."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"IDM checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "model_config", "action_mean", "action_std"}
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"IDM checkpoint is missing keys: {sorted(missing)}")

    model = PixelTransformerIDM.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device).requires_grad_(False)

    action_mean = torch.as_tensor(checkpoint["action_mean"], dtype=torch.float32)
    action_std = torch.as_tensor(checkpoint["action_std"], dtype=torch.float32)
    expected_shape = (model.raw_action_dim,)
    if action_mean.shape != expected_shape or action_std.shape != expected_shape:
        raise ValueError(f"IDM action statistics must have shape {expected_shape}")
    if not torch.isfinite(action_mean).all():
        raise ValueError("IDM action mean must be finite")
    if not torch.isfinite(action_std).all() or torch.any(action_std <= 0):
        raise ValueError("IDM action standard deviation must be finite and positive")

    return model, action_mean, action_std
