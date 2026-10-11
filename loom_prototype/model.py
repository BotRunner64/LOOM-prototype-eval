"""Small spatial visual encoder + independent Transformer flow-matching head.

Action convention and flow direction follow bq_native_policy; single-arm adaptation.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PolicyConfig:
    chunk_size: int = 16
    context_frames: int = 16
    hidden_dim: int = 192
    num_heads: int = 6
    num_layers: int = 3
    translation_scale: float = 0.1
    denoise_steps: int = 10
    context_dropout: float = 0.5

    def __post_init__(self):
        sizes = (
            self.chunk_size,
            self.context_frames,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.denoise_steps,
        )
        if any(type(v) is not int or v < 1 for v in sizes):
            raise ValueError("Positive integer dimensions required")
        if self.hidden_dim % self.num_heads or self.hidden_dim % 2:
            raise ValueError("Invalid hidden/head dimensions")
        if (
            not 0 < self.context_dropout < 1
            or not math.isfinite(self.translation_scale)
            or self.translation_scale <= 0
        ):
            raise ValueError(
                "Require context/no-context training and finite positive scale"
            )


class ContextPolicy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = PolicyConfig() if config is None else config
        self.config = config
        d = config.hidden_dim
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, d, 3, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.patch_position = nn.Parameter(torch.randn(16, d) * 0.02)
        self.context_time = nn.Parameter(torch.randn(config.context_frames, d) * 0.02)
        self.kind = nn.Parameter(torch.randn(2, d) * 0.02)
        # Fixed instruction embedding: this prototype has exactly one fixed task.
        self.instruction = nn.Parameter(torch.randn(1, d) * 0.02)
        self.state_projection = nn.Sequential(
            nn.Linear(10, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.calibration_projection = nn.Sequential(
            nn.Linear(20, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.action_projection = nn.Linear(10, d)
        self.time_projection = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.action_position = nn.Parameter(torch.randn(config.chunk_size, d) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d,
                    config.num_heads,
                    4 * d,
                    dropout=0,
                    activation=F.silu,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 10))
        center = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5], dtype=torch.float32)
        scale = torch.tensor(
            [config.translation_scale] * 3 + [1] * 6 + [0.5], dtype=torch.float32
        )
        self.register_buffer("action_center", center)
        self.register_buffer("action_scale", scale)
        self.register_buffer(
            "state_center", torch.tensor([0.45, 0, 0.25, 1, 0, 0, 0, 1, 0, 0.5])
        )
        self.register_buffer(
            "state_scale", torch.tensor([0.3, 0.3, 0.3, 1, 1, 1, 1, 1, 1, 0.5])
        )
        self.register_buffer(
            "frequencies", torch.exp(torch.linspace(0, math.log(1000), d // 2))
        )

    def normalize(self, action):
        return (action - self.action_center) / self.action_scale

    def normalize_state(self, state):
        return (state - self.state_center) / self.state_scale

    def encode(self, batch):
        b = len(batch["rgb"])
        c = self.config.context_frames
        if batch["context_rgb"].shape[1] != c:
            raise ValueError("Wrong context length")
        images = torch.cat([batch["rgb"][:, None], batch["context_rgb"]], 1)
        features = (
            self.visual(images.flatten(0, 1) * 2 - 1)
            .flatten(2)
            .transpose(1, 2)
            .reshape(b, c + 1, 16, -1)
        )
        features = features + self.patch_position
        current = features[:, 0] + self.kind[0]
        ctx_state = torch.cat(
            [
                self.normalize_state(batch["context_state"]),
                self.normalize(batch["context_action"]),
            ],
            -1,
        )
        reference = (
            features[:, 1:]
            + self.calibration_projection(ctx_state)[:, :, None]
            + self.context_time[None, :, None]
            + self.kind[1]
        )
        state = self.state_projection(self.normalize_state(batch["state"]))[:, None]
        instruction = self.instruction[None].expand(b, -1, -1)
        memory = torch.cat([current, state, instruction, reference.flatten(1, 2)], 1)
        mask = torch.cat(
            [
                torch.ones(b, 18, dtype=torch.bool, device=memory.device),
                batch["context_mask"].repeat_interleave(16, dim=1),
            ],
            1,
        )
        return memory, mask

    def velocity(self, action, time, memory, mask, action_mask):
        phase = time[:, None] * self.frequencies[None] * 2 * math.pi
        t = self.time_projection(torch.cat([phase.sin(), phase.cos()], -1))
        tokens = self.action_projection(action) + self.action_position + t[:, None]
        for layer in self.layers:
            tokens = layer(
                tokens,
                memory,
                tgt_key_padding_mask=~action_mask,
                memory_key_padding_mask=~mask,
            )
        return self.output(tokens)

    def forward(self, batch, drop_context=True):
        if not batch["action_mask"].any(dim=1).all():
            raise ValueError("Each sample requires a valid target")
        memory, mask = self.encode(batch)
        if drop_context:
            keep = (
                torch.rand(len(mask), device=mask.device) >= self.config.context_dropout
            )
            mask = mask.clone()
            mask[:, 18:] &= keep[:, None]
        target = self.normalize(batch["actions"])
        valid = batch["action_mask"]
        target = torch.where(valid[:, :, None], target, 0)
        noise = torch.randn_like(target)
        time = torch.rand(len(target), device=target.device)
        noisy = (1 - time[:, None, None]) * target + time[:, None, None] * noise
        noisy = torch.where(valid[:, :, None], noisy, 0)
        prediction = self.velocity(noisy, time, memory, mask, valid)
        squared = (prediction.float() - (noise - target).float()).square()
        return torch.where(valid[:, :, None], squared, 0).sum() / (valid.sum() * 10)

    @torch.no_grad()
    def sample_actions(self, batch, generator=None):
        was_training = self.training
        self.eval()
        try:
            memory, mask = self.encode(batch)
            b = len(memory)
            device = memory.device
            action = torch.randn(
                b, self.config.chunk_size, 10, device=device, generator=generator
            )
            valid = torch.ones(
                b, self.config.chunk_size, dtype=torch.bool, device=device
            )
            for i in range(self.config.denoise_steps):
                time = torch.full(
                    (b,), 1 - i / self.config.denoise_steps, device=device
                )
                action -= (
                    self.velocity(action, time, memory, mask, valid)
                    / self.config.denoise_steps
                )
            return project_actions(action * self.action_scale + self.action_center)
        finally:
            self.train(was_training)


def project_actions(action):
    action = action.clone()
    a = action[..., 3:6]
    b = action[..., 6:9]
    fallback = torch.zeros_like(a)
    fallback[..., 0] = 1
    a = torch.where(
        a.norm(dim=-1, keepdim=True) > 1e-6, F.normalize(a, dim=-1), fallback
    )
    b = b - (a * b).sum(-1, keepdim=True) * a
    fb = F.one_hot(a.abs().argmin(-1), 3).to(a.dtype)
    fb = fb - (a * fb).sum(-1, keepdim=True) * a
    b = torch.where(b.norm(dim=-1, keepdim=True) > 1e-6, b, fb)
    action[..., 3:9] = torch.cat([a, F.normalize(b, dim=-1)], -1)
    action[..., 9].clamp_(0, 1)
    return action
