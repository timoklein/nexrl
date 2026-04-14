"""SGF networks: VICReg-based world model with augmentation-driven representation learning.

"Simple, Good, Fast: Self-Supervised World Models Free of Baggage" (Robine et al., ICLR 2025)

Replaces Dreamer's RSSM with a simpler deterministic encoder + VICReg representation
learning. Uses image augmentations to model stochasticity instead of a stochastic
latent space. Trains actor-critic purely in imagination.

Dimension key:
    B: batch size
    C: input channels (frame_stack * grayscale channels)
    H: image height
    W: image width
    Y: encoder output dim (y_dim, default 512)
    Z: projector output dim (z_dim, default 2048)
    A: action dimension (flat one-hot stacked actions)
    N: num_bins (symlog two-hot)
"""

import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.dreamer import (
    ContinuePredictor,
    RewardPredictor,
)
from nexrl.networks.mrq import LNBlock


# ---------------------------------------------------------------------------
# Image augmentations (pure functions, operate on NCHW float tensors)
# ---------------------------------------------------------------------------


def random_shift(rng: jax.Array, obs_BCHW: jax.Array, pad: int = 3) -> jax.Array:
    """Pad with edge replication and random crop back to original size.

    Per-sample random shifts up to `pad` pixels in any direction.

    Args:
        rng: PRNG key
        obs_BCHW: (B, C, H, W) float tensor
        pad: number of pixels to pad on each side

    Returns:
        Shifted observations, same shape as input
    """
    B, C, H, W = obs_BCHW.shape
    # Pad with edge replication
    padded = jnp.pad(obs_BCHW, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
    # Random offsets in [0, pad] inclusive for each sample
    rng_h, rng_w = jax.random.split(rng)
    h_offsets = jax.random.randint(rng_h, (B,), 0, pad + 1)
    w_offsets = jax.random.randint(rng_w, (B,), 0, pad + 1)

    # Use vmap + dynamic_slice for JIT compatibility
    def _crop_single(padded_single, h_off, w_off):
        return jax.lax.dynamic_slice(padded_single, (0, h_off, w_off), (C, H, W))

    return jax.vmap(_crop_single)(padded, h_offsets, w_offsets)


def random_intensity(rng: jax.Array, obs_BCHW: jax.Array, scale: float = 0.05) -> jax.Array:
    """Multiply by 1 + scale * noise where noise ~ N(0,1) clipped to [-2, 2].

    Per-sample noise (same across channels and spatial dims).

    Args:
        rng: PRNG key
        obs_BCHW: (B, C, H, W) float tensor
        scale: noise scale

    Returns:
        Intensity-jittered observations, same shape as input
    """
    B = obs_BCHW.shape[0]
    noise = jax.random.normal(rng, (B, 1, 1, 1))
    noise = jnp.clip(noise, -2.0, 2.0)
    return obs_BCHW * (1.0 + scale * noise)


def augment(rng: jax.Array, obs_BCHW: jax.Array, shift_pad: int = 3, intensity_scale: float = 0.05) -> jax.Array:
    """Apply random shift + random intensity augmentation pipeline."""
    rng_shift, rng_int = jax.random.split(rng)
    obs_BCHW = random_shift(rng_shift, obs_BCHW, pad=shift_pad)
    obs_BCHW = random_intensity(rng_int, obs_BCHW, scale=intensity_scale)
    return obs_BCHW


# ---------------------------------------------------------------------------
# VICReg loss
# ---------------------------------------------------------------------------


def vicreg_loss(
    z_BZ: jax.Array,
    z_prime_BZ: jax.Array,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    eps: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
    """Variance-Invariance-Covariance regularization (Bardes et al., 2022).

    Applied symmetrically to both z and z'. Returns separate var and cov losses
    for metrics tracking. The invariance/similarity term is handled separately
    as temporal consistency (MSE between predictor output and sg(z')).

    Args:
        z_BZ: projected embeddings (B, Z)
        z_prime_BZ: projected embeddings from next observation (B, Z)
        var_weight: weight for variance regularization (rho in paper, default 25.0)
        cov_weight: weight for covariance regularization (nu in paper, default 1.0)
        eps: numerical stability epsilon

    Returns:
        (var_loss, cov_loss) — weighted scalar losses
    """
    D = z_BZ.shape[-1]

    def _vc_single(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Variance: hinge(1 - std(z, dim=0)) averaged over features
        std = jnp.sqrt(jnp.var(z, axis=0) + eps)
        var_loss = jnp.mean(jax.nn.relu(1.0 - std))
        # Covariance: sum of squared off-diagonal / D
        z_centered = z - jnp.mean(z, axis=0)
        N = z.shape[0]
        cov = (z_centered.T @ z_centered) / (N - 1)
        # Off-diagonal: mask out diagonal
        mask = 1.0 - jnp.eye(D)
        cov_loss = jnp.sum(mask * cov**2) / D
        return var_loss, cov_loss

    var1, cov1 = _vc_single(z_BZ)
    var2, cov2 = _vc_single(z_prime_BZ)

    var_loss = var_weight * (var1 + var2) / 2.0
    cov_loss = cov_weight * (cov1 + cov2) / 2.0
    return var_loss, cov_loss


# ---------------------------------------------------------------------------
# SGF Encoder (custom CNN with LN+SiLU)
# ---------------------------------------------------------------------------


def _conv_output_size(h: int, kernel: int, stride: int, padding: int) -> int:
    """Compute output spatial dim for convolution with explicit padding."""
    return (h + 2 * padding - kernel) // stride + 1


class SGFEncoder(nnx.Module):
    """SGF convolutional encoder with LayerNorm + SiLU throughout.

    Reference architecture (Robine et al., 2025, Appendix F):
        Conv(in, 64, 4, 2, 1) -> LN -> SiLU
        Conv(64, 128, 4, 2, 1) -> LN -> SiLU
        Conv(128, 256, 4, 2, 1) -> LN -> SiLU
        Conv(256, 64, 3, 1, 1) -> LN -> SiLU
        Flatten -> Linear(flat, y_dim) -> LN

    Final LayerNorm on output is critical for stabilizing representations
    (no activation after it).

    Input: (B, C, H, W) uint8 or float
    Output: (B, Y) float32 features
    """

    def __init__(self, in_channels: int, height: int, width: int, y_dim: int, rngs: nnx.Rngs):
        self._y_dim = y_dim
        # 4 conv layers — Flax nnx.Conv operates in NHWC internally
        # We use explicit SAME-like padding by specifying ((pad, pad), (pad, pad))
        self.conv1 = nnx.Conv(in_channels, 64, kernel_size=(4, 4), strides=(2, 2), padding=((1, 1), (1, 1)), rngs=rngs)
        self.ln1 = nnx.LayerNorm(64, epsilon=1e-3, rngs=rngs)
        self.conv2 = nnx.Conv(64, 128, kernel_size=(4, 4), strides=(2, 2), padding=((1, 1), (1, 1)), rngs=rngs)
        self.ln2 = nnx.LayerNorm(128, epsilon=1e-3, rngs=rngs)
        self.conv3 = nnx.Conv(128, 256, kernel_size=(4, 4), strides=(2, 2), padding=((1, 1), (1, 1)), rngs=rngs)
        self.ln3 = nnx.LayerNorm(256, epsilon=1e-3, rngs=rngs)
        self.conv4 = nnx.Conv(256, 64, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), rngs=rngs)
        self.ln4 = nnx.LayerNorm(64, epsilon=1e-3, rngs=rngs)

        # Compute flattened size after conv layers
        h = _conv_output_size(_conv_output_size(_conv_output_size(height, 4, 2, 1), 4, 2, 1), 4, 2, 1)
        w = _conv_output_size(_conv_output_size(_conv_output_size(width, 4, 2, 1), 4, 2, 1), 4, 2, 1)
        # Last conv: kernel=3, stride=1, padding=1 -> same spatial size
        flat_size = h * w * 64

        self.dense = nnx.Linear(flat_size, y_dim, rngs=rngs)
        self.ln_out = nnx.LayerNorm(y_dim, epsilon=1e-3, rngs=rngs)

    @property
    def features_dim(self) -> int:
        return self._y_dim

    def _apply_conv_ln(self, x_BHWC: jax.Array, conv: nnx.Conv, ln: nnx.LayerNorm) -> jax.Array:
        """Conv -> channel-wise LayerNorm -> SiLU (in NHWC layout)."""
        x = conv(x_BHWC)  # (B, H', W', C')
        # LayerNorm on last axis (channel dim) — nnx.LayerNorm does this by default
        x = ln(x)
        return nnx.silu(x)

    def __call__(self, obs_BCHW: jax.Array) -> jax.Array:
        """Encode observations.

        Args:
            obs_BCHW: (B, C, H, W) uint8 or float observations

        Returns:
            (B, Y) encoder features with LayerNorm applied
        """
        # NCHW -> NHWC, normalize uint8 to [0, 1]
        x = obs_BCHW.transpose(0, 2, 3, 1).astype(jnp.float32) / 255.0
        x = self._apply_conv_ln(x, self.conv1, self.ln1)
        x = self._apply_conv_ln(x, self.conv2, self.ln2)
        x = self._apply_conv_ln(x, self.conv3, self.ln3)
        x = self._apply_conv_ln(x, self.conv4, self.ln4)
        x = x.reshape(x.shape[0], -1)  # flatten spatial dims
        x = self.ln_out(self.dense(x))  # Linear -> LN (no activation on output)
        return x


# ---------------------------------------------------------------------------
# SGF MLPs (projector, predictor, transition predictor)
# ---------------------------------------------------------------------------


class SGFProjector(nnx.Module):
    """Projector: y -> z via MLP with LN+SiLU hidden layers.

    Reference: Linear(y_dim, hidden) -> LN -> SiLU -> Linear -> LN -> SiLU -> Linear(hidden, z_dim)
    Output head has no LayerNorm and no bias (reference: out_norm=False, out_bias=False).
    """

    def __init__(self, y_dim: int, z_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.block1 = LNBlock(y_dim, hidden_dim, nnx.silu, rngs)
        self.block2 = LNBlock(hidden_dim, hidden_dim, nnx.silu, rngs)
        self.output = nnx.Linear(hidden_dim, z_dim, use_bias=False, rngs=rngs)

    def __call__(self, y_BY: jax.Array) -> jax.Array:
        x = self.block1(y_BY)
        x = self.block2(x)
        return self.output(x)


class SGFPredictor(nnx.Module):
    """Predictor: (z, action) -> z_pred via MLP with LN+SiLU.

    Same architecture as projector but input is cat(z, flat_action).
    Output head has bias (reference: out_bias=True, out_norm=False).
    """

    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.block1 = LNBlock(z_dim + action_dim, hidden_dim, nnx.silu, rngs)
        self.block2 = LNBlock(hidden_dim, hidden_dim, nnx.silu, rngs)
        self.output = nnx.Linear(hidden_dim, z_dim, rngs=rngs)

    def __call__(self, z_BZ: jax.Array, action_BA: jax.Array) -> jax.Array:
        x = jnp.concatenate([z_BZ, action_BA], axis=-1)
        x = self.block1(x)
        x = self.block2(x)
        return self.output(x)


class SGFTransitionPredictor(nnx.Module):
    """Transition predictor: (y, action) -> next_y with skip connection.

    Reference: 5 hidden layers of 1024 with LN+SiLU, output Linear(1024, y_dim).
    Forward: next_y = y + mlp(cat(y, flat_action))
    """

    def __init__(self, y_dim: int, action_dim: int, hidden_dim: int, num_hidden: int, rngs: nnx.Rngs):
        layers = []
        in_dim = y_dim + action_dim
        for _ in range(num_hidden):
            layers.append(LNBlock(in_dim, hidden_dim, nnx.silu, rngs))
            in_dim = hidden_dim
        self.hidden_blocks = nnx.List(layers)
        self.output = nnx.Linear(hidden_dim, y_dim, rngs=rngs)

    def __call__(self, y_BY: jax.Array, action_BA: jax.Array) -> jax.Array:
        """Predict next state with skip connection: next_y = y + delta."""
        x = jnp.concatenate([y_BY, action_BA], axis=-1)
        for block in self.hidden_blocks:
            x = block(x)
        delta = self.output(x)
        return y_BY + delta


# ---------------------------------------------------------------------------
# SGF World Model wrapper
# ---------------------------------------------------------------------------


class SGFWorldModel(nnx.Module):
    """Bundles encoder + projector + predictor for single-optimizer training.

    The representation optimizer trains these jointly via the VICReg +
    temporal consistency + reward + terminal losses.

    Note: transition_predictor is trained by a separate optimizer.
    """

    def __init__(
        self,
        encoder: SGFEncoder,
        projector: SGFProjector,
        predictor: SGFPredictor,
        reward_pred: RewardPredictor,
        continue_pred: ContinuePredictor,
    ):
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.reward_pred = reward_pred
        self.continue_pred = continue_pred
