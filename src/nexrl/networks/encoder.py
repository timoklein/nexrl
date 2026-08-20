"""CNN encoder architectures for image-based observations.

NatureCNNEncoder: Nature DQN architecture (Mnih et al., 2015)
ImpalaCNNEncoder: IMPALA ResNet architecture (Espeholt et al., 2018)

Both handle NCHW->NHWC transposition and uint8->float32 normalization internally.

Dimension key:
    B: batch size
    C: channels (input)
    H: height
    W: width
    F: features_dim (output embedding size)
"""

from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp


def _conv_output_size(h: int, kernel: int, stride: int, padding: int = 0) -> int:
    """Compute output spatial dim for VALID convolution."""
    return (h - kernel + 2 * padding) // stride + 1


class NatureCNNEncoder(nnx.Module):
    """Nature DQN convolutional encoder.

    Conv(32, 8x8, stride=4) -> ReLU -> Conv(64, 4x4, stride=2) -> ReLU ->
    Conv(64, 3x3, stride=1) -> ReLU -> flatten -> Dense(features_dim) -> ReLU

    Input: (B, C, H, W) uint8 or float
    Output: (B, F) features in compute_dtype (default fp32)
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        features_dim: int,
        rngs: nnx.Rngs,
        *,
        compute_dtype: jnp.dtype = jnp.float32,
        kernel_init: nnx.Initializer | None = None,
    ):
        self._features_dim = features_dim
        self._compute_dtype = compute_dtype
        # kernel_init=None -> NNX default (lecun_normal), preserving behavior for
        # existing callers; DrQ/Dopamine-reference callers pass xavier_uniform.
        init_kwargs: dict[str, Any] = {} if kernel_init is None else {"kernel_init": kernel_init}
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
            **init_kwargs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
            **init_kwargs,
        )
        self.conv3 = nnx.Conv(
            in_features=64,
            out_features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
            **init_kwargs,
        )

        # Compute flattened size after convolutions (VALID padding is nnx.Conv default)
        h = _conv_output_size(_conv_output_size(_conv_output_size(height, 8, 4), 4, 2), 3, 1)
        w = _conv_output_size(_conv_output_size(_conv_output_size(width, 8, 4), 4, 2), 3, 1)
        flat_size = h * w * 64

        self.dense = nnx.Linear(
            in_features=flat_size,
            out_features=features_dim,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
            **init_kwargs,
        )

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def __call__(self, obs: jax.Array) -> jax.Array:
        # NCHW -> NHWC, normalize to [0, 1] in compute_dtype
        x = obs.transpose(0, 2, 3, 1).astype(self._compute_dtype) / 255.0
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten spatial dims
        x = nnx.relu(self.dense(x))
        return x  # (B, F)


class _ResBlock(nnx.Module):
    """Residual block: ReLU -> Conv -> ReLU -> Conv -> + input."""

    def __init__(
        self,
        channels: int,
        rngs: nnx.Rngs,
        *,
        compute_dtype: jnp.dtype = jnp.float32,
        kernel_init: nnx.Initializer | None = None,
    ):
        # kernel_init=None -> use NNX's default (lecun_normal); a passed initializer
        # (e.g. xavier_uniform for the BBF reference) overrides it. Spread as kwargs
        # so None preserves the exact NNX default for non-BBF callers.
        init_kwargs: dict[str, Any] = {} if kernel_init is None else {"kernel_init": kernel_init}
        self.conv1 = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
            **init_kwargs,
        )
        self.conv2 = nnx.Conv(
            in_features=channels,
            out_features=channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
            **init_kwargs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = nnx.relu(x)
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        return x + residual


class _ConvSequence(nnx.Module):
    """Conv -> MaxPool -> ResBlock -> ResBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rngs: nnx.Rngs,
        *,
        compute_dtype: jnp.dtype = jnp.float32,
        kernel_init: nnx.Initializer | None = None,
    ):
        init_kwargs: dict[str, Any] = {} if kernel_init is None else {"kernel_init": kernel_init}
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
            **init_kwargs,
        )
        self.res1 = _ResBlock(out_channels, rngs, compute_dtype=compute_dtype, kernel_init=kernel_init)
        self.res2 = _ResBlock(out_channels, rngs, compute_dtype=compute_dtype, kernel_init=kernel_init)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        # MaxPool 3x3, stride 2, SAME padding -- NHWC layout
        x = jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, 3, 3, 1),
            window_strides=(1, 2, 2, 1),
            padding="SAME",
        )
        x = self.res1(x)
        x = self.res2(x)
        return x


def _same_pool_output_size(h: int, stride: int) -> int:
    """Output size for SAME-padded max pool."""
    return (h + stride - 1) // stride


class ImpalaCNNEncoder(nnx.Module):
    """IMPALA ResNet convolutional encoder.

    3x ConvSequence(channels=[16*ws, 32*ws, 32*ws]):
        Conv(ch, 3x3) -> MaxPool(3x3, stride=2, SAME) -> ResBlock -> ResBlock
    -> ReLU -> flatten -> Dense(features_dim) -> ReLU

    Input: (B, C, H, W) uint8 or float
    Output: (B, F) features in compute_dtype (default fp32)
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        features_dim: int,
        rngs: nnx.Rngs,
        *,
        width_scale: int = 1,
        compute_dtype: jnp.dtype = jnp.float32,
        return_spatial: bool = False,
        kernel_init: nnx.Initializer | None = None,
    ):
        self._features_dim = features_dim
        self._compute_dtype = compute_dtype
        self._return_spatial = return_spatial
        # kernel_init=None -> NNX default (lecun_normal), preserving behavior for
        # PPO/SGF/etc. callers; BBF passes xavier_uniform to match the reference.
        init_kwargs: dict[str, Any] = {} if kernel_init is None else {"kernel_init": kernel_init}
        channels = [16 * width_scale, 32 * width_scale, 32 * width_scale]
        self.seq1 = _ConvSequence(in_channels, channels[0], rngs, compute_dtype=compute_dtype, kernel_init=kernel_init)
        self.seq2 = _ConvSequence(channels[0], channels[1], rngs, compute_dtype=compute_dtype, kernel_init=kernel_init)
        self.seq3 = _ConvSequence(channels[1], channels[2], rngs, compute_dtype=compute_dtype, kernel_init=kernel_init)

        # Compute spatial size after 3x SAME-padded max pool (stride 2)
        h, w = height, width
        for _ in range(3):
            h = _same_pool_output_size(h, 2)
            w = _same_pool_output_size(w, 2)
        self._spatial_shape = (h, w, channels[-1])
        flat_size = h * w * channels[-1]

        # In spatial mode the encoder returns the (B, H', W', C) conv latent and
        # owns no projection — the consumer (e.g. BBFNetwork) flattens + projects
        # so the same convolutional latent can feed a convolutional transition.
        if not return_spatial:
            self.dense = nnx.Linear(
                in_features=flat_size,
                out_features=features_dim,
                dtype=compute_dtype,
                param_dtype=jnp.float32,
                rngs=rngs,
                **init_kwargs,
            )

    @property
    def features_dim(self) -> int:
        return self._features_dim

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """(H', W', C) of the spatial latent (only meaningful in return_spatial mode)."""
        return self._spatial_shape

    def __call__(self, obs: jax.Array) -> jax.Array:
        # NCHW -> NHWC, normalize to [0, 1] in compute_dtype
        x = obs.transpose(0, 2, 3, 1).astype(self._compute_dtype) / 255.0
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = nnx.relu(x)
        if self._return_spatial:
            return x  # (B, H', W', C) spatial latent
        x = x.reshape(x.shape[0], -1)  # flatten spatial dims
        x = nnx.relu(self.dense(x))
        return x  # (B, F)
