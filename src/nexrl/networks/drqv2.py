"""DrQ-v2 networks: CNN encoder with LN trunk, deterministic actor, and twin critic.

"Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning"
(Yarats et al., 2022)

Dimension key:
    B: batch size
    C: input channels (frame_stack * 3 for RGB)
    H: image height (84)
    W: image width (84)
    R: repr_dim (encoder output, 32 * 35 * 35 = 39200)
    F: feature_dim (trunk output, default 50)
    A: action dimension
"""

from collections.abc import Sequence

import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.base import MLP
from nexrl.networks.mrq import LNBlock


class DrQV2Encoder(nnx.Module):
    """4-layer CNN encoder for 84x84 pixel observations.

    Architecture: 4x Conv(32, 3x3) with strides [2, 1, 1, 1], ReLU.
    Input normalized to [-0.5, 0.5] range (obs/255 - 0.5).
    For 84x84 input: 84->41->39->37->35, repr_dim = 32*35*35 = 39200.
    """

    def __init__(self, in_channels: int, rngs: nnx.Rngs):
        strides = [2, 1, 1, 1]
        self.convs = nnx.List(
            [
                nnx.Conv(
                    in_features=in_channels if i == 0 else 32,
                    out_features=32,
                    kernel_size=(3, 3),
                    strides=(s, s),
                    padding="VALID",
                    rngs=rngs,
                )
                for i, s in enumerate(strides)
            ]
        )

    def __call__(self, obs_BCHW: jax.Array) -> jax.Array:
        # Normalize uint8 to [-0.5, 0.5]
        x = obs_BCHW.astype(jnp.float32) / 255.0 - 0.5
        # NCHW -> NHWC for Flax Conv
        x = jnp.transpose(x, (0, 2, 3, 1))
        for conv in self.convs:
            x = nnx.relu(conv(x))
        # Flatten spatial dims
        return x.reshape(x.shape[0], -1)


class DrQV2Actor(nnx.Module):
    """Deterministic actor with LN trunk.

    trunk: Linear(repr_dim, feature_dim) -> LayerNorm -> Tanh
    policy: MLP(feature_dim, hidden_sizes, relu) -> Linear(hidden[-1], action_dim) -> tanh
    """

    def __init__(
        self,
        repr_dim: int,
        feature_dim: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.trunk = LNBlock(repr_dim, feature_dim, nnx.tanh, rngs=rngs)
        self.policy = MLP(feature_dim, hidden_layer_sizes, nnx.relu, rngs=rngs)
        self.head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)

    def __call__(self, features_BR: jax.Array) -> jax.Array:
        x = self.trunk(features_BR)
        x = self.policy(x)
        return jnp.tanh(self.head(x))


class DrQV2Critic(nnx.Module):
    """Twin Q-critic with separate LN trunks.

    Each Q-head: trunk(repr_dim -> feature_dim) + MLP(feature_dim+action_dim, hidden, relu) + Linear(1)
    """

    def __init__(
        self,
        repr_dim: int,
        feature_dim: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.trunk = LNBlock(repr_dim, feature_dim, nnx.tanh, rngs=rngs)
        self.q1_mlp = MLP(feature_dim + action_dim, hidden_layer_sizes, nnx.relu, rngs=rngs)
        self.q1_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)
        self.q2_mlp = MLP(feature_dim + action_dim, hidden_layer_sizes, nnx.relu, rngs=rngs)
        self.q2_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)

    def __call__(self, features_BR: jax.Array, action_BA: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = self.trunk(features_BR)
        ha = jnp.concatenate([h, action_BA], axis=-1).reshape(h.shape[0], -1)
        q1_B = self.q1_head(self.q1_mlp(ha)).squeeze(-1)
        q2_B = self.q2_head(self.q2_mlp(ha)).squeeze(-1)
        return q1_B, q2_B
