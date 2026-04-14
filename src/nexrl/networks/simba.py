"""SimBa (Simplicity Bias) network architectures.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    A: action dimension (action_dim for continuous)
    H: hidden layer size (hidden_dim)
"""

import distrax
import jax
from flax import nnx
from jax import numpy as jnp


class ResidualBlock(nnx.Module):
    """Pre-LayerNorm residual block with inverted bottleneck (4x expansion)."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.up = nnx.Linear(hidden_dim, 4 * hidden_dim, rngs=rngs)
        self.down = nnx.Linear(4 * hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.down(nnx.relu(self.up(self.norm(x))))  # (B, H)


class SimBaEncoder(nnx.Module):
    """SimBa encoder: linear projection -> N residual blocks -> LayerNorm."""

    def __init__(self, in_features: int, hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        self.projection = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.blocks = nnx.List([ResidualBlock(hidden_dim, rngs) for _ in range(num_blocks)])
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape((x.shape[0], -1))
        x_BH = self.projection(x)
        for block in self.blocks:
            x_BH = block(x_BH)
        return self.norm(x_BH)  # (B, H)


class SimBaSquashedGaussianPolicy(nnx.Module):
    """SquashedGaussianPolicy with SimBaEncoder instead of MLP."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_dim: int,
        num_blocks: int,
        log_std_range: tuple[float, float],
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.action_range = action_range
        self.log_std_range = log_std_range
        self.features = SimBaEncoder(in_features, hidden_dim, num_blocks, rngs)
        self.action_mean = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.action_log_std = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.bij = distrax.Tanh()

    @property
    def action_loc(self) -> float:
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self) -> float:
        return (self.action_range[1] - self.action_range[0]) / 2

    def _action_dist(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        features_BH = self.features(obs)
        mean_BA = self.action_mean(features_BH)
        log_std_BA = self.action_log_std(features_BH)
        log_std_BA = jnp.clip(log_std_BA, *self.log_std_range)
        return distrax.MultivariateNormalDiag(loc=mean_BA, scale_diag=jnp.exp(log_std_BA))

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_BA = action_dist.sample(seed=rng)
        log_prob_B = action_dist.log_prob(action_BA)
        action_BA, log_det_j = self.bij.forward_and_log_det(action_BA)
        action_BA = self.action_loc + action_BA * self.action_scale
        log_prob_B -= log_det_j.sum(axis=-1)
        return action_BA, log_prob_B  # pyright: ignore[reportReturnType]

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self(obs, rng)

    def log_prob(self, obs: jax.Array, action: jax.Array, epsilon: float = 1e-6) -> jax.Array:
        low, high = self.action_range
        action = jnp.clip(action, low + epsilon, high - epsilon)
        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        action, log_det_j = self.bij.inverse_and_log_det(action)  # type: ignore[assignment]
        log_prob_B = action_dist.log_prob(action)
        log_prob_B += log_det_j.sum(axis=-1)
        return log_prob_B  # pyright: ignore[reportReturnType]

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_BA, _ = self(obs, rng)
        return action_BA


class SimBaQNetwork(nnx.Module):
    """Q(s,a) network with SimBaEncoder instead of MLP."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_blocks: int,
        rngs: nnx.Rngs,
    ):
        self.encoder = SimBaEncoder(obs_dim + action_dim, hidden_dim, num_blocks, rngs)
        self.q_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1)  # (B, D+A)
        x = self.encoder(x)
        return self.q_head(x).squeeze(1)  # (B,)
