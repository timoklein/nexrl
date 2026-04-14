"""Policy networks for discrete and continuous action spaces.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    A: action dimension (num_actions for discrete, action_dim for continuous)
    H: hidden layer size
"""

from collections.abc import Callable, Sequence

import distrax
import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.base import MLP


class DiscretePolicy(nnx.Module):
    """Policy for discrete action spaces with categorical distribution."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
    ):
        self.encoder = encoder
        effective_in = encoder.features_dim if encoder else in_features
        self.features = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.action_logits = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)

    def _action_dist(self, obs: jax.Array) -> distrax.Categorical:
        x = self.encoder(obs) if self.encoder else obs
        features_BH = self.features(x)
        logits_BA = self.action_logits(features_BH)
        return distrax.Categorical(logits=logits_BA)

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_B = action_dist.sample(seed=rng)
        return action_B, action_dist.log_prob(action_B), action_dist.entropy()  # pyright: ignore[reportReturnType]

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_B, _, _ = self(obs, rng)
        return action_B

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()  # pyright: ignore[reportReturnType]

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_B = action_dist.sample(seed=rng)
        return action_B, action_dist.log_prob(action_B)  # pyright: ignore[reportReturnType]

    def get_logits(self, obs: jax.Array) -> jax.Array:
        x = self.encoder(obs) if self.encoder else obs
        features_BH = self.features(x)
        return self.action_logits(features_BH)  # (B, A)


class GaussianPolicy(nnx.Module):
    """Policy for continuous action spaces with diagonal Gaussian distribution."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
    ):
        self.encoder = encoder
        self.action_dim = action_dim
        self.action_range = action_range
        effective_in = encoder.features_dim if encoder else in_features
        self.features = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.action_mean = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        # Learnable log std parameter (not dependent on input)
        self.action_log_std = nnx.Param(jnp.zeros(action_dim))

    def _action_dist(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        x = self.encoder(obs) if self.encoder else obs
        features_BH = self.features(x)
        mean_BA = self.action_mean(features_BH)
        return distrax.MultivariateNormalDiag(loc=mean_BA, scale_diag=jnp.exp(self.action_log_std[...]))

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_BA = action_dist.sample(seed=rng)
        return action_BA, action_dist.log_prob(action_BA), action_dist.entropy()  # pyright: ignore[reportReturnType]

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_BA, _, _ = self(obs, rng)
        return jnp.clip(action_BA, self.action_range[0], self.action_range[1])

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()  # pyright: ignore[reportReturnType]

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_BA = action_dist.sample(seed=rng)
        return action_BA, action_dist.log_prob(action_BA)  # pyright: ignore[reportReturnType]

    def get_mean_std(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = self.encoder(obs) if self.encoder else obs
        features_BH = self.features(x)
        mean_BA = self.action_mean(features_BH)
        std_A = jnp.exp(self.action_log_std[...])
        return mean_BA, jnp.broadcast_to(std_A, mean_BA.shape)  # (B, A), (B, A)


class SquashedGaussianPolicy(nnx.Module):
    """Policy with squashed Gaussian distribution for bounded continuous actions.

    Uses tanh transformation to bound actions, with log probability correction.
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        log_std_range: tuple[float, float],
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
    ):
        self.encoder = encoder
        self.action_dim = action_dim
        self.action_range = action_range
        self.log_std_range = log_std_range
        effective_in = encoder.features_dim if encoder else in_features
        self.features = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.action_mean = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.action_log_std = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.bij = distrax.Tanh()

    @property
    def action_loc(self) -> float:
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self) -> float:
        return (self.action_range[1] - self.action_range[0]) / 2

    def _action_dist(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        # We have to transform the action manually, since we need to calculate log_probs
        # *before* the tanh transform. Doing it afterwards runs into numerical issues
        # because we cannot invert the tanh for +-1, which can easily be sampled.
        # (e.g. jnp.tanh(8) = 1)
        x = self.encoder(obs) if self.encoder else obs
        features_BH = self.features(x)
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


class DeterministicPolicy(nnx.Module):
    """Deterministic policy for continuous action spaces (e.g., DDPG, TD3)."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.action_range = action_range
        self.features = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.output_layer = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)

    @property
    def action_loc(self) -> float:
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self) -> float:
        return (self.action_range[1] - self.action_range[0]) / 2

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.features(x)  # (B, D) -> (B, H)
        x = self.output_layer(x)  # (B, H) -> (B, A)
        x = jnp.tanh(x)

        action_BA = self.action_loc + x * self.action_scale
        return action_BA

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_BA = self(obs)
        return action_BA
