"""Value and Q-networks for discrete and continuous action spaces.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    A: action dimension (num_actions for discrete, action_dim for continuous)
    H: hidden layer size
    K: num quantile samples (IQN only)
    E: cosine embedding dimension (IQN only)
    N: number of atoms (C51 only)
"""

from collections.abc import Callable, Sequence

import distrax
import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.base import MLP


class VNetwork(nnx.Module):
    """State-value network V(s)."""

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
    ):
        self.encoder = encoder
        effective_in = encoder.features_dim if encoder else in_features
        self.mlp = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.value_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.encoder(obs) if self.encoder else obs
        x = self.mlp(x)
        return self.value_head(x).squeeze(1)  # (B,)


class QNetwork(nnx.Module):
    """Action-value network Q(s, a) for continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
    ):
        self.encoder = encoder
        effective_obs_dim = encoder.features_dim if encoder else obs_dim
        self.mlp = MLP(effective_obs_dim + action_dim, hidden_layer_sizes, activation, rngs)
        self.q_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        obs_flat = self.encoder(obs) if self.encoder else obs.reshape(obs.shape[0], -1)
        x = jnp.concatenate([obs_flat, action], axis=-1)  # (B, D+A)
        x = self.mlp(x)
        return self.q_head(x).squeeze(1)  # (B,)


class DiscreteQNetwork(nnx.Module):
    """Action-value network Q(s, a) for discrete actions."""

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
        self.action_dim = action_dim
        effective_in = encoder.features_dim if encoder else in_features
        self.mlp = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.q_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.encoder(obs) if self.encoder else obs
        x = self.mlp(x)
        return self.q_head(x)  # (B, A)

    def take(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-values for specific actions."""
        q_BA = self(obs)
        q_B = jnp.take_along_axis(q_BA, action[:, None], axis=1).squeeze(1)
        return q_B

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        q_BA = self(obs)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng)


class DuelingQNetwork(nnx.Module):
    """Dueling Q-network architecture with separate value and advantage streams."""

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
        self.action_dim = action_dim
        effective_in = encoder.features_dim if encoder else in_features
        self.mlp = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.value_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)
        self.advantage_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.encoder(obs) if self.encoder else obs
        x = self.mlp(x)
        value_B1 = self.value_head(x)  # (B, 1)
        advantage_BA = self.advantage_head(x)  # (B, A)
        advantage_BA = advantage_BA - jnp.mean(advantage_BA, axis=-1, keepdims=True)
        return value_B1 + advantage_BA  # (B, A)

    def take(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-values for specific actions."""
        q_BA = self(obs)
        q_B = jnp.take_along_axis(q_BA, action[:, None], axis=1).squeeze(1)
        return q_B

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        q_BA = self(obs)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng)


class CategoricalQNetwork(nnx.Module):
    """Categorical (C51) Q-network for distributional RL.

    Outputs a discrete distribution over n_atoms for each action.

    Dimension key (additional):
        N: number of atoms (n_atoms)
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        n_atoms: int,
        v_min: float,
        v_max: float,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
    ):
        self.encoder = encoder
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        effective_in = encoder.features_dim if encoder else in_features
        self.mlp = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.logits_head = nnx.Linear(hidden_layer_sizes[-1], action_dim * n_atoms, rngs=rngs)

    @property
    def support(self) -> jax.Array:
        return jnp.linspace(self.v_min, self.v_max, self.n_atoms)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Raw logits. Shape: (B, A, N)."""
        x = self.encoder(obs) if self.encoder else obs
        x = self.mlp(x)
        logits_BAN = self.logits_head(x).reshape(-1, self.action_dim, self.n_atoms)
        return logits_BAN

    def probs(self, obs: jax.Array) -> jax.Array:
        """Probability distribution over atoms. Shape: (B, A, N)."""
        return jax.nn.softmax(self(obs), axis=-1)

    def q_values(self, obs: jax.Array) -> jax.Array:
        """Expected Q-values: (probs * support).sum(-1). Shape: (B, A)."""
        probs_BAN = self.probs(obs)
        return (probs_BAN * self.support).sum(axis=-1)

    def take_probs(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Probs for selected actions. Shape: (B, N)."""
        probs_BAN = self.probs(obs)
        idx_B1_1 = action[:, None, None]
        return jnp.take_along_axis(probs_BAN, jnp.broadcast_to(idx_B1_1, (*action.shape, 1, self.n_atoms)), axis=1).squeeze(1)

    def take_logprobs(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Log-probs for selected actions. Shape: (B, N)."""
        logits_BAN = self(obs)
        log_probs_BAN = jax.nn.log_softmax(logits_BAN, axis=-1)
        idx_B1_1 = action[:, None, None]
        return jnp.take_along_axis(
            log_probs_BAN, jnp.broadcast_to(idx_B1_1, (*action.shape, 1, self.n_atoms)), axis=1
        ).squeeze(1)

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        q_BA = self.q_values(obs)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng)


class ImplicitQuantileNetwork(nnx.Module):
    """Implicit Quantile Network (IQN) for distributional RL.

    Learns the full distribution of Q-values by sampling quantiles.
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        risk_distortion: Callable = lambda tau: tau,
    ):
        # risk_distortion: Callable = lambda tau: tau
        # risk_distortion: Callable = lambda tau: 0.8 * tau
        # Or e.g.: tau ** 0.71 / (tau ** 0.71 + (1 - tau) ** 0.71) ** (1 / 0.71)
        self.action_dim = action_dim
        self.activation = activation
        self.risk_distortion = risk_distortion
        self.hidden_layer_sizes = hidden_layer_sizes

        # State embedding network
        self.state_mlp = MLP(in_features, hidden_layer_sizes, activation, rngs)

        # Quantile embedding network
        self.phi_dense = nnx.Linear(self.embedding_dim, self.embedding_dim, rngs=rngs)

        # Combine state and quantile embeddings
        self.combine_dense = nnx.Linear(self.embedding_dim, 64, rngs=rngs)
        self.output_dense = nnx.Linear(64, action_dim, rngs=rngs)

    @property
    def embedding_dim(self) -> int:
        return self.hidden_layer_sizes[-1]

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass that samples quantiles.

        Args:
            obs: Observations with shape (batch_size, obs_dim)
            rng: RNG key for sampling quantiles

        Returns:
            z_values: Quantile values with shape (batch_size, action_dim)
            tau: Sampled quantiles with shape (batch_size,)
        """
        x = obs.reshape(obs.shape[0], -1)
        psi_BH = self.state_mlp(x)

        tau_B = distrax.Uniform(0, 1).sample(seed=rng, sample_shape=obs.shape[0])
        tau_B = self.risk_distortion(tau_B)
        phi_BE = jnp.cos(jnp.pi * jnp.outer(tau_B, jnp.arange(self.embedding_dim)))
        phi_BE = nnx.relu(self.phi_dense(phi_BE))

        x = nnx.swish(self.combine_dense(psi_BH * phi_BE))
        return self.output_dense(x), tau_B  # (B, A), (B,)

    def q(self, obs: jax.Array, rng: jax.Array, num_samples: int = 32) -> jax.Array:
        """Compute expected Q-values by averaging over quantile samples.

        Args:
            obs: Observations with shape (batch_size, obs_dim)
            rng: RNG key
            num_samples: Number of quantile samples to average

        Returns:
            Q-values with shape (batch_size, action_dim)
        """
        rng = jax.random.split(rng, num_samples)
        z_KBA, _ = jax.vmap(self, in_axes=(None, 0))(obs, rng)  # (K, B, A)
        return z_KBA.mean(axis=0)  # (B, A)

    def best_action(self, obs: jax.Array, rng: jax.Array, num_samples: int = 32) -> jax.Array:
        """Select best action based on expected Q-values.

        Args:
            obs: Observations with shape (batch_size, obs_dim)
            rng: RNG key
            num_samples: Number of quantile samples to average

        Returns:
            Best actions with shape (batch_size,)
        """
        q_BA = self.q(obs, rng, num_samples)
        best_action_B = jnp.argmax(q_BA, axis=1)
        return best_action_B

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        rng_tau, rng_epsilon = jax.random.split(rng)
        q_BA = self.q(obs, rng_tau)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng_epsilon)


class NormalizedDiscreteQNetwork(nnx.Module):
    """Q-network with LayerNorm for discrete actions. Matches PureJaxQL."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        *,
        norm_input: bool = False,
    ):
        self.action_dim = action_dim
        self.activation = activation
        self.norm_input = norm_input

        if norm_input:
            self.input_norm = nnx.LayerNorm(in_features, rngs=rngs)

        layers = []
        norms = []
        current_in = in_features
        for size in hidden_layer_sizes:
            layers.append(nnx.Linear(current_in, size, rngs=rngs))
            norms.append(nnx.LayerNorm(size, rngs=rngs))
            current_in = size

        self.layers = nnx.List(layers)
        self.norms = nnx.List(norms)
        self.q_head = nnx.Linear(current_in, action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs.reshape((obs.shape[0], -1))
        if self.norm_input:
            x = self.input_norm(x)
        for layer, norm in zip(self.layers, self.norms):
            x = self.activation(norm(layer(x)))
        return self.q_head(x)  # (B, A)

    def take(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-values for specific actions."""
        q_BA = self(obs)
        return jnp.take_along_axis(q_BA, action[:, None], axis=1).squeeze(1)

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        q_BA = self(obs)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng)
