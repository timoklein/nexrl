"""World model networks for MuZero: representation, dynamics, and prediction.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    S: latent state dimension (latent_dim)
    A: action dimension (num_actions)
    N: number of atoms (categorical support size)
"""

from collections.abc import Callable, Sequence

import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.base import MLP


class RepresentationNetwork(nnx.Module):
    """Encodes raw observations into latent states: obs_BD -> latent_BS."""

    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        encoder: nnx.Module | None = None,
    ):
        self.encoder = encoder
        effective_in = encoder.features_dim if encoder else in_features
        self.mlp = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.latent_head = nnx.Linear(hidden_layer_sizes[-1], latent_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.encoder(obs) if self.encoder else obs
        x = self.mlp(x)
        return self.latent_head(x)  # (B, S)


class DynamicsNetwork(nnx.Module):
    """Predicts next latent state and reward: (latent_BS, action_B) -> (next_latent_BS, reward_logits_BN)."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        reward_num_atoms: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.mlp = MLP(latent_dim + num_actions, hidden_layer_sizes, activation, rngs)
        self.latent_head = nnx.Linear(hidden_layer_sizes[-1], latent_dim, rngs=rngs)
        self.reward_head = nnx.Linear(hidden_layer_sizes[-1], reward_num_atoms, rngs=rngs)

    def __call__(self, latent: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_onehot_BA = jax.nn.one_hot(action, self.num_actions)  # (B, A)
        x = jnp.concatenate([latent, action_onehot_BA], axis=-1)  # (B, S+A)
        x = self.mlp(x)
        next_latent_BS = self.latent_head(x)  # (B, S)
        reward_logits_BN = self.reward_head(x)  # (B, N)
        return next_latent_BS, reward_logits_BN


class PredictionNetwork(nnx.Module):
    """Predicts policy and value from latent: latent_BS -> (policy_logits_BA, value_logits_BN)."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        value_num_atoms: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.mlp = MLP(latent_dim, hidden_layer_sizes, activation, rngs)
        self.policy_head = nnx.Linear(hidden_layer_sizes[-1], num_actions, rngs=rngs)
        self.value_head = nnx.Linear(hidden_layer_sizes[-1], value_num_atoms, rngs=rngs)

    def __call__(self, latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = self.mlp(latent)
        policy_logits_BA = self.policy_head(x)  # (B, A)
        value_logits_BN = self.value_head(x)  # (B, N)
        return policy_logits_BA, value_logits_BN


class MuZeroModel(nnx.Module):
    """Wrapper combining representation, dynamics, and prediction for joint optimization."""

    def __init__(
        self,
        representation: RepresentationNetwork,
        dynamics: DynamicsNetwork,
        prediction: PredictionNetwork,
    ):
        self.representation = representation
        self.dynamics = dynamics
        self.prediction = prediction
