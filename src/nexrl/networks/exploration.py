"""Exploration modules: RND and ICM.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    H: hidden layer size
    L: ICM latent dimension
    A: action dimension (num_actions, ICM inverse model)
"""

from collections.abc import Sequence

import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.base import MLP, _freeze_params


class RNDModule(nnx.Module):
    """Random Network Distillation module for exploration.

    Target network uses FrozenParam (never updated); predictor is trained
    to match target output. MSE between them is the intrinsic reward.
    """

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: Sequence[int],
        embedding_dim: int,
        rngs: nnx.Rngs,
    ):
        # Target: fixed random features (frozen)
        self.target_mlp = MLP(in_features, hidden_layer_sizes, nnx.relu, rngs)
        self.target_head = nnx.Linear(hidden_layer_sizes[-1], embedding_dim, rngs=rngs)
        _freeze_params(self.target_mlp)
        _freeze_params(self.target_head)

        # Predictor: trainable (standard nnx.Param)
        self.predictor_mlp = MLP(in_features, hidden_layer_sizes, nnx.relu, rngs)
        self.predictor_head = nnx.Linear(hidden_layer_sizes[-1], embedding_dim, rngs=rngs)

    def intrinsic_reward(self, obs: jax.Array) -> jax.Array:
        """MSE between predictor and target outputs. Shape: (B,)."""
        target_BH = jax.lax.stop_gradient(self.target_head(self.target_mlp(obs)))
        pred_BH = self.predictor_head(self.predictor_mlp(obs))
        return jnp.square(target_BH - pred_BH).mean(axis=-1)

    def prediction_loss(self, obs: jax.Array, done: jax.Array) -> jax.Array:
        """RND predictor loss, masked at terminal states. Returns scalar."""
        target_BH = jax.lax.stop_gradient(self.target_head(self.target_mlp(obs)))
        pred_BH = self.predictor_head(self.predictor_mlp(obs))
        error_BH = (target_BH - pred_BH) * (1 - done[:, None])
        return jnp.square(error_BH).mean()


class ICMModule(nnx.Module):
    """Intrinsic Curiosity Module: encoder + forward model + inverse model.

    Used by both PPO-ICM and PPO-E3B. All parameters are trainable (nnx.Param).
    """

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        latent_size: int,
        hidden_layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.latent_size = latent_size

        # Encoder: obs -> latent
        self.encoder = MLP(in_features, hidden_layer_sizes, nnx.relu, rngs)
        self.encoder_head = nnx.Linear(hidden_layer_sizes[-1], latent_size, rngs=rngs)

        # Forward model: (latent + one_hot_action) -> predicted next latent
        self.forward_model = MLP(latent_size + num_actions, hidden_layer_sizes, nnx.relu, rngs)
        self.forward_head = nnx.Linear(hidden_layer_sizes[-1], latent_size, rngs=rngs)

        # Inverse model: (latent + next_latent) -> action logits
        self.inverse_model = MLP(latent_size * 2, hidden_layer_sizes, nnx.relu, rngs)
        self.inverse_head = nnx.Linear(hidden_layer_sizes[-1], num_actions, rngs=rngs)

    def encode(self, obs: jax.Array) -> jax.Array:
        """Encode observations to latent space. (B, D) -> (B, L)."""
        return self.encoder_head(self.encoder(obs))

    def forward_prediction(self, latent: jax.Array, action: jax.Array) -> jax.Array:
        """Predict next latent from current latent and action. (B, L), (B,) -> (B, L)."""
        action_oh = jax.nn.one_hot(action, self.num_actions)
        x = jnp.concatenate([latent, action_oh], axis=-1)
        return self.forward_head(self.forward_model(x))

    def inverse_prediction(self, latent: jax.Array, next_latent: jax.Array) -> jax.Array:
        """Predict action log-probs from latent pair. (B, L), (B, L) -> (B, A)."""
        x = jnp.concatenate([latent, next_latent], axis=-1)
        return jax.nn.log_softmax(self.inverse_head(self.inverse_model(x)))

    def intrinsic_reward(self, obs: jax.Array, next_obs: jax.Array, action: jax.Array) -> jax.Array:
        """Forward prediction error as intrinsic reward. (B,)."""
        latent_BL = jax.lax.stop_gradient(self.encode(obs))
        next_latent_BL = jax.lax.stop_gradient(self.encode(next_obs))
        pred_next_BL = self.forward_prediction(latent_BL, action)
        return jnp.square(pred_next_BL - next_latent_BL).mean(axis=-1)
