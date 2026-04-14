"""Actor-critic composite networks.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    A: action dimension (num_actions for discrete, action_dim for continuous)
    H: hidden layer size
    F: shared embedding dimension (layer_width, SharedDiscreteActorCritic only)
"""

from collections.abc import Callable, Sequence
from math import prod

import distrax
import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.base import MLP
from nexrl.networks.policies import DiscretePolicy, GaussianPolicy
from nexrl.networks.value import VNetwork


def _max_pool_2d(x: jax.Array, window: tuple[int, int] = (2, 2), strides: tuple[int, int] = (1, 1)) -> jax.Array:
    """Max pool over spatial dims for NHWC layout."""
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, *window, 1),
        window_strides=(1, *strides, 1),
        padding="VALID",
    )


class ActorCritic(nnx.Module):
    """Actor-critic wrapper with separate policy and value networks.

    Works with any policy that implements action_log_prob, log_prob_entropy, and act
    (e.g. DiscretePolicy, GaussianPolicy).
    """

    def __init__(self, actor: DiscretePolicy | GaussianPolicy, critic: VNetwork):
        self.actor = actor
        self.critic = critic

    def get_action_and_value(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample action, log_prob, and value. For collection."""
        action, log_prob_B = self.actor.action_log_prob(obs, rng)
        value_B = self.critic(obs)
        return action, log_prob_B, value_B

    def get_pi_and_value(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute log_prob, entropy, and value. For PPO update."""
        log_prob_B, entropy_B = self.actor.log_prob_entropy(obs, action)
        value_B = self.critic(obs)
        return log_prob_B, entropy_B, value_B

    def value(self, obs: jax.Array) -> jax.Array:
        return self.critic(obs)

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        return self.actor.act(obs, rng)


class SharedMLPDiscreteActorCritic(nnx.Module):
    """Shared-encoder MLP actor-critic for discrete actions (PPG).

    The critic receives ``stop_gradient(hidden)`` so value loss gradients
    do not flow through the shared encoder during the policy phase.

    Extra methods for PPG auxiliary phase:
        ``get_hidden``: returns encoder output
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        cnn_encoder: nnx.Module | None = None,
    ):
        self.action_dim = action_dim
        self.cnn_encoder = cnn_encoder
        effective_in = cnn_encoder.features_dim if cnn_encoder else in_features
        self.encoder = MLP(effective_in, hidden_layer_sizes, activation, rngs)
        self.actor_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.critic_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)

    def get_hidden(self, obs: jax.Array) -> jax.Array:
        """Encoder forward pass. (B, D) -> (B, H)."""
        x = self.cnn_encoder(obs) if self.cnn_encoder else obs
        return self.encoder(x)

    def _action_dist(self, obs: jax.Array) -> distrax.Categorical:
        """Return categorical distribution from obs."""
        hidden_BH = self.get_hidden(obs)
        logits_BA = self.actor_head(hidden_BH)
        return distrax.Categorical(logits=logits_BA)

    def get_action_and_value(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample action, log_prob, and value. Critic gets stop_gradient(hidden)."""
        hidden_BH = self.get_hidden(obs)
        logits_BA = self.actor_head(hidden_BH)
        dist = distrax.Categorical(logits=logits_BA)
        action_B = dist.sample(seed=rng)
        value_B = self.critic_head(jax.lax.stop_gradient(hidden_BH)).squeeze(-1)
        return action_B, dist.log_prob(action_B), value_B  # pyright: ignore[reportReturnType]

    def get_pi_and_value(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute log_prob, entropy, and value. Critic gets stop_gradient(hidden)."""
        hidden_BH = self.get_hidden(obs)
        logits_BA = self.actor_head(hidden_BH)
        dist = distrax.Categorical(logits=logits_BA)
        value_B = self.critic_head(jax.lax.stop_gradient(hidden_BH)).squeeze(-1)
        return dist.log_prob(action), dist.entropy(), value_B  # pyright: ignore[reportReturnType]

    def value(self, obs: jax.Array) -> jax.Array:
        """Critic forward: obs -> scalar value. Uses stop_gradient on encoder."""
        hidden_BH = self.get_hidden(obs)
        return self.critic_head(jax.lax.stop_gradient(hidden_BH)).squeeze(-1)

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Sample action for evaluation."""
        dist = self._action_dist(obs)
        return dist.sample(seed=rng)


class SharedDiscreteActorCritic(nnx.Module):
    """Shared-backbone actor-critic with conv encoder for structured spatial obs.

    Matches ActorCriticConvSymbolicCraftax from Craftax_Baselines reference.
    Uses relu activation throughout and orthogonal initialization.
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        map_obs_shape: tuple[int, ...],
        layer_width: int,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.map_obs_shape = map_obs_shape
        self.map_size = prod(map_obs_shape)

        ortho_hidden = jax.nn.initializers.orthogonal(2.0)
        ortho_actor_out = jax.nn.initializers.orthogonal(0.01)
        ortho_critic_out = jax.nn.initializers.orthogonal(1.0)
        zero_bias = jax.nn.initializers.constant(0.0)

        in_channels = map_obs_shape[-1]

        # Shared conv encoder
        self.conv1 = nnx.Conv(in_channels, 32, kernel_size=(2, 2), kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)
        self.conv2 = nnx.Conv(32, 32, kernel_size=(2, 2), kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)

        # Compute conv output spatial dims: SAME conv + VALID maxpool(2x2, stride=1) per stage
        # Each pool reduces spatial dims by 1; two stages = -2 in each dim
        h, w = map_obs_shape[0], map_obs_shape[1]
        conv_flat_size = (h - 2) * (w - 2) * 32
        inventory_size = in_features - self.map_size
        embed_in = conv_flat_size + inventory_size

        # Embedding layer (after concat of conv_flat + inventory)
        self.embed = nnx.Linear(embed_in, layer_width, kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)

        # Actor head: Dense(W) -> relu -> Dense(A) -> relu -> Dense(A)
        self.actor_h1 = nnx.Linear(layer_width, layer_width, kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)
        self.actor_h2 = nnx.Linear(layer_width, action_dim, kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)
        self.actor_out = nnx.Linear(action_dim, action_dim, kernel_init=ortho_actor_out, bias_init=zero_bias, rngs=rngs)

        # Critic head: Dense(W) -> relu -> Dense(W) -> relu -> Dense(W) -> relu -> Dense(1)
        self.critic_h1 = nnx.Linear(layer_width, layer_width, kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)
        self.critic_h2 = nnx.Linear(layer_width, layer_width, kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)
        self.critic_h3 = nnx.Linear(layer_width, layer_width, kernel_init=ortho_hidden, bias_init=zero_bias, rngs=rngs)
        self.critic_out = nnx.Linear(layer_width, 1, kernel_init=ortho_critic_out, bias_init=zero_bias, rngs=rngs)

    def _encode(self, obs: jax.Array) -> jax.Array:
        """Shared conv+dense encoder. obs: (B, D) flat."""
        obs_BD = obs.reshape((obs.shape[0], -1))
        map_flat = obs_BD[:, : self.map_size]
        inventory = obs_BD[:, self.map_size :]

        x = map_flat.reshape(-1, *self.map_obs_shape)  # (B, H, W, C)
        x = nnx.relu(self.conv1(x))
        x = _max_pool_2d(x)
        x = nnx.relu(self.conv2(x))
        x = _max_pool_2d(x)
        x = x.reshape((x.shape[0], -1))  # (B, conv_flat)

        x_BF = jnp.concatenate([x, inventory], axis=-1)
        return nnx.relu(self.embed(x_BF))  # (B, F)

    def _logits(self, embedding: jax.Array) -> jax.Array:
        """Actor head: embedding -> logits."""
        x = nnx.relu(self.actor_h1(embedding))
        x = nnx.relu(self.actor_h2(x))
        return self.actor_out(x)  # (B, A)

    def _value_head(self, embedding: jax.Array) -> jax.Array:
        """Critic head: embedding -> scalar value."""
        x = nnx.relu(self.critic_h1(embedding))
        x = nnx.relu(self.critic_h2(x))
        x = nnx.relu(self.critic_h3(x))
        return self.critic_out(x).squeeze(-1)  # (B,)

    def value(self, obs: jax.Array) -> jax.Array:
        """Critic forward: obs -> scalar value."""
        return self._value_head(self._encode(obs))

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample action and return log_prob. For trajectory collection."""
        emb = self._encode(obs)
        logits_BA = self._logits(emb)
        dist = distrax.Categorical(logits=logits_BA)
        action_B = dist.sample(seed=rng)
        return action_B, dist.log_prob(action_B)  # pyright: ignore[reportReturnType]

    def get_action_and_value(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample action, log_prob, and value from single shared encoding. For collection."""
        emb = self._encode(obs)
        logits_BA = self._logits(emb)
        dist = distrax.Categorical(logits=logits_BA)
        action_B = dist.sample(seed=rng)
        return action_B, dist.log_prob(action_B), self._value_head(emb)  # pyright: ignore[reportReturnType]

    def get_pi_and_value(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute log_prob, entropy, and value from single shared encoding. For PPO update."""
        emb = self._encode(obs)
        logits_BA = self._logits(emb)
        dist = distrax.Categorical(logits=logits_BA)
        return dist.log_prob(action), dist.entropy(), self._value_head(emb)  # pyright: ignore[reportReturnType]

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute log_prob and entropy for given obs-action pairs. For PPO update."""
        emb = self._encode(obs)
        logits_BA = self._logits(emb)
        dist = distrax.Categorical(logits=logits_BA)
        return dist.log_prob(action), dist.entropy()  # pyright: ignore[reportReturnType]

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Sample action for evaluation."""
        action_B, _ = self.action_log_prob(obs, rng)
        return action_B
