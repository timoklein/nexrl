"""Recurrent actor-critic for PPO-RNN.

Ported from Craftax baselines (Flax Linen) to Flax NNX.
Reference: https://github.com/MichaelTMatthews/craftax_baselines

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    I: input dimension (embedding output / hidden_dim)
    H: hidden state dimension (layer_size)
    A: action dimension (num_actions)
"""

import distrax
import jax
import jax.numpy as jnp
from flax import nnx


class StandardGRUCell(nnx.Module):
    """Standard GRU cell WITHOUT update-gate bias toward -1.

    Unlike dreamer.py's GRUCell which biases the update gate by -1 to
    keep old state, this uses the standard GRU formulation matching
    the Craftax baselines reference.
    """

    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        # Input-to-hidden: reset, update, new gates
        self.linear_ih = nnx.Linear(input_dim, 3 * hidden_dim, rngs=rngs)
        # Hidden-to-hidden: no bias (bias is in linear_ih)
        self.linear_hh = nnx.Linear(hidden_dim, 3 * hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, h_BH: jax.Array, x_BI: jax.Array) -> jax.Array:
        """Single GRU step.

        Args:
            h_BH: previous hidden state (B, H)
            x_BI: input (B, I)

        Returns:
            New hidden state (B, H)
        """
        H = self.hidden_dim
        gates_ih = self.linear_ih(x_BI)  # (B, 3H)
        gates_hh = self.linear_hh(h_BH)  # (B, 3H)

        # Split into reset, update, new
        r_ih, z_ih, n_ih = gates_ih[:, :H], gates_ih[:, H : 2 * H], gates_ih[:, 2 * H :]
        r_hh, z_hh, n_hh = gates_hh[:, :H], gates_hh[:, H : 2 * H], gates_hh[:, 2 * H :]

        reset_BH = jax.nn.sigmoid(r_ih + r_hh)
        # Standard GRU: NO -1 bias (unlike Dreamer's GRUCell)
        update_BH = jax.nn.sigmoid(z_ih + z_hh)
        new_BH = jnp.tanh(n_ih + reset_BH * n_hh)

        h_new_BH = (1.0 - update_BH) * new_BH + update_BH * h_BH
        return h_new_BH


def _ortho_init(scale: float):
    """Orthogonal initializer with given scale."""
    return nnx.initializers.orthogonal(scale=scale)


def _zero_bias():
    """Constant zero bias initializer."""
    return nnx.initializers.constant(0.0)


class ActorCriticRNN(nnx.Module):
    """Recurrent actor-critic matching the Craftax baselines reference.

    Architecture:
        Embedding: Linear(D, H, ortho(sqrt(2))) + relu
        GRU:      StandardGRUCell(H, H)
        Actor:    Linear(H,H,ortho(2)) + relu -> Linear(H,H,ortho(2)) + relu -> Linear(H,A,ortho(0.01))
        Critic:   Linear(H,H,ortho(2)) + relu -> Linear(H,H,ortho(2)) + relu -> Linear(H,1,ortho(1.0))

    All biases initialized to 0. ReLU activation throughout. Discrete actions only.
    """

    def __init__(self, in_features: int, action_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        sqrt2 = 2.0**0.5

        # Embedding
        self.embedding = nnx.Linear(in_features, hidden_dim, kernel_init=_ortho_init(sqrt2), bias_init=_zero_bias(), rngs=rngs)

        # GRU
        self.gru = StandardGRUCell(hidden_dim, hidden_dim, rngs=rngs)

        # Actor head: 2 hidden layers + output
        self.actor_fc1 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=_ortho_init(2.0), bias_init=_zero_bias(), rngs=rngs)
        self.actor_fc2 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=_ortho_init(2.0), bias_init=_zero_bias(), rngs=rngs)
        self.actor_out = nnx.Linear(hidden_dim, action_dim, kernel_init=_ortho_init(0.01), bias_init=_zero_bias(), rngs=rngs)

        # Critic head: 2 hidden layers + output
        self.critic_fc1 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=_ortho_init(2.0), bias_init=_zero_bias(), rngs=rngs)
        self.critic_fc2 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=_ortho_init(2.0), bias_init=_zero_bias(), rngs=rngs)
        self.critic_out = nnx.Linear(hidden_dim, 1, kernel_init=_ortho_init(1.0), bias_init=_zero_bias(), rngs=rngs)

        self.hidden_dim = hidden_dim

    def initialize_carry(self, batch_size: int) -> jax.Array:
        """Create zero initial hidden state (B, H)."""
        return jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32)

    def __call__(self, hstate_BH: jax.Array, obs_BD: jax.Array, done_B: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass through the full network.

        Resets hidden state on done BEFORE the GRU step (done at timestep t
        means obs at t is from a new episode).

        Returns:
            (new_hstate_BH, logits_BA, value_B)
        """
        obs_BD = obs_BD.reshape((obs_BD.shape[0], -1))

        # Embedding
        x_BH = nnx.relu(self.embedding(obs_BD))  # (B, H)

        # Reset hidden state where episode ended
        hstate_BH = hstate_BH * (1.0 - done_B[:, None])

        # GRU step
        hstate_BH = self.gru(hstate_BH, x_BH)  # (B, H)

        # Actor
        a_BH = nnx.relu(self.actor_fc1(hstate_BH))
        a_BH = nnx.relu(self.actor_fc2(a_BH))
        logits_BA = self.actor_out(a_BH)  # (B, A)

        # Critic
        c_BH = nnx.relu(self.critic_fc1(hstate_BH))
        c_BH = nnx.relu(self.critic_fc2(c_BH))
        value_B = self.critic_out(c_BH).squeeze(-1)  # (B,)

        return hstate_BH, logits_BA, value_B

    def get_action_and_value(
        self, hstate_BH: jax.Array, obs_BD: jax.Array, done_B: jax.Array, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample action and compute log_prob + value.

        Returns:
            (new_hstate_BH, action_B, log_prob_B, value_B)
        """
        hstate_BH, logits_BA, value_B = self(hstate_BH, obs_BD, done_B)
        dist = distrax.Categorical(logits=logits_BA)
        action_B = dist.sample(seed=rng)
        log_prob_B = dist.log_prob(action_B)
        return hstate_BH, action_B, log_prob_B, value_B

    def get_pi_and_value(
        self, hstate_BH: jax.Array, obs_BD: jax.Array, done_B: jax.Array, action_B: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Recompute log_prob and entropy for given actions.

        Returns:
            (new_hstate_BH, log_prob_B, entropy_B, value_B)
        """
        hstate_BH, logits_BA, value_B = self(hstate_BH, obs_BD, done_B)
        dist = distrax.Categorical(logits=logits_BA)
        log_prob_B = dist.log_prob(action_B)
        entropy_B = dist.entropy()
        return hstate_BH, log_prob_B, entropy_B, value_B

    def value(self, hstate_BH: jax.Array, obs_BD: jax.Array, done_B: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute value only.

        Returns:
            (new_hstate_BH, value_B)
        """
        hstate_BH, _, value_B = self(hstate_BH, obs_BD, done_B)
        return hstate_BH, value_B

    def act(self, hstate_BH: jax.Array, obs_BD: jax.Array, done_B: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample action (for eval).

        Returns:
            (new_hstate_BH, action_B)
        """
        hstate_BH, logits_BA, _ = self(hstate_BH, obs_BD, done_B)
        dist = distrax.Categorical(logits=logits_BA)
        action_B = dist.sample(seed=rng)
        return hstate_BH, action_B
