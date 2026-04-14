"""MRS.Q networks: MrQ with simplicial embeddings and Q-ensemble.

"Model-based Representations for Search and Q-learning" (Fujimoto et al.)

Builds on MrQ's LN-MLP architecture with two key changes:
1. Simplicial embeddings (SEM) replace activation on state embeddings
2. Ensemble of N Q-networks (default 10) replaces twin Q

Dimension key:
    B: batch size
    D: observation dimension
    A: action dimension
    S: zs_dim (state embedding)
    X: zsa_dim (state-action embedding)
    H: hidden layer size
    N: num_bins (symexp two-hot)
    Q: num_critics (ensemble size)
"""

from collections.abc import Callable

import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.mrq import LNBaseMLP


def sem(x: jax.Array, groups: int = 8) -> jax.Array:
    """Simplicial embedding: reshape last dim into groups and softmax.

    Each group is independently normalized via softmax, producing a
    concatenation of probability simplices. Requires x.shape[-1] % groups == 0.

    Args:
        x: input array, any shape with last dim divisible by groups
        groups: number of softmax groups

    Returns:
        Array of same shape with softmax applied within each group.
    """
    *batch, dim = x.shape
    x = x.reshape(*batch, -1, groups)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*batch, dim)


class MrsQEncoder(nnx.Module):
    """Learned state embedding with forward dynamics model and simplicial embeddings.

    Identical to MrQEncoder except zs() uses SEM instead of activation, and
    model_all() applies SEM + LayerNorm to predicted next_zs.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        zs_dim: int,
        za_dim: int,
        zsa_dim: int,
        num_bins: int,
        discrete: bool,
        activation: Callable,
        sem_groups: int,
        rngs: nnx.Rngs,
    ):
        self.discrete = discrete
        self.zs_dim = zs_dim
        self.num_bins = num_bins
        self.activation = activation
        self.sem_groups = sem_groups

        # obs -> zs (with final LN + SEM)
        self.zs_net = LNBaseMLP(obs_dim, hidden_dim, zs_dim, activation, rngs)
        self.zs_ln = nnx.LayerNorm(zs_dim, rngs=rngs)

        # Predicted next_zs gets its own LayerNorm + SEM
        self.zs_pred_ln = nnx.LayerNorm(zs_dim, rngs=rngs)

        # action -> za
        self.za_linear = nnx.Linear(action_dim, za_dim, rngs=rngs)

        # (zs, za) -> zsa (raw output, no final LN+activ)
        self.zsa_net = LNBaseMLP(zs_dim + za_dim, hidden_dim, zsa_dim, activation, rngs)

        # Model head: zsa -> (next_zs, reward_logits, done)
        self.model_head = nnx.Linear(zsa_dim, zs_dim + num_bins + 1, rngs=rngs)

    def zs(self, obs_BD: jax.Array) -> jax.Array:
        """Encode observation to state embedding with SEM. Returns (B, S)."""
        return sem(self.zs_ln(self.zs_net(obs_BD)), self.sem_groups)

    def zsa(self, zs_BS: jax.Array, action_BA: jax.Array) -> jax.Array:
        """Create state-action embedding. Returns (B, X)."""
        za_BA = self.activation(self.za_linear(action_BA))
        x = jnp.concatenate([zs_BS, za_BA], axis=-1)
        return self.zsa_net(x)

    def model_all(self, zs_BS: jax.Array, action_BA: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Predict (done_logit, next_zs, reward_logits) from state embedding and action.

        Returns:
            done_logit_B: (B,) done prediction logit
            next_zs_BS: (B, S) predicted next state embedding (SEM applied)
            reward_logits_BN: (B, N) reward prediction logits
        """
        zsa_BX = self.zsa(zs_BS, action_BA)
        out = self.model_head(zsa_BX)
        next_zs_BS = out[:, : self.zs_dim]
        reward_logits_BN = out[:, self.zs_dim : self.zs_dim + self.num_bins]
        done_logit_B = out[:, -1]
        # Apply SEM to predicted next state
        next_zs_BS = sem(self.zs_pred_ln(next_zs_BS), self.sem_groups)
        return done_logit_B, next_zs_BS, reward_logits_BN


class QHead(nnx.Module):
    """Single Q-network: LNBaseMLP -> LN -> activation -> Linear(1).

    Matches one branch of MrQValue's architecture.
    """

    def __init__(
        self,
        zsa_dim: int,
        hidden_dim: int,
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.base = LNBaseMLP(zsa_dim, hidden_dim, hidden_dim, activation, rngs)
        self.ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.activation = activation

    def __call__(self, zsa_BX: jax.Array) -> jax.Array:
        """Compute single Q-value. Returns (B,)."""
        h = self.activation(self.ln(self.base(zsa_BX)))
        return self.head(h).squeeze(-1)


class MrsQValue(nnx.Module):
    """Ensemble of N Q-networks operating on state-action embeddings.

    Returns stacked Q-values from all heads for min-aggregation.
    """

    def __init__(
        self,
        zsa_dim: int,
        hidden_dim: int,
        num_critics: int,
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.q_heads = nnx.List([QHead(zsa_dim, hidden_dim, activation, rngs) for _ in range(num_critics)])

    def __call__(self, zsa_BX: jax.Array) -> jax.Array:
        """Compute ensemble Q-values. Returns (B, Q)."""
        return jnp.stack([head(zsa_BX) for head in self.q_heads], axis=-1)
