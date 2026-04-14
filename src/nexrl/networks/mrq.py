"""MrQ networks: learned state embeddings with LN-MLP architecture.

"Towards General-Purpose Model-Free RL" (Fujimoto et al.)

Uses LayerNorm after every hidden layer, symexp two-hot for reward prediction,
and separate encoder/policy/value modules with distinct optimizers.

Dimension key:
    B: batch size
    D: observation dimension
    A: action dimension
    S: zs_dim (state embedding)
    X: zsa_dim (state-action embedding)
    H: hidden layer size
    N: num_bins (symexp two-hot)
"""

from collections.abc import Callable

import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.dreamer import symexp


# ---------------------------------------------------------------------------
# Symexp two-hot utilities
# ---------------------------------------------------------------------------


def make_symexp_support(lower: float, upper: float, num_bins: int) -> jax.Array:
    """Create non-uniform support bins via symexp(linspace(lower, upper, num_bins)).

    Unlike Dreamer's two-hot (uniform bins in symlog space), MrQ's bins are
    non-uniformly spaced in the original value space.

    Args:
        lower: lower bound in symlog space
        upper: upper bound in symlog space
        num_bins: number of bins

    Returns:
        Support array of shape (num_bins,) in original value space.
    """
    return symexp(jnp.linspace(lower, upper, num_bins))


def symexp_twohot_encode(x: jax.Array, bins: jax.Array) -> jax.Array:
    """Two-hot encode scalar values against a symexp support.

    Args:
        x: scalar values, any shape
        bins: support bins from make_symexp_support, shape (N,)

    Returns:
        Two-hot vectors, shape (*x.shape, N)
    """
    num_bins = bins.shape[0]
    x_clipped = jnp.clip(x, bins[0], bins[-1])
    below = jnp.sum((bins[None] <= x_clipped[..., None]).astype(jnp.float32), axis=-1) - 1
    below = jnp.clip(below, 0, num_bins - 2).astype(jnp.int32)
    above = below + 1
    bins_below = bins[below]
    bins_above = bins[above]
    weight = (x_clipped - bins_below) / jnp.maximum(bins_above - bins_below, 1e-8)
    weight = jnp.clip(weight, 0.0, 1.0)
    target = jax.nn.one_hot(below, num_bins, dtype=jnp.float32) * (1.0 - weight[..., None])
    target = target + jax.nn.one_hot(above, num_bins, dtype=jnp.float32) * weight[..., None]
    return target


def symexp_twohot_cross_entropy(pred_logits: jax.Array, target_scalar: jax.Array, bins: jax.Array) -> jax.Array:
    """Cross-entropy loss between predicted logits and two-hot encoded targets.

    Args:
        pred_logits: raw logits, shape (*, N)
        target_scalar: scalar target values, shape (*)
        bins: support bins from make_symexp_support

    Returns:
        Per-sample cross-entropy loss, shape (*)
    """
    target_twohot = symexp_twohot_encode(target_scalar, bins)
    log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
    return -jnp.sum(target_twohot * log_probs, axis=-1)


def symexp_twohot_decode(logits: jax.Array, bins: jax.Array) -> jax.Array:
    """Decode symexp two-hot logits to scalar values.

    Args:
        logits: raw logits, shape (*, N)
        bins: support bins from make_symexp_support

    Returns:
        Decoded scalar values, shape (*)
    """
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * bins, axis=-1)


# ---------------------------------------------------------------------------
# LN-MLP building blocks (MrQ-specific, not shared with other algos)
# ---------------------------------------------------------------------------


class LNBlock(nnx.Module):
    """Linear -> LayerNorm -> activation.

    MrQ uses LN+activ after every hidden layer, unlike the base MLP.
    """

    def __init__(self, in_features: int, out_features: int, activation: Callable, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.ln = nnx.LayerNorm(out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.activation(self.ln(self.linear(x)))


class LNBaseMLP(nnx.Module):
    """3-layer MLP using LNBlock for hidden layers, raw Linear for output.

    Matches the reference BaseMLP pattern: hidden1 -> LN+act -> hidden2 -> LN+act -> output.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.block1 = LNBlock(in_features, hidden_dim, activation, rngs)
        self.block2 = LNBlock(hidden_dim, hidden_dim, activation, rngs)
        self.output_linear = nnx.Linear(hidden_dim, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.block1(x)
        x = self.block2(x)
        return self.output_linear(x)


# ---------------------------------------------------------------------------
# MrQ modules
# ---------------------------------------------------------------------------


class MrQEncoder(nnx.Module):
    """Learned state embedding with forward dynamics model.

    Maps observations to state embeddings (zs), creates state-action embeddings
    (zsa), and predicts next state, reward, and done via a model head.
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
        rngs: nnx.Rngs,
    ):
        self.discrete = discrete
        self.zs_dim = zs_dim
        self.num_bins = num_bins
        self.activation = activation

        # obs -> zs (with final LN+activ)
        self.zs_net = LNBaseMLP(obs_dim, hidden_dim, zs_dim, activation, rngs)
        self.zs_ln = nnx.LayerNorm(zs_dim, rngs=rngs)

        # action -> za
        self.za_linear = nnx.Linear(action_dim, za_dim, rngs=rngs)

        # (zs, za) -> zsa (raw output, no final LN+activ)
        self.zsa_net = LNBaseMLP(zs_dim + za_dim, hidden_dim, zsa_dim, activation, rngs)

        # Model head: zsa -> (next_zs, reward_logits, done)
        self.model_head = nnx.Linear(zsa_dim, zs_dim + num_bins + 1, rngs=rngs)

    def zs(self, obs_BD: jax.Array) -> jax.Array:
        """Encode observation to state embedding. Returns (B, S)."""
        return self.activation(self.zs_ln(self.zs_net(obs_BD)))

    def zsa(self, zs_BS: jax.Array, action_BA: jax.Array) -> jax.Array:
        """Create state-action embedding. Returns (B, X)."""
        za_BA = self.activation(self.za_linear(action_BA))
        x = jnp.concatenate([zs_BS, za_BA], axis=-1)
        return self.zsa_net(x)

    def model_all(self, zs_BS: jax.Array, action_BA: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Predict (done_logit, next_zs, reward_logits) from state embedding and action.

        Returns:
            done_logit_B: (B,) done prediction logit
            next_zs_BS: (B, S) predicted next state embedding
            reward_logits_BN: (B, N) reward prediction logits
        """
        zsa_BX = self.zsa(zs_BS, action_BA)
        out = self.model_head(zsa_BX)
        next_zs_BS = out[:, : self.zs_dim]
        reward_logits_BN = out[:, self.zs_dim : self.zs_dim + self.num_bins]
        done_logit_B = out[:, -1]
        return done_logit_B, next_zs_BS, reward_logits_BN


class MrQPolicy(nnx.Module):
    """Deterministic policy operating on state embeddings.

    Discrete: Gumbel-softmax sampling during training, argmax during eval.
    Continuous: tanh-squashed output.
    """

    def __init__(
        self,
        zs_dim: int,
        action_dim: int,
        hidden_dim: int,
        discrete: bool,
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.discrete = discrete
        self.action_dim = action_dim
        self.net = LNBaseMLP(zs_dim, hidden_dim, action_dim, activation, rngs)

    def __call__(self, zs_BS: jax.Array, rng: jax.Array, tau: float = 1.0) -> tuple[jax.Array, jax.Array]:
        """Sample action with exploration noise.

        Args:
            zs_BS: state embedding (B, S)
            rng: PRNG key for Gumbel noise (discrete) or unused (continuous)
            tau: Gumbel-softmax temperature (discrete only)

        Returns:
            (action_BA, pre_activ_BA): action and pre-activation outputs
            Discrete: action is soft one-hot (B, A), pre_activ is logits (B, A)
            Continuous: action is tanh(pre_activ) (B, A), pre_activ is raw output (B, A)
        """
        pre_activ_BA = self.net(zs_BS)
        if self.discrete:
            # Gumbel-softmax
            u = jax.random.uniform(rng, pre_activ_BA.shape, minval=1e-8, maxval=1.0 - 1e-8)
            gumbel_noise = -jnp.log(-jnp.log(u))
            action_BA = jax.nn.softmax((pre_activ_BA + gumbel_noise) / tau, axis=-1)
            return action_BA, pre_activ_BA
        else:
            action_BA = jnp.tanh(pre_activ_BA)
            return action_BA, pre_activ_BA

    def act(self, zs_BS: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Deterministic action (no noise).

        Returns:
            (action, pre_activ)
            Discrete: action is one-hot of argmax (B, A)
            Continuous: action is tanh(pre_activ) (B, A)
        """
        pre_activ_BA = self.net(zs_BS)
        if self.discrete:
            action_BA = jax.nn.one_hot(jnp.argmax(pre_activ_BA, axis=-1), self.action_dim, dtype=jnp.float32)
            return action_BA, pre_activ_BA
        else:
            return jnp.tanh(pre_activ_BA), pre_activ_BA


class MrQValue(nnx.Module):
    """Twin Q-value network operating on state-action embeddings.

    Two independent networks, each: LNBaseMLP(hidden) -> LN+activ -> Linear(1).
    Matches the reference: 4 hidden layers with LN+activ total (3 in BaseMLP + 1 extra).
    """

    def __init__(
        self,
        zsa_dim: int,
        hidden_dim: int,
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        # Q1: BaseMLP(3 layers) + LN+activ + Linear(1)
        self.q1_base = LNBaseMLP(zsa_dim, hidden_dim, hidden_dim, activation, rngs)
        self.q1_ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.q1_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

        # Q2: BaseMLP(3 layers) + LN+activ + Linear(1)
        self.q2_base = LNBaseMLP(zsa_dim, hidden_dim, hidden_dim, activation, rngs)
        self.q2_ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.q2_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

        self.activation = activation

    def __call__(self, zsa_BX: jax.Array) -> jax.Array:
        """Compute twin Q-values. Returns (B, 2)."""
        h1 = self.activation(self.q1_ln(self.q1_base(zsa_BX)))
        q1_B = self.q1_head(h1).squeeze(-1)

        h2 = self.activation(self.q2_ln(self.q2_base(zsa_BX)))
        q2_B = self.q2_head(h2).squeeze(-1)

        return jnp.stack([q1_B, q2_B], axis=-1)  # (B, 2)
