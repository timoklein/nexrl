"""Dreamer-v3 networks: RSSM world model, encoder/decoder, actor/critic.

Learns a latent world model (RSSM) from experience, then trains actor-critic
purely in imagination (latent rollouts).

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    G: deter_dim (GRU hidden state)
    Z: stoch_dim (C * L flattened)
    C: stoch_categories
    L: stoch_classes
    A: action dimension
    H: hidden layer size
    N: num_bins (symlog two-hot)
    F: encoder features dim
"""

from collections.abc import Callable, Sequence

import jax
from flax import nnx
from jax import numpy as jnp

from nexrl.networks.base import MLP


# ---------------------------------------------------------------------------
# Symlog utilities
# ---------------------------------------------------------------------------


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric logarithmic compression: sign(x) * log(1 + |x|)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def twohot_encode(x: jax.Array, num_bins: int) -> jax.Array:
    """Encode scalar values as two-hot vectors on symlog support [-20, 20].

    Args:
        x: scalar values, any shape
        num_bins: number of bins in the support

    Returns:
        Two-hot encoded vectors, shape (*x.shape, num_bins)
    """
    support = jnp.linspace(-20.0, 20.0, num_bins)
    x_symlog = symlog(x)
    # Clamp to support range
    x_symlog = jnp.clip(x_symlog, -20.0, 20.0)
    # Find lower bin index
    below = jnp.sum((support[None] <= x_symlog[..., None]).astype(jnp.float32), axis=-1) - 1
    below = jnp.clip(below, 0, num_bins - 2).astype(jnp.int32)
    above = below + 1
    # Interpolation weight
    support_below = support[below]
    support_above = support[above]
    weight = (x_symlog - support_below) / jnp.maximum(support_above - support_below, 1e-8)
    weight = jnp.clip(weight, 0.0, 1.0)
    # Two-hot: spread probability between adjacent bins
    target = jax.nn.one_hot(below, num_bins, dtype=jnp.float32) * (1.0 - weight[..., None])
    target = target + jax.nn.one_hot(above, num_bins, dtype=jnp.float32) * weight[..., None]
    return target


def twohot_decode(logits: jax.Array, num_bins: int) -> jax.Array:
    """Decode symlog two-hot logits to scalar values.

    Args:
        logits: raw logits, shape (*, num_bins)
        num_bins: number of bins in the support

    Returns:
        Decoded scalar values in original (non-symlog) space
    """
    support = jnp.linspace(-20.0, 20.0, num_bins)
    probs = jax.nn.softmax(logits, axis=-1)
    expected_symlog = jnp.sum(probs * support, axis=-1)
    return symexp(expected_symlog)


# ---------------------------------------------------------------------------
# GRU Cell
# ---------------------------------------------------------------------------


class GRUCell(nnx.Module):
    """Standard GRU cell with update-gate bias toward keeping old state.

    Simplified from reference's BlockLinear grouped GRU — sufficient for
    256-512 dim states used in gymnax environments.
    """

    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        # Input-to-hidden: reset, update, new gates
        self.linear_ih = nnx.Linear(input_dim, 3 * hidden_dim, rngs=rngs)
        # Hidden-to-hidden: no bias (bias is in linear_ih)
        self.linear_hh = nnx.Linear(hidden_dim, 3 * hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, h_BG: jax.Array, x_BI: jax.Array) -> jax.Array:
        """Single GRU step.

        Args:
            h_BG: previous hidden state (B, G)
            x_BI: input (B, I)

        Returns:
            New hidden state (B, G)
        """
        G = self.hidden_dim
        gates_ih = self.linear_ih(x_BI)  # (B, 3G)
        gates_hh = self.linear_hh(h_BG)  # (B, 3G)

        # Split into reset, update, new
        r_ih, z_ih, n_ih = gates_ih[:, :G], gates_ih[:, G : 2 * G], gates_ih[:, 2 * G :]
        r_hh, z_hh, n_hh = gates_hh[:, :G], gates_hh[:, G : 2 * G], gates_hh[:, 2 * G :]

        reset_BG = jax.nn.sigmoid(r_ih + r_hh)
        # Update bias toward -1: keeps old state by default (reference pattern)
        update_BG = jax.nn.sigmoid(z_ih + z_hh - 1.0)
        new_BG = jnp.tanh(n_ih + reset_BG * n_hh)

        h_new_BG = (1.0 - update_BG) * new_BG + update_BG * h_BG
        return h_new_BG


# ---------------------------------------------------------------------------
# RSSM (Recurrent State-Space Model)
# ---------------------------------------------------------------------------


class RSSM(nnx.Module):
    """Recurrent State-Space Model for Dreamer-v3.

    State = (h_BG, z_BZ) where:
        h_BG is the deterministic recurrent state (GRU hidden)
        z_BZ is the stochastic state (flattened product of categoricals)
    """

    def __init__(
        self,
        stoch_categories: int,
        stoch_classes: int,
        deter_dim: int,
        action_dim: int,
        embed_dim: int,
        hidden_dim: int,
        activation: Callable,
        unimix: float,
        rngs: nnx.Rngs,
    ):
        self.stoch_categories = stoch_categories
        self.stoch_classes = stoch_classes
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_categories * stoch_classes
        self.unimix = unimix

        # Pre-GRU: project (z, action) -> GRU input
        self.pre_gru = nnx.Linear(self.stoch_dim + action_dim, hidden_dim, rngs=rngs)
        self.pre_gru_act = activation
        self.gru = GRUCell(hidden_dim, deter_dim, rngs=rngs)

        # Prior: h -> prior_logits
        self.prior_mlp = MLP(deter_dim, (hidden_dim,), activation, rngs)
        self.prior_head = nnx.Linear(hidden_dim, self.stoch_dim, rngs=rngs)

        # Posterior: (h, embed) -> post_logits
        self.post_mlp = MLP(deter_dim + embed_dim, (hidden_dim,), activation, rngs)
        self.post_head = nnx.Linear(hidden_dim, self.stoch_dim, rngs=rngs)

    def initial_state(self, batch_size: int) -> tuple[jax.Array, jax.Array]:
        """Return zero-initialized (h_BG, z_BZ)."""
        h_BG = jnp.zeros((batch_size, self.deter_dim), dtype=jnp.float32)
        z_BZ = jnp.zeros((batch_size, self.stoch_dim), dtype=jnp.float32)
        return h_BG, z_BZ

    def _get_prior(self, h_BG: jax.Array) -> jax.Array:
        """Compute prior logits from deterministic state."""
        x = self.prior_mlp(h_BG)
        return self.prior_head(x).reshape(h_BG.shape[0], self.stoch_categories, self.stoch_classes)

    def _get_posterior(self, h_BG: jax.Array, embed_BF: jax.Array) -> jax.Array:
        """Compute posterior logits from deterministic state and observation embedding."""
        x = jnp.concatenate([h_BG, embed_BF], axis=-1)
        x = self.post_mlp(x)
        return self.post_head(x).reshape(h_BG.shape[0], self.stoch_categories, self.stoch_classes)

    def _sample_stoch(self, logits_BCL: jax.Array, rng: jax.Array) -> jax.Array:
        """Sample from categorical with straight-through gradients and uniform mixing.

        Args:
            logits_BCL: (B, C, L) raw logits
            rng: PRNG key

        Returns:
            Flattened stochastic state (B, Z) where Z = C * L
        """
        B = logits_BCL.shape[0]
        probs_BCL = jax.nn.softmax(logits_BCL, axis=-1)
        # Uniform mixing to prevent categorical collapse
        uniform = jnp.ones_like(probs_BCL) / self.stoch_classes
        probs_BCL = (1.0 - self.unimix) * probs_BCL + self.unimix * uniform
        # Sample one-hot with straight-through gradients
        flat_logits = jnp.log(probs_BCL + 1e-8).reshape(B * self.stoch_categories, self.stoch_classes)
        sample_idx = jax.random.categorical(rng, flat_logits)
        sample_onehot = jax.nn.one_hot(sample_idx, self.stoch_classes, dtype=jnp.float32)
        sample_BCL = sample_onehot.reshape(B, self.stoch_categories, self.stoch_classes)
        # Straight-through: forward uses sample, backward uses probs
        z_BCL = sample_BCL - jax.lax.stop_gradient(probs_BCL) + probs_BCL
        return z_BCL.reshape(B, self.stoch_dim)

    def _advance_deterministic(self, h_BG: jax.Array, z_BZ: jax.Array, action_BA: jax.Array) -> jax.Array:
        """Advance GRU: (h, z, action) -> h'."""
        # Action normalization from reference: action / max(1, |action|)
        action_norm = action_BA / jnp.maximum(1.0, jnp.abs(action_BA))
        x = jnp.concatenate([z_BZ, action_norm], axis=-1)
        x = self.pre_gru_act(self.pre_gru(x))
        return self.gru(h_BG, x)

    def obs_step(
        self, h_BG: jax.Array, z_BZ: jax.Array, action_BA: jax.Array, embed_BF: jax.Array, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Posterior step: observe embedding, produce posterior sample.

        Returns:
            (h'_BG, z_post_BZ, prior_logits_BCL, post_logits_BCL)
        """
        h_BG = self._advance_deterministic(h_BG, z_BZ, action_BA)
        prior_logits_BCL = self._get_prior(h_BG)
        post_logits_BCL = self._get_posterior(h_BG, embed_BF)
        z_post_BZ = self._sample_stoch(post_logits_BCL, rng)
        return h_BG, z_post_BZ, prior_logits_BCL, post_logits_BCL

    def img_step(
        self, h_BG: jax.Array, z_BZ: jax.Array, action_BA: jax.Array, rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Prior step (imagination): no observation, sample from prior.

        Returns:
            (h'_BG, z_prior_BZ, prior_logits_BCL)
        """
        h_BG = self._advance_deterministic(h_BG, z_BZ, action_BA)
        prior_logits_BCL = self._get_prior(h_BG)
        z_prior_BZ = self._sample_stoch(prior_logits_BCL, rng)
        return h_BG, z_prior_BZ, prior_logits_BCL


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class DreamerEncoder(nnx.Module):
    """Observation encoder: obs -> embedding with symlog preprocessing."""

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.mlp = MLP(in_features, hidden_layer_sizes, activation, rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Encode observations with symlog preprocessing.

        Args:
            obs: (B, D) raw observations

        Returns:
            (B, H) embedding where H is the last hidden layer size
        """
        return self.mlp(symlog(obs))


class DreamerDecoder(nnx.Module):
    """Observation decoder: (h, z) -> obs reconstruction via symlog MSE."""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.mlp = MLP(state_dim, hidden_layer_sizes, activation, rngs)
        self.output_head = nnx.Linear(hidden_layer_sizes[-1], obs_dim, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        """Predict symlog(obs) from latent state.

        Args:
            state: (B, G+Z) concatenated deterministic + stochastic state

        Returns:
            (B, D) predicted symlog(obs)
        """
        x = self.mlp(state)
        return self.output_head(x)


# ---------------------------------------------------------------------------
# Reward / Continue predictors
# ---------------------------------------------------------------------------


class RewardPredictor(nnx.Module):
    """Predict reward from latent state via symlog two-hot distribution.

    Output head is zero-initialized (outscale=0) per Dreamer-v3 reference.
    """

    def __init__(
        self,
        state_dim: int,
        num_bins: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.mlp = MLP(state_dim, hidden_layer_sizes, activation, rngs)
        self.head = nnx.Linear(hidden_layer_sizes[-1], num_bins, rngs=rngs)
        # Zero-initialize output head
        self.head.kernel[...] = jnp.zeros_like(self.head.kernel[...])
        assert self.head.bias is not None
        self.head.bias[...] = jnp.zeros_like(self.head.bias[...])

    def __call__(self, state: jax.Array) -> jax.Array:
        """Returns raw logits (B, N) for symlog two-hot."""
        x = self.mlp(state)
        return self.head(x)


class ContinuePredictor(nnx.Module):
    """Predict episode continuation probability from latent state."""

    def __init__(
        self,
        state_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.mlp = MLP(state_dim, hidden_layer_sizes, activation, rngs)
        self.head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)

    def __call__(self, state: jax.Array) -> jax.Array:
        """Returns continue logit (B,)."""
        x = self.mlp(state)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Actor / Critic
# ---------------------------------------------------------------------------


class DreamerActor(nnx.Module):
    """Policy network operating on latent states.

    Discrete: categorical with straight-through + unimix.
    Continuous: tanh-bounded normal with clamped std.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_discrete: bool,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        action_range: tuple[float, float] | None = None,
        min_std: float = 0.1,
        max_std: float = 1.0,
        unimix: float = 0.01,
    ):
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.action_range = action_range
        self.min_std = min_std
        self.max_std = max_std
        self.unimix = unimix

        self.mlp = MLP(state_dim, hidden_layer_sizes, activation, rngs)
        if is_discrete:
            self.logits_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
            # outscale=0.01: scale down output head to keep initial policy near-uniform
            self.logits_head.kernel[...] = self.logits_head.kernel[...] * 0.01
            assert self.logits_head.bias is not None
            self.logits_head.bias[...] = jnp.zeros_like(self.logits_head.bias[...])
        else:
            self.mean_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
            self.log_std_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
            # outscale=0.01 for continuous mean head
            self.mean_head.kernel[...] = self.mean_head.kernel[...] * 0.01
            assert self.mean_head.bias is not None
            self.mean_head.bias[...] = jnp.zeros_like(self.mean_head.bias[...])

    def _mixed_probs(self, logits_BA: jax.Array) -> jax.Array:
        """Apply softmax + uniform mixing (discrete only)."""
        probs_BA = jax.nn.softmax(logits_BA, axis=-1)
        uniform = jnp.ones_like(probs_BA) / self.action_dim
        return (1.0 - self.unimix) * probs_BA + self.unimix * uniform

    def _bounded_std(self, log_std_BA: jax.Array) -> jax.Array:
        """Compute std from log_std with sigmoid bounding (continuous only)."""
        return self.min_std + (self.max_std - self.min_std) * jax.nn.sigmoid(log_std_BA)

    def _scale_mean(self, mean_raw_BA: jax.Array) -> jax.Array:
        """Tanh + optional action range scaling (continuous only)."""
        mean_BA = jnp.tanh(mean_raw_BA)
        if self.action_range is not None:
            lo, hi = self.action_range
            mean_BA = lo + (mean_BA + 1.0) * 0.5 * (hi - lo)
        return mean_BA

    def __call__(self, state_BX: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample action and compute log probability.

        Returns:
            (action, log_prob)
            Discrete: action (B,), log_prob (B,)
            Continuous: action (B, A), log_prob (B,)
        """
        features_BH = self.mlp(state_BX)

        if self.is_discrete:
            logits_BA = self.logits_head(features_BH)
            probs_BA = self._mixed_probs(logits_BA)
            # Sample with straight-through
            sample_B = jax.random.categorical(rng, jnp.log(probs_BA + 1e-8))
            onehot_BA = jax.nn.one_hot(sample_B, self.action_dim, dtype=jnp.float32)
            # Straight-through for action embedding in imagination
            action_BA = onehot_BA - jax.lax.stop_gradient(probs_BA) + probs_BA
            log_prob_B = jnp.log(probs_BA[jnp.arange(probs_BA.shape[0]), sample_B] + 1e-8)
            return action_BA, log_prob_B
        else:
            mean_raw_BA = self.mean_head(features_BH)
            log_std_BA = self.log_std_head(features_BH)
            std_BA = self._bounded_std(log_std_BA)
            mean_BA = self._scale_mean(mean_raw_BA)
            # Bounded normal: tanh on mean only, sample is plain Gaussian
            # (reference pattern — no tanh Jacobian, numerically stable)
            noise = jax.random.normal(rng, mean_BA.shape)
            action_BA = mean_BA + std_BA * noise
            # Normal log_prob (no Jacobian correction needed)
            log_prob_BA = -0.5 * ((action_BA - mean_BA) / std_BA) ** 2 - jnp.log(std_BA) - 0.5 * jnp.log(2 * jnp.pi)
            log_prob_B = log_prob_BA.sum(axis=-1)
            return action_BA, log_prob_B

    def act(self, state_BX: jax.Array, rng: jax.Array) -> jax.Array:
        """Sample action for environment stepping."""
        action, _ = self(state_BX, rng)
        if self.is_discrete:
            # Return integer action for env
            return jnp.argmax(action, axis=-1)
        return action

    def act_greedy(self, state_BX: jax.Array) -> jax.Array:
        """Deterministic action for evaluation (argmax/mean, no sampling)."""
        features_BH = self.mlp(state_BX)
        if self.is_discrete:
            logits_BA = self.logits_head(features_BH)
            return jnp.argmax(logits_BA, axis=-1)
        else:
            return self._scale_mean(self.mean_head(features_BH))

    def log_prob_of(self, state_BX: jax.Array, action: jax.Array) -> jax.Array:
        """Compute log probability of given actions (for REINFORCE on sg states).

        Args:
            state_BX: latent state (B, G+Z) — should be stop_gradient'd
            action: (B,) int action indices for discrete, (B, A) for continuous

        Returns:
            log_prob (B,)
        """
        features_BH = self.mlp(state_BX)
        if self.is_discrete:
            probs_BA = self._mixed_probs(self.logits_head(features_BH))
            return jnp.log(probs_BA[jnp.arange(probs_BA.shape[0]), action] + 1e-8)
        else:
            std_BA = self._bounded_std(self.log_std_head(features_BH))
            mean_BA = self._scale_mean(self.mean_head(features_BH))
            log_prob_BA = -0.5 * ((action - mean_BA) / std_BA) ** 2 - jnp.log(std_BA) - 0.5 * jnp.log(2 * jnp.pi)
            return log_prob_BA.sum(axis=-1)

    def entropy(self, state_BX: jax.Array) -> jax.Array:
        """Compute entropy of the policy distribution."""
        features_BH = self.mlp(state_BX)
        if self.is_discrete:
            probs_BA = self._mixed_probs(self.logits_head(features_BH))
            return -jnp.sum(probs_BA * jnp.log(probs_BA + 1e-8), axis=-1)
        else:
            std_BA = self._bounded_std(self.log_std_head(features_BH))
            # Gaussian entropy: 0.5 * log(2*pi*e * sigma^2) per dimension
            return 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * std_BA**2), axis=-1)


class DreamerCritic(nnx.Module):
    """Value network via symlog two-hot distribution.

    Zero-initialized output head (outscale=0) per Dreamer-v3 reference.
    """

    def __init__(
        self,
        state_dim: int,
        num_bins: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.mlp = MLP(state_dim, hidden_layer_sizes, activation, rngs)
        self.head = nnx.Linear(hidden_layer_sizes[-1], num_bins, rngs=rngs)
        # Zero-initialize output head
        self.head.kernel[...] = jnp.zeros_like(self.head.kernel[...])
        assert self.head.bias is not None
        self.head.bias[...] = jnp.zeros_like(self.head.bias[...])

    def __call__(self, state: jax.Array) -> jax.Array:
        """Returns raw logits (B, N) for symlog two-hot."""
        x = self.mlp(state)
        return self.head(x)

    def value(self, state: jax.Array) -> jax.Array:
        """Decode value to scalar."""
        logits_BN = self(state)
        return twohot_decode(logits_BN, self.num_bins)


# ---------------------------------------------------------------------------
# World Model wrapper
# ---------------------------------------------------------------------------


class DreamerWorldModel(nnx.Module):
    """Bundles all Dreamer-v3 components for single-optimizer training.

    Following MuZero's MuZeroModel pattern — single wrapper for joint optimization.
    Includes actor + critic since Dreamer-v3 reference uses a single optimizer.
    """

    def __init__(
        self,
        encoder: DreamerEncoder,
        rssm: RSSM,
        decoder: DreamerDecoder,
        reward_pred: RewardPredictor,
        continue_pred: ContinuePredictor,
        actor: DreamerActor,
        critic: DreamerCritic,
    ):
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_pred = reward_pred
        self.continue_pred = continue_pred
        self.actor = actor
        self.critic = critic
