"""REPPO networks: RMSNorm tanh-Gaussian policy and HL-Gauss distributional critic with auxiliary heads.

Paper: "Relative Entropy Pathwise Policy Optimization" (REPPO).
Port of RL-X's reppo flax networks (flax.linen). Defaults of ``nnx.Linear`` (lecun_normal kernel,
zero bias) and ``nnx.RMSNorm`` (epsilon 1e-6, scale ones) match flax.linen's ``nn.Dense`` /
``nn.RMSNorm``. The activation is swish (SiLU).

Dimension key:
    B: batch size
    D: observation dimension
    A: action dimension
    H: hidden layer size
    N: number of bins of the categorical value distribution
    K: number of action samples per state
"""

import math

import distrax
import jax
from flax import nnx
from jax import numpy as jnp
from jax.scipy.special import erf


class REPPOPolicy(nnx.Module):
    """Tanh-squashed Gaussian policy; actions lie in (-1, 1) (rescaling happens outside).

    The entropy and KL coefficients are log-parameters on the policy so that the policy optimizer
    trains them together with the network weights.
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        *,
        policy_obs_dim: int | None = None,
        min_std: float = 0.0,
        init_entropy_coefficient: float = 0.01,
        init_kl_coefficient: float = 0.01,
    ):
        # in_features is the dimension AFTER slicing obs to policy_obs_dim.
        self.policy_obs_dim = policy_obs_dim
        self.min_std = min_std
        self.linear_in = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.norm_in = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.linear_hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm_hidden = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.linear_out = nnx.Linear(hidden_dim, action_dim * 2, rngs=rngs)
        self.log_entropy_coefficient = nnx.Param(jnp.full((1,), math.log(init_entropy_coefficient), dtype=jnp.float32))
        self.log_kl_coefficient = nnx.Param(jnp.full((1,), math.log(init_kl_coefficient), dtype=jnp.float32))

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Returns the pre-tanh Gaussian (loc, log_std), each of shape (..., A)."""
        if self.policy_obs_dim is not None:
            obs = obs[..., : self.policy_obs_dim]
        h_BH = nnx.swish(self.norm_in(self.linear_in(obs)))
        h_BH = nnx.swish(self.norm_hidden(self.linear_hidden(h_BH)))
        out_B2A = self.linear_out(h_BH)
        loc_BA, log_std_BA = jnp.split(out_B2A, 2, axis=-1)
        return loc_BA, log_std_BA

    def _action_dist(self, loc: jax.Array, log_std: jax.Array) -> distrax.Transformed:
        base = distrax.MultivariateNormalDiag(loc=loc, scale_diag=jnp.exp(log_std) + self.min_std)
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))

    def sample_and_log_prob(
        self,
        loc: jax.Array,
        log_std: jax.Array,
        rng: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:
        """Returns the action of shape sample_shape + loc.shape and its log prob of shape sample_shape + loc.shape[:-1]."""
        action_dist = self._action_dist(loc, log_std)
        action, log_prob = action_dist.sample_and_log_prob(seed=rng, sample_shape=sample_shape)
        return action, log_prob  # pyright: ignore[reportReturnType]

    def log_prob(self, loc: jax.Array, log_std: jax.Array, action: jax.Array) -> jax.Array:
        """Log prob of a tanh-squashed action; the action is clipped away from +-1 before the atanh."""
        eps = 1e-6
        clipped_action = jnp.clip(action, -1.0 + eps, 1.0 - eps)
        return self._action_dist(loc, log_std).log_prob(clipped_action)  # pyright: ignore[reportReturnType]

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Deterministic action (rng unused, kept for the make_eval_act interface)."""
        loc_BA, _ = self(obs)
        return jnp.tanh(loc_BA)


class REPPOCritic(nnx.Module):
    """HL-Gauss distributional critic on concat(obs, action) with a reward/feature prediction head.

    The critic uses the full observation. ``features`` feeds both the value head and the
    prediction head (next-step features and reward).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_bins: int,
        v_min: float,
        v_max: float,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.linear_in = nnx.Linear(obs_dim + action_dim, hidden_dim, rngs=rngs)
        self.norm_in = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.linear_features = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        # Value head.
        self.critic_linear = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.critic_norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.critic_out = nnx.Linear(hidden_dim, num_bins, rngs=rngs)
        # Prediction head: output is [reward (1), next features (hidden_dim)].
        self.pred_linear = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.pred_norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.pred_out = nnx.Linear(hidden_dim, hidden_dim + 1, rngs=rngs)
        # Trainable HL-Gauss distribution of the value 0, added (scaled by 40) to the logits.
        cdf_N1 = erf(self.support / (jnp.sqrt(2) * self.sigma))
        zero_distribution_N = (cdf_N1[1:] - cdf_N1[:-1]) / (cdf_N1[-1] - cdf_N1[0])
        self.zero_distribution = nnx.Param(zero_distribution_N)

    @property
    def bin_width(self) -> float:
        return (self.v_max - self.v_min) / (self.num_bins - 1)

    @property
    def centers(self) -> jax.Array:
        """Bin centers, shape (N,)."""
        return jnp.linspace(self.v_min, self.v_max, self.num_bins, dtype=jnp.float32)

    @property
    def support(self) -> jax.Array:
        """Bin edges, shape (N + 1,)."""
        half_width = self.bin_width / 2
        return jnp.linspace(self.v_min - half_width, self.v_max + half_width, self.num_bins + 1, dtype=jnp.float32)

    @property
    def sigma(self) -> float:
        """HL-Gauss standard deviation: 0.75 bin widths."""
        return 0.75 * self.bin_width

    def __call__(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Returns (features (B, H), logits (B, N), predicted features (B, H), predicted reward (B, 1))."""
        x_BD = jnp.concatenate([obs, action], axis=-1)
        features_BH = nnx.swish(self.norm_in(self.linear_in(x_BD)))
        features_BH = self.linear_features(features_BH)
        activated_BH = nnx.swish(features_BH)

        critic_h_BH = nnx.swish(self.critic_norm(self.critic_linear(activated_BH)))
        logits_BN = self.critic_out(critic_h_BH) + 40.0 * self.zero_distribution[...]

        pred_h_BH = nnx.swish(self.pred_norm(self.pred_linear(activated_BH)))
        pred = self.pred_out(pred_h_BH)  # (B, 1 + H): reward first, then next features
        pred_features_BH = pred[..., 1:]
        pred_reward_B1 = pred[..., :1]
        return features_BH, logits_BN, pred_features_BH, pred_reward_B1

    def value(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Expected value under the categorical distribution, shape (B,)."""
        _, logits_BN, _, _ = self(obs, action)
        return jax.nn.softmax(logits_BN, axis=-1) @ self.centers
