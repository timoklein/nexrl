"""FastSAC networks: LayerNorm policy and distributional (C51-style) twin critic.

Port of RL-X's flax.linen FastSAC (rl_x/algorithms/fastsac/flax_full_jit). Defaults of
``nnx.Linear`` (lecun_normal kernel, zero bias) and ``nnx.LayerNorm`` (epsilon 1e-6, scale and
bias) match flax.linen's ``nn.Dense`` / ``nn.LayerNorm``.

Dimension key:
    B: batch size
    D: observation dimension
    A: action dimension
    H: hidden layer size
    C: number of critics in the ensemble
    N: number of atoms of the categorical value distribution
"""

from collections.abc import Sequence

import distrax
import jax
from flax import nnx
from jax import numpy as jnp


class LayerNormMLP(nnx.Module):
    """Blocks of Linear -> LayerNorm -> SiLU."""

    def __init__(self, in_features: int, hidden_layer_sizes: Sequence[int], rngs: nnx.Rngs):
        layers = []
        norms = []
        current_in = in_features
        for size in hidden_layer_sizes:
            layers.append(nnx.Linear(current_in, size, rngs=rngs))
            norms.append(nnx.LayerNorm(size, rngs=rngs))
            current_in = size
        self.layers = nnx.List(layers)
        self.norms = nnx.List(norms)
        self.out_features = current_in

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer, norm in zip(self.layers, self.norms):
            x = nnx.silu(norm(layer(x)))
        return x


class FastSACPolicy(nnx.Module):
    """Tanh-squashed Gaussian policy with bounded log std and zero-initialized heads."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        log_std_range: tuple[float, float],
        rngs: nnx.Rngs,
        *,
        policy_obs_dim: int | None = None,
    ):
        # in_features is the dimension AFTER slicing obs to policy_obs_dim.
        self.policy_obs_dim = policy_obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.log_std_range = log_std_range
        self.features = LayerNormMLP(in_features, hidden_layer_sizes, rngs)
        self.action_mean = nnx.Linear(
            self.features.out_features,
            action_dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.action_log_std = nnx.Linear(
            self.features.out_features,
            action_dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    @property
    def action_loc(self) -> float:
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self) -> float:
        return (self.action_range[1] - self.action_range[0]) / 2

    def _mean_log_std(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Every forward path (__call__, act) goes through here; the policy sees obs[..., :policy_obs_dim].
        if self.policy_obs_dim is not None:
            obs = obs[..., : self.policy_obs_dim]
        features_BH = self.features(obs)
        mean_BA = self.action_mean(features_BH)
        log_std_BA = jnp.tanh(self.action_log_std(features_BH))
        log_std_min, log_std_max = self.log_std_range
        log_std_BA = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std_BA + 1)
        return mean_BA, log_std_BA

    def _action_dist(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        # Pre-tanh base Gaussian; the tanh correction is applied by hand in __call__.
        mean_BA, log_std_BA = self._mean_log_std(obs)
        return distrax.MultivariateNormalDiag(loc=mean_BA, scale_diag=jnp.exp(log_std_BA))

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        raw_BA = action_dist.sample(seed=rng)
        action_BA = self.action_loc + jnp.tanh(raw_BA) * self.action_scale
        # Reference correction (eps inside the log), not distrax.Tanh's exact log-det.
        tanh_correction_B = jnp.log(1 - jnp.tanh(raw_BA) ** 2 + 1e-6).sum(axis=-1)
        scale_correction = jnp.log(self.action_scale + 1e-6) * self.action_dim  # per-dim term, summed
        log_prob_B = action_dist.log_prob(raw_BA) - tanh_correction_B - scale_correction
        return action_BA, log_prob_B  # pyright: ignore[reportReturnType]

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Deterministic action (rng unused, kept for the make_eval_act interface)."""
        mean_BA, _ = self._mean_log_std(obs)
        return self.action_loc + jnp.tanh(mean_BA) * self.action_scale


class _CategoricalQ(nnx.Module):
    """One critic: LayerNormMLP on concat(obs, action) -> logits over atoms."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int,
        hidden_layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.features = LayerNormMLP(obs_dim + action_dim, hidden_layer_sizes, rngs)
        self.logits = nnx.Linear(self.features.out_features, num_atoms, rngs=rngs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x_BD = jnp.concatenate([obs, action], axis=-1)
        return self.logits(self.features(x_BD))  # (B, N)


class FastSACCritic(nnx.Module):
    """Ensemble of distributional critics with independent parameters (vmapped)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
        num_critics: int = 2,
    ):
        self.num_critics = num_critics
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        @nnx.split_rngs(splits=num_critics)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def _make(rngs: nnx.Rngs) -> _CategoricalQ:
            return _CategoricalQ(obs_dim, action_dim, num_atoms, hidden_layer_sizes, rngs)

        self.ensemble = _make(rngs)

    @property
    def support(self) -> jax.Array:
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms, dtype=jnp.float32)  # (N,)

    def __call__(self, obs_BD: jax.Array, action_BA: jax.Array) -> jax.Array:
        @nnx.vmap(in_axes=(0, None, None), out_axes=0)
        def _forward(model: _CategoricalQ, obs: jax.Array, action: jax.Array) -> jax.Array:
            return model(obs, action)

        return _forward(self.ensemble, obs_BD, action_BA)  # (C, B, N)

    def q_values(self, obs_BD: jax.Array, action_BA: jax.Array) -> jax.Array:
        probs_CBN = jax.nn.softmax(self(obs_BD, action_BA), axis=-1)
        return (probs_CBN * self.support).sum(axis=-1)  # (C, B)
