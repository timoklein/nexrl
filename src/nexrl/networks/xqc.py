"""XQC networks: BatchNorm MLP policy and distributional twin critic with weight normalization.

Port of RL-X's xqc flax networks (flax.linen). The trunk is an input BatchNorm followed by blocks
of bias-free Dense -> BatchNorm -> ReLU (no skip connections, no output norm). The heads have an
orthogonal kernel and a zero bias. ``weight_norm`` keeps every Linear kernel at unit column norm.

Dimension key:
    B: batch size
    D: observation dimension
    A: action dimension
    H: hidden layer size
    C: number of critics in the ensemble
    N: number of atoms of the categorical value distribution
"""

import math

import jax
from flax import nnx
from jax import numpy as jnp


_SQRT2 = math.sqrt(2.0)


def _linear(in_features: int, out_features: int, rngs: nnx.Rngs, kernel_scale: float = _SQRT2) -> nnx.Linear:
    """Head layer: orthogonal kernel, zero bias."""
    return nnx.Linear(
        in_features,
        out_features,
        kernel_init=nnx.initializers.orthogonal(kernel_scale),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
    )


def _batch_norm(features: int, rngs: nnx.Rngs) -> nnx.BatchNorm:
    return nnx.BatchNorm(features, momentum=0.99, epsilon=1e-3, rngs=rngs)


class _Block(nnx.Module):
    """Linear (no bias) -> BN -> ReLU."""

    def __init__(self, in_features: int, hidden_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            in_features,
            hidden_dim,
            use_bias=False,
            kernel_init=nnx.initializers.orthogonal(_SQRT2),
            rngs=rngs,
        )
        self.norm = _batch_norm(hidden_dim, rngs)

    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        return nnx.relu(self.norm(self.linear(x), use_running_average=not train))


class _Trunk(nnx.Module):
    """Input BatchNorm -> blocks."""

    def __init__(self, in_features: int, hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        self.input_norm = _batch_norm(in_features, rngs)
        self.blocks = nnx.List([_Block(in_features if i == 0 else hidden_dim, hidden_dim, rngs) for i in range(num_blocks)])

    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = self.input_norm(x, use_running_average=not train)
        for block in self.blocks:
            x = block(x, train)
        return x


class XQCPolicy(nnx.Module):
    """Tanh-squashed Gaussian policy; actions lie in [-1, 1] (rescaling happens outside)."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_dim: int,
        num_blocks: int,
        rngs: nnx.Rngs,
        log_std_range: tuple[float, float] = (-10.0, 2.0),
    ):
        self.log_std_range = log_std_range
        self.trunk = _Trunk(in_features, hidden_dim, num_blocks, rngs)
        self.mean_head = _linear(hidden_dim, action_dim, rngs)
        self.log_std_head = _linear(hidden_dim, action_dim, rngs, kernel_scale=1.0)

    def _mean_log_std(self, obs: jax.Array, train: bool) -> tuple[jax.Array, jax.Array]:
        features_BH = self.trunk(obs, train)
        mean_BA = self.mean_head(features_BH)
        log_std_BA = jnp.clip(self.log_std_head(features_BH), *self.log_std_range)
        return mean_BA, log_std_BA

    def __call__(self, obs: jax.Array, rng: jax.Array, train: bool) -> tuple[jax.Array, jax.Array]:
        mean_BA, log_std_BA = self._mean_log_std(obs, train)
        noise_BA = jax.random.normal(rng, mean_BA.shape)
        u_BA = mean_BA + jnp.exp(log_std_BA) * noise_BA
        action_BA = jnp.tanh(u_BA)
        gaussian_log_prob_BA = -0.5 * noise_BA**2 - 0.5 * jnp.log(2.0 * jnp.pi) - log_std_BA
        tanh_correction_BA = 2.0 * (jnp.log(2.0) - u_BA - jax.nn.softplus(-2.0 * u_BA))
        log_prob_B = jnp.sum(gaussian_log_prob_BA - tanh_correction_BA, axis=-1)
        return action_BA, log_prob_B

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Deterministic eval-mode action (rng unused, kept for the make_eval_act interface)."""
        mean_BA, _ = self._mean_log_std(obs, False)
        return jnp.tanh(mean_BA)


class _CategoricalQ(nnx.Module):
    """One critic: trunk on concat(obs, action) -> logits over atoms."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int,
        hidden_dim: int,
        num_blocks: int,
        rngs: nnx.Rngs,
    ):
        self.trunk = _Trunk(obs_dim + action_dim, hidden_dim, num_blocks, rngs)
        self.logits = _linear(hidden_dim, num_atoms, rngs)

    def __call__(self, obs: jax.Array, action: jax.Array, train: bool) -> jax.Array:
        obs_action = jnp.concatenate([obs, action], axis=-1)  # (B, D + A)
        return self.logits(self.trunk(obs_action, train))  # (B, N)


class XQCCritic(nnx.Module):
    """Ensemble of distributional critics with independent params and batch stats (vmapped)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        num_blocks: int,
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
            return _CategoricalQ(obs_dim, action_dim, num_atoms, hidden_dim, num_blocks, rngs)

        self.ensemble = _make(rngs)

    @property
    def support(self) -> jax.Array:
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms, dtype=jnp.float32)  # (N,)

    def __call__(self, obs_BD: jax.Array, action_BA: jax.Array, train: bool) -> jax.Array:
        @nnx.vmap(in_axes=(0, None, None), out_axes=0)
        def _forward(model: _CategoricalQ, obs: jax.Array, action: jax.Array) -> jax.Array:
            return model(obs, action, train)

        return _forward(self.ensemble, obs_BD, action_BA)  # (C, B, N)

    def q_values(self, obs_BD: jax.Array, action_BA: jax.Array, train: bool = False) -> jax.Array:
        probs_CBN = jax.nn.softmax(self(obs_BD, action_BA, train), axis=-1)
        return (probs_CBN * self.support).sum(axis=-1)  # (C, B)


def weight_norm(model: nnx.Module) -> None:
    """Normalize Linear kernels in place to unit L2 norm per output column (pure jnp, jit-safe).

    Only the input axis (-2) is reduced, so the leading ensemble axis of vmapped critic parameters
    is preserved. Biases and BatchNorm parameters are untouched. Like the reference, there is no eps.
    """
    for _, module in nnx.iter_modules(model):
        if isinstance(module, nnx.Linear):
            kernel = jnp.asarray(module.kernel[...])
            module.kernel[...] = kernel / jnp.linalg.norm(kernel, axis=-2, keepdims=True)
