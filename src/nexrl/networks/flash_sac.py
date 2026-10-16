"""FlashSAC networks: BatchNorm/RMSNorm residual policy and distributional twin critic.

Port of RL-X's flashsac flax_full_jit networks (flax.linen). Linear layers have no bias and an
orthogonal kernel, the heads have an orthogonal kernel and zero bias, and ``project_params``
keeps kernels and norm parameters on fixed-norm manifolds after every optimizer step.

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


def _unit_linear(in_features: int, out_features: int, rngs: nnx.Rngs) -> nnx.Linear:
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=False,
        kernel_init=nnx.initializers.orthogonal(1.0),
        rngs=rngs,
    )


def _batch_norm(features: int, rngs: nnx.Rngs) -> nnx.BatchNorm:
    return nnx.BatchNorm(features, momentum=0.99, epsilon=1e-5, rngs=rngs)


class _Embedder(nnx.Module):
    """BatchNorm on the input, then a bias-free linear projection to the hidden size."""

    def __init__(self, in_features: int, hidden_dim: int, rngs: nnx.Rngs):
        self.norm = _batch_norm(in_features, rngs)
        self.linear = _unit_linear(in_features, hidden_dim, rngs)

    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = self.norm(x, use_running_average=not train)
        return self.linear(x)


class _Block(nnx.Module):
    """Inverted-bottleneck residual block: Linear -> BN -> ReLU -> Linear -> BN -> ReLU, + input."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs, expansion: int = 4):
        self.linear_up = _unit_linear(hidden_dim, hidden_dim * expansion, rngs)
        self.norm_up = _batch_norm(hidden_dim * expansion, rngs)
        self.linear_down = _unit_linear(hidden_dim * expansion, hidden_dim, rngs)
        self.norm_down = _batch_norm(hidden_dim, rngs)

    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        residual = x
        x = nnx.relu(self.norm_up(self.linear_up(x), use_running_average=not train))
        x = nnx.relu(self.norm_down(self.linear_down(x), use_running_average=not train))
        return x + residual


class _Trunk(nnx.Module):
    """Embedder -> residual blocks -> RMSNorm."""

    def __init__(self, in_features: int, hidden_dim: int, num_blocks: int, rngs: nnx.Rngs):
        self.embedder = _Embedder(in_features, hidden_dim, rngs)
        self.blocks = nnx.List([_Block(hidden_dim, rngs) for _ in range(num_blocks)])
        self.out_norm = nnx.RMSNorm(hidden_dim, epsilon=1e-6, rngs=rngs)

    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = self.embedder(x, train)
        for block in self.blocks:
            x = block(x, train)
        return self.out_norm(x)


class FlashSACPolicy(nnx.Module):
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
        self.mean_head = nnx.Linear(
            hidden_dim,
            action_dim,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.log_std_head = nnx.Linear(
            hidden_dim,
            action_dim,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def _mean_std(self, obs: jax.Array, train: bool) -> tuple[jax.Array, jax.Array]:
        features_BH = self.trunk(obs, train)
        mean_BA = self.mean_head(features_BH)
        raw_log_std_BA = self.log_std_head(features_BH)
        log_std_min, log_std_max = self.log_std_range
        log_std_BA = log_std_min + (log_std_max - log_std_min) * 0.5 * (1.0 + jnp.tanh(raw_log_std_BA))
        return mean_BA, jnp.exp(log_std_BA)

    def __call__(self, obs: jax.Array, rng: jax.Array, train: bool) -> tuple[jax.Array, jax.Array]:
        mean_BA, std_BA = self._mean_std(obs, train)
        noise_BA = jax.random.normal(rng, mean_BA.shape)
        u_BA = mean_BA + std_BA * noise_BA
        action_BA = jnp.tanh(u_BA)
        gaussian_log_prob_BA = -0.5 * noise_BA**2 - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(std_BA)
        tanh_correction_BA = 2.0 * (jnp.log(2.0) - u_BA - jax.nn.softplus(-2.0 * u_BA))
        log_prob_B = jnp.sum(gaussian_log_prob_BA - tanh_correction_BA, axis=-1)
        return action_BA, log_prob_B

    def act_with_noise(self, obs: jax.Array, noise_BA: jax.Array) -> jax.Array:
        """Eval-mode action with externally supplied noise: tanh(mean + std * noise)."""
        mean_BA, std_BA = self._mean_std(obs, False)
        return jnp.tanh(mean_BA + std_BA * noise_BA)

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Deterministic eval-mode action (rng unused, kept for the make_eval_act interface)."""
        mean_BA, _ = self._mean_std(obs, False)
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
        self.logits = nnx.Linear(
            hidden_dim,
            num_atoms,
            kernel_init=nnx.initializers.orthogonal(1.0),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array, action: jax.Array, train: bool) -> jax.Array:
        x_BD = jnp.concatenate([obs, action], axis=-1)
        return self.logits(self.trunk(x_BD, train))  # (B, N)


class FlashSACCritic(nnx.Module):
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


def project_params(model: nnx.Module) -> None:
    """Project parameters in place (pure jnp, usable under jit / nnx.fori_loop).

    Linear kernels get unit L2 norm over the input axis, RMSNorm scales get norm sqrt(d), and the
    (scale, bias) pair of each BatchNorm is jointly rescaled to norm sqrt(d). Only the last two
    axes are reduced, so the leading ensemble axis of vmapped critic parameters is preserved.
    """
    for _, module in nnx.iter_modules(model):
        if isinstance(module, nnx.Linear):
            kernel = jnp.asarray(module.kernel[...])
            norm = jnp.asarray(jnp.linalg.norm(kernel, axis=-2, keepdims=True))
            module.kernel[...] = kernel / jnp.where(norm < 1e-8, 1.0, norm)
        elif isinstance(module, nnx.RMSNorm):
            assert module.scale is not None
            scale = module.scale[...]
            dim = scale.shape[-1]
            module.scale[...] = scale * math.sqrt(dim) * jax.lax.rsqrt(jnp.sum(scale * scale, axis=-1, keepdims=True) + 1e-8)
        elif isinstance(module, nnx.BatchNorm):
            assert module.scale is not None and module.bias is not None
            scale = module.scale[...]
            bias = module.bias[...]
            dim = scale.shape[-1]
            sq = jnp.sum(scale * scale + bias * bias, axis=-1, keepdims=True)
            factor = math.sqrt(dim) * jax.lax.rsqrt(sq + 1e-8)
            module.scale[...] = scale * factor
            module.bias[...] = bias * factor
