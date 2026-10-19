"""FastTD3 networks: deterministic ReLU policy and distributional (C51-style) twin critic.

Port of RL-X's flax.linen FastTD3 (rl_x/algorithms/fasttd3/flax_full_jit). Defaults of
``nnx.Linear`` (lecun_normal kernel, zero bias) match flax.linen's ``nn.Dense``. There is no
LayerNorm anywhere.

Dimension key:
    B: batch size
    D: observation dimension
    A: action dimension
    H: hidden layer size
    C: number of critics in the ensemble
    N: number of atoms of the categorical value distribution
"""

from collections.abc import Sequence

import jax
from flax import nnx
from jax import numpy as jnp


class _ReLUMLP(nnx.Module):
    """Blocks of Linear -> ReLU."""

    def __init__(self, in_features: int, hidden_layer_sizes: Sequence[int], rngs: nnx.Rngs):
        layers = []
        current_in = in_features
        for size in hidden_layer_sizes:
            layers.append(nnx.Linear(current_in, size, rngs=rngs))
            current_in = size
        self.layers = nnx.List(layers)
        self.out_features = current_in

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = nnx.relu(layer(x))
        return x


class FastTD3Policy(nnx.Module):
    """Deterministic tanh policy with a small-init output head."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.features = _ReLUMLP(in_features, hidden_layer_sizes, rngs)
        self.action_mean = nnx.Linear(
            self.features.out_features,
            action_dim,
            kernel_init=nnx.initializers.normal(0.01),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        features_BH = self.features(obs)
        return jnp.tanh(self.action_mean(features_BH))  # (B, A) in [-1, 1]

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Same as __call__ (rng unused, kept for the make_eval_act interface)."""
        return self(obs)


class _CategoricalQ(nnx.Module):
    """One critic: ReLU MLP on concat(obs, action) -> logits over atoms."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_atoms: int,
        hidden_layer_sizes: Sequence[int],
        rngs: nnx.Rngs,
    ):
        self.features = _ReLUMLP(obs_dim + action_dim, hidden_layer_sizes, rngs)
        self.logits = nnx.Linear(self.features.out_features, num_atoms, rngs=rngs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x_BD = jnp.concatenate([obs, action], axis=-1)
        return self.logits(self.features(x_BD))  # (B, N)


class FastTD3Critic(nnx.Module):
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
