"""Tests for MuZero algorithm."""

import jax
import jax.numpy as jnp
import pytest

from nexrl import MuZeroConfig, get_train_fn, train_muzero


@pytest.fixture
def cartpole_config():
    return MuZeroConfig(
        env="CartPole-v1",
        num_envs=4,
        buffer_size=500,
        fill_buffer=50,
        batch_size=16,
        num_epochs=2,
        learning_rate=5e-4,
        max_grad_norm=10.0,
        total_timesteps=2000,
        eval_freq=500,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=1.0,
        reward_coef=1.0,
        num_simulations=4,
        latent_dim=32,
        unroll_steps=3,
        n_steps=5,
        gradient_scale=0.5,
        value_min=-10.0,
        value_max=10.0,
        value_num_atoms=21,
        reward_min=-10.0,
        reward_max=10.0,
        reward_num_atoms=21,
        run_evaluation=True,
    )


def test_muzero_runs(cartpole_config):
    """Test that MuZero training completes without errors."""
    rng = jax.random.PRNGKey(0)
    _state, metrics = jax.jit(train_muzero, static_argnums=(0,))(cartpole_config, rng)
    assert "eval_returns" in metrics
    assert "eval_lengths" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "reward_loss" in metrics
    assert "entropy" in metrics
    assert "global_step" in metrics
    assert "train_return" in metrics


def test_muzero_get_train_fn(cartpole_config):
    """Test that MuZero is registered and accessible via get_train_fn."""
    train_fn = get_train_fn("muzero")
    assert train_fn is train_muzero

    rng = jax.random.PRNGKey(42)
    _state, metrics = jax.jit(train_fn, static_argnums=(0,))(cartpole_config, rng)
    assert metrics["eval_returns"].shape[0] > 0


def test_muzero_vmap(cartpole_config):
    """Test that MuZero can be vmapped over seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmap_train = jax.jit(jax.vmap(train_muzero, in_axes=(None, 0)), static_argnums=(0,))
    _state, metrics = vmap_train(cartpole_config, keys)
    assert metrics["eval_returns"].shape[0] == 2


def test_muzero_from_yaml():
    """Test that MuZero config loads from YAML."""
    config = MuZeroConfig.from_yaml("configs/muzero/gymnax/cartpole.yaml")
    assert config.env == "CartPole-v1"
    assert config.latent_dim == 64
    assert config.unroll_steps == 5
    assert config.n_steps == 5
    assert config.value_num_atoms == 21
    assert config.reward_num_atoms == 21


def test_muzero_categorical_transforms():
    """Test the scalar <-> categorical roundtrip."""
    from nexrl.algos.muzero import categorical_to_scalar, scalar_to_categorical

    scalars = jnp.array([0.0, 1.0, -1.0, 5.0, -5.0])
    cats = jax.vmap(lambda s: scalar_to_categorical(s, -10.0, 10.0, 51))(scalars)
    assert cats.shape == (5, 51)
    # Check each is a valid distribution (sums to ~1)
    assert jnp.allclose(cats.sum(axis=-1), 1.0, atol=1e-5)

    # Roundtrip
    recovered = jax.vmap(lambda c: categorical_to_scalar(c, -10.0, 10.0, 51))(cats)
    assert jnp.allclose(recovered, scalars, atol=0.1)
