"""Tests for MrQ algorithm."""

import jax
import pytest

from nexrl import MrQConfig, get_train_fn, train_mrq


@pytest.fixture
def config():
    return MrQConfig(
        env="CartPole-v1",
        num_envs=4,
        buffer_size=500,
        fill_buffer=50,
        batch_size=16,
        total_timesteps=2000,
        eval_freq=500,
        zs_dim=32,
        za_dim=16,
        zsa_dim=32,
        hidden_dim=32,
        target_update_freq=50,
        run_evaluation=True,
    )


@pytest.mark.slow
def test_mrq_runs(config):
    """Test that MrQ training completes without errors."""
    rng = jax.random.PRNGKey(0)
    _state, metrics = jax.jit(train_mrq, static_argnums=(0,))(config, rng)
    assert "eval_returns" in metrics
    assert "eval_lengths" in metrics
    assert "encoder_loss" in metrics
    assert "value_loss" in metrics
    assert "policy_loss" in metrics
    assert "global_step" in metrics
    assert "train_return" in metrics


@pytest.mark.slow
def test_mrq_get_train_fn(config):
    """Test that MrQ is registered and accessible via get_train_fn."""
    train_fn = get_train_fn("mrq")
    assert train_fn is train_mrq

    rng = jax.random.PRNGKey(42)
    _state, metrics = jax.jit(train_fn, static_argnums=(0,))(config, rng)
    assert metrics["eval_returns"].shape[0] > 0


@pytest.mark.slow
def test_mrq_vmap(config):
    """Test that MrQ can be vmapped over seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmap_train = jax.jit(jax.vmap(train_mrq, in_axes=(None, 0)), static_argnums=(0,))
    _state, metrics = vmap_train(config, keys)
    assert metrics["eval_returns"].shape[0] == 2


def test_mrq_from_yaml():
    """Test that MrQ config loads from YAML."""
    config = MrQConfig.from_yaml("configs/mrq/gymnax/cartpole.yaml")
    assert config.env == "CartPole-v1"
    assert config.zs_dim == 256
    assert config.za_dim == 128
    assert config.zsa_dim == 256
