"""Tests for AlphaZero algorithm."""

import jax
import pytest

from nexrl import AlphaZeroConfig, get_train_fn, train_alphazero


@pytest.fixture
def config():
    return AlphaZeroConfig(
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
        num_simulations=4,
        run_evaluation=True,
    )


@pytest.mark.slow
def test_alphazero_runs(config):
    """Test that AlphaZero training completes without errors."""
    rng = jax.random.PRNGKey(0)
    _state, metrics = jax.jit(train_alphazero, static_argnums=(0,))(config, rng)
    assert "eval_returns" in metrics
    assert "eval_lengths" in metrics
    assert "actor_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "global_step" in metrics
    assert "train_return" in metrics


@pytest.mark.slow
def test_alphazero_get_train_fn(config):
    """Test that AlphaZero is registered and accessible via get_train_fn."""
    train_fn = get_train_fn("alphazero")
    assert train_fn is train_alphazero

    rng = jax.random.PRNGKey(42)
    _state, metrics = jax.jit(train_fn, static_argnums=(0,))(config, rng)
    assert metrics["eval_returns"].shape[0] > 0


@pytest.mark.slow
def test_alphazero_vmap(config):
    """Test that AlphaZero can be vmapped over seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmap_train = jax.jit(jax.vmap(train_alphazero, in_axes=(None, 0)), static_argnums=(0,))
    _state, metrics = vmap_train(config, keys)
    assert metrics["eval_returns"].shape[0] == 2


def test_alphazero_from_yaml():
    """Test that AlphaZero config loads from YAML."""
    config = AlphaZeroConfig.from_yaml("configs/alphazero/gymnax/cartpole.yaml")
    assert config.env == "CartPole-v1"
    assert config.num_simulations == 32
    assert config.mcts_policy == "muzero"
