"""Tests for MRS.Q algorithm."""

import jax
import pytest

from nexrl import MrsQConfig, get_train_fn, train_mrsq


@pytest.fixture
def config():
    return MrsQConfig(
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
        num_critics=3,
        sem_groups=4,
        num_samples=32,
        num_pi_trajs=4,
        num_elites=8,
        mpc_horizon=2,
        mpc_iterations=2,
        run_evaluation=True,
    )


@pytest.mark.slow
def test_mrsq_runs(config):
    """Test that MRS.Q training completes without errors."""
    rng = jax.random.PRNGKey(0)
    _state, metrics = jax.jit(train_mrsq, static_argnums=(0,))(config, rng)
    assert "eval_returns" in metrics
    assert "eval_lengths" in metrics
    assert "encoder_loss" in metrics
    assert "value_loss" in metrics
    assert "policy_loss" in metrics
    assert "global_step" in metrics
    assert "train_return" in metrics


@pytest.mark.slow
def test_mrsq_get_train_fn(config):
    """Test that MRS.Q is registered and accessible via get_train_fn."""
    train_fn = get_train_fn("mrsq")
    assert train_fn is train_mrsq

    rng = jax.random.PRNGKey(42)
    _state, metrics = jax.jit(train_fn, static_argnums=(0,))(config, rng)
    assert metrics["eval_returns"].shape[0] > 0


@pytest.mark.slow
def test_mrsq_vmap(config):
    """Test that MRS.Q can be vmapped over seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmap_train = jax.jit(jax.vmap(train_mrsq, in_axes=(None, 0)), static_argnums=(0,))
    _state, metrics = vmap_train(config, keys)
    assert metrics["eval_returns"].shape[0] == 2


def test_mrsq_from_yaml():
    """Test that MRS.Q config loads from YAML."""
    config = MrsQConfig.from_yaml("configs/mrsq/gymnax/cartpole.yaml")
    assert config.env == "CartPole-v1"
    assert config.zs_dim == 256
    assert config.num_critics == 10
    assert config.sem_groups == 8
