"""Tests for PPO-E3B algorithm."""

import jax
import pytest

from nexrl import PPOE3BConfig, get_train_fn, train_ppo_e3b


@pytest.fixture
def config():
    return PPOE3BConfig(
        env="CartPole-v1",
        num_envs=8,
        num_steps=16,
        num_epochs=2,
        num_minibatches=2,
        learning_rate=7.5e-4,
        total_timesteps=2000,
        eval_freq=500,
        icm_latent_size=16,
        icm_hidden_layer_sizes=(32, 32),
        e3b_lambda=0.1,
        run_evaluation=True,
    )


@pytest.mark.slow
def test_ppo_e3b_runs(config):
    """Test that PPO-E3B training completes without errors."""
    rng = jax.random.PRNGKey(0)
    _state, metrics = jax.jit(train_ppo_e3b, static_argnums=(0,))(config, rng)
    assert "eval_returns" in metrics
    assert "eval_lengths" in metrics
    assert "actor_loss" in metrics
    assert "ext_critic_loss" in metrics
    assert "int_critic_loss" in metrics
    assert "icm_forward_loss" in metrics
    assert "icm_inverse_loss" in metrics
    assert "entropy" in metrics
    assert "mean_intrinsic_reward" in metrics
    assert "global_step" in metrics
    assert "train_return" in metrics


@pytest.mark.slow
def test_ppo_e3b_get_train_fn(config):
    """Test that PPO-E3B is registered and accessible via get_train_fn."""
    train_fn = get_train_fn("ppo_e3b")
    assert train_fn is train_ppo_e3b

    rng = jax.random.PRNGKey(42)
    _state, metrics = jax.jit(train_fn, static_argnums=(0,))(config, rng)
    assert metrics["eval_returns"].shape[0] > 0


@pytest.mark.slow
def test_ppo_e3b_vmap(config):
    """Test that PPO-E3B can be vmapped over seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmap_train = jax.jit(jax.vmap(train_ppo_e3b, in_axes=(None, 0)), static_argnums=(0,))
    _state, metrics = vmap_train(config, keys)
    assert metrics["eval_returns"].shape[0] == 2


def test_ppo_e3b_from_yaml():
    """Test that PPO-E3B config loads from YAML."""
    config = PPOE3BConfig.from_yaml("configs/ppo_e3b/gymnax/cartpole.yaml")
    assert config.env == "CartPole-v1"
    assert config.e3b_lambda == 0.1
    assert config.icm_latent_size == 64
