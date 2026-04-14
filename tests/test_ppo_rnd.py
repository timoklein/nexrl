"""Tests for PPO-RND algorithm."""

import jax
import pytest

from nexrl import PPORNDConfig, get_train_fn, train_ppo_rnd


@pytest.fixture
def config():
    return PPORNDConfig(
        env="CartPole-v1",
        num_envs=8,
        num_steps=16,
        num_epochs=2,
        num_minibatches=2,
        learning_rate=7.5e-4,
        total_timesteps=2000,
        eval_freq=500,
        rnd_embedding_dim=16,
        rnd_hidden_layer_sizes=(32, 32),
        run_evaluation=True,
    )


@pytest.mark.slow
def test_ppo_rnd_runs(config):
    """Test that PPO-RND training completes without errors."""
    rng = jax.random.PRNGKey(0)
    _state, metrics = jax.jit(train_ppo_rnd, static_argnums=(0,))(config, rng)
    assert "eval_returns" in metrics
    assert "eval_lengths" in metrics
    assert "actor_loss" in metrics
    assert "ext_critic_loss" in metrics
    assert "int_critic_loss" in metrics
    assert "rnd_loss" in metrics
    assert "entropy" in metrics
    assert "global_step" in metrics
    assert "train_return" in metrics


@pytest.mark.slow
def test_ppo_rnd_get_train_fn(config):
    """Test that PPO-RND is registered and accessible via get_train_fn."""
    train_fn = get_train_fn("ppo_rnd")
    assert train_fn is train_ppo_rnd

    rng = jax.random.PRNGKey(42)
    _state, metrics = jax.jit(train_fn, static_argnums=(0,))(config, rng)
    assert metrics["eval_returns"].shape[0] > 0


@pytest.mark.slow
def test_ppo_rnd_vmap(config):
    """Test that PPO-RND can be vmapped over seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmap_train = jax.jit(jax.vmap(train_ppo_rnd, in_axes=(None, 0)), static_argnums=(0,))
    _state, metrics = vmap_train(config, keys)
    assert metrics["eval_returns"].shape[0] == 2


def test_ppo_rnd_from_yaml():
    """Test that PPO-RND config loads from YAML."""
    config = PPORNDConfig.from_yaml("configs/ppo_rnd/gymnax/cartpole.yaml")
    assert config.env == "CartPole-v1"
    assert config.rnd_embedding_dim == 64
