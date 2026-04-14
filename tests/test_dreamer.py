"""Tests for Dreamer-v3 algorithm."""

import jax
import jax.numpy as jnp
import pytest

from nexrl import DreamerConfig, NetworkConfig, RSSMConfig, get_train_fn, train_dreamer
from nexrl.networks.dreamer import symexp, symlog, twohot_encode


@pytest.fixture
def cartpole_config():
    return DreamerConfig(
        env="CartPole-v1",
        agent_kwargs=NetworkConfig(activation="silu", hidden_layer_sizes=(32, 32)),
        rssm_kwargs=RSSMConfig(deter_dim=32, stoch_categories=4, stoch_classes=4, hidden_dim=32),
        num_envs=4,
        buffer_size=500,
        fill_buffer=50,
        batch_size=4,
        sequence_length=8,
        imagination_horizon=5,
        num_epochs=1,
        learning_rate=3e-4,
        max_grad_norm=100.0,
        total_timesteps=2000,
        eval_freq=500,
        gamma=0.99,
        value_num_bins=31,
        run_evaluation=True,
    )


def test_dreamer_runs(cartpole_config):
    """Test that Dreamer training completes without errors."""
    rng = jax.random.PRNGKey(0)
    _state, metrics = jax.jit(train_dreamer, static_argnums=(0,))(cartpole_config, rng)
    assert "eval_returns" in metrics
    assert "eval_lengths" in metrics
    assert "world_model_loss" in metrics
    assert "recon_loss" in metrics
    assert "reward_loss" in metrics
    assert "continue_loss" in metrics
    assert "dyn_kl" in metrics
    assert "rep_kl" in metrics
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "entropy" in metrics
    assert "global_step" in metrics
    assert "train_return" in metrics


def test_dreamer_get_train_fn(cartpole_config):
    """Test that Dreamer is registered and accessible via get_train_fn."""
    train_fn = get_train_fn("dreamer")
    assert train_fn is train_dreamer

    rng = jax.random.PRNGKey(42)
    _state, metrics = jax.jit(train_fn, static_argnums=(0,))(cartpole_config, rng)
    assert metrics["eval_returns"].shape[0] > 0


def test_dreamer_vmap(cartpole_config):
    """Test that Dreamer can be vmapped over seeds."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    vmap_train = jax.jit(jax.vmap(train_dreamer, in_axes=(None, 0)), static_argnums=(0,))
    _state, metrics = vmap_train(cartpole_config, keys)
    assert metrics["eval_returns"].shape[0] == 2


def test_dreamer_from_yaml():
    """Test that Dreamer config loads from YAML."""
    config = DreamerConfig.from_yaml("configs/dreamer/gymnax/cartpole.yaml")
    assert config.env == "CartPole-v1"
    assert config.rssm_kwargs.deter_dim == 256
    assert config.rssm_kwargs.stoch_categories == 16
    assert config.imagination_horizon == 15
    assert config.value_num_bins == 255


def test_symlog_roundtrip():
    """Test symlog/symexp roundtrip: symexp(symlog(x)) == x."""
    x = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    recovered = symexp(symlog(x))
    assert jnp.allclose(recovered, x, atol=1e-5)


def test_twohot_roundtrip():
    """Test two-hot encode then decode approximately recovers original values."""
    values = jnp.array([0.0, 1.0, -1.0, 5.0, -5.0, 50.0])
    num_bins = 255
    encoded = twohot_encode(values, num_bins)
    assert encoded.shape == (6, num_bins)
    # Check each is a valid distribution (sums to ~1)
    assert jnp.allclose(encoded.sum(axis=-1), 1.0, atol=1e-5)
    # Decode via logits (use log of encoded as logits approximation)
    # For a proper test, decode from the encoded distribution directly
    support = jnp.linspace(-20.0, 20.0, num_bins)
    expected_symlog = jnp.sum(encoded * support, axis=-1)
    recovered = symexp(expected_symlog)
    assert jnp.allclose(recovered, values, atol=0.5)
