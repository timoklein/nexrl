import jax
import pytest
from jax import numpy as jnp

from nexrl import VMPOConfig, train_vmpo_continuous

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


CONTINUOUS_ARGS = {
    "num_envs": 64,
    "num_steps": 16,
    "num_epochs": 10,
    "learning_rate": 0.0003,
    "critic_learning_rate": 0.0003,
    "dual_learning_rate": 0.01,
    "total_timesteps": 131072,
    "eval_freq": 131072,
    "init_log_alpha_mean": 1.0,
    "init_log_alpha_stddev": 1.0,
    "skip_initial_evaluation": True,
}


def _train(env):
    config = VMPOConfig(**CONTINUOUS_ARGS)
    return train_vmpo_continuous(config, jax.random.PRNGKey(0), env=env)


def _get_critic(state):
    return state["critic"]


def _make_act(state, config=None):
    from nexrl.algos.utils import make_eval_act

    cfg = config or VMPOConfig(**CONTINUOUS_ARGS)
    model = state["actor"]
    return make_eval_act(lambda obs, rng: model.act(obs, rng), cfg, state.get("obs_rms_state"))


def test_env1_continuous():
    state, _ = _train(TestEnv1Continuous())
    critic = _get_critic(state)
    value = critic(jax.numpy.array([0]))
    assert value == pytest.approx(1.0, abs=0.1)


def test_env2_continuous():
    state, _ = _train(TestEnv2Continuous())
    critic = _get_critic(state)

    obs = jax.numpy.array([[-1], [1]])
    value = critic(obs)

    for v, r in zip(value, obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.1)


def test_env3_continuous():
    state, _ = _train(TestEnv3Continuous())
    critic = _get_critic(state)
    gamma = VMPOConfig(**CONTINUOUS_ARGS).gamma

    obs = jax.numpy.array([[-1], [1]])
    rew = [1 * gamma, 1]
    value = critic(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.1)


def test_env4_continuous():
    state, _ = _train(TestEnv4Continuous())
    act = _make_act(state)
    critic = _get_critic(state)

    best_action = 2.0
    value = critic(jax.numpy.array([0]))
    assert value.item() == pytest.approx(best_action, abs=0.1)

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

    for a in actions:
        assert float(a) == pytest.approx(best_action, abs=0.1)


def test_env5_continuous():
    state, _ = _train(TestEnv5Continuous())

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    # Policy direction: mean actions should be positively correlated with obs
    # (V-MPO with KL constraints is more conservative than PPO/SAC)
    act = _make_act(state)
    vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
    num_rngs = 100
    rngs = jax.random.split(rng, num_rngs)
    actions = vmap_act(obs, rngs)

    mean_actions = actions.mean(axis=0).squeeze()  # (10,)
    obs_flat = obs.squeeze()  # (10,)
    correlation = jnp.corrcoef(mean_actions, obs_flat)[0, 1]
    assert correlation > 0.8
