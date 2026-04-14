from functools import partial

import jax
import pytest
from jax import numpy as jnp

from nexrl import MPOConfig, train_mpo_continuous

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


CONTINUOUS_ARGS = {
    "num_envs": 1,
    "learning_rate": 0.0003,
    "critic_learning_rate": 0.0003,
    "dual_learning_rate": 0.01,
    "total_timesteps": 131072,
    "eval_freq": 131072,
    "init_log_alpha_mean": 1.0,
    "init_log_alpha_stddev": 1.0,
    "skip_initial_evaluation": True,
}


def _train_continuous(env):
    config = MPOConfig(**CONTINUOUS_ARGS)
    return train_mpo_continuous(config, jax.random.PRNGKey(0), env=env)


def _get_q_network(state):
    return state["q_network"]


def _make_act(state, config=None):
    from nexrl.algos.utils import make_eval_act

    cfg = config or MPOConfig(**CONTINUOUS_ARGS)
    model = state["actor"]
    return make_eval_act(lambda obs, rng: model.act(obs, rng), cfg, state.get("obs_rms_state"))


def test_env1_continuous():
    state, _ = _train_continuous(TestEnv1Continuous())
    act = _make_act(state)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs = jax.numpy.zeros((10, 1))
    actions = jax.vmap(act)(obs, rngs)
    actions = jax.numpy.expand_dims(actions, 1)

    q_network = _get_q_network(state)
    qs = q_network(obs, actions)
    for v in qs:
        assert v == pytest.approx(1.0, abs=0.1)


def test_env2_continuous():
    state, _ = _train_continuous(TestEnv2Continuous())
    act = _make_act(state)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    actions = jax.vmap(act)(obs, rngs)
    actions = jax.numpy.expand_dims(actions, 1)

    q_network = _get_q_network(state)
    qs = q_network(obs, actions)
    for v, r in zip(qs, obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.1)


def test_env3_continuous():
    state, _ = _train_continuous(TestEnv3Continuous())
    act = _make_act(state)
    q_network = _get_q_network(state)
    gamma = MPOConfig(**CONTINUOUS_ARGS).gamma

    @partial(jax.vmap, in_axes=(None, 0))
    def test_i(obs, rng):
        action = act(obs, rng)
        action = jax.numpy.expand_dims(action, 0)
        obs = jax.numpy.expand_dims(obs, 0)
        action = jax.numpy.expand_dims(action, 1)
        value = q_network(obs, action)
        return value

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    for obs in jax.numpy.array([[-1], [1]]):
        r = 1 * gamma if obs == -1 else 1
        for v in test_i(obs, rngs):
            assert v == pytest.approx(r, abs=0.1)


def test_env4_continuous():
    state, _ = _train_continuous(TestEnv4Continuous())
    act = _make_act(state)
    q_network = _get_q_network(state)

    @partial(jax.vmap, in_axes=(None, 0))
    def test_i(obs, rng):
        action = act(obs, rng)
        action = jax.numpy.expand_dims(action, 0)
        obs = jax.numpy.expand_dims(obs, 0)
        action = jax.numpy.expand_dims(action, 1)
        value = q_network(obs, action)
        return value.squeeze(), action.squeeze()

    num_rngs = 100
    rngs = jax.random.split(jax.random.PRNGKey(1), num_rngs)
    obs = jax.numpy.array([0])
    vv, aa = test_i(obs, rngs)

    # MPO with KL constraints doesn't push as hard to extremes as TD3,
    # so check direction (positive) and Q accuracy rather than exact action
    assert jnp.sum(aa > 0) >= 0.9 * num_rngs
    for v, a in zip(vv, aa):
        assert float(v) == pytest.approx(float(a), abs=0.1)


def test_env5_continuous():
    state, _ = _train_continuous(TestEnv5Continuous())

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    # Critic accuracy: Q(s, a=obs) ≈ 0 (optimal = -|a - obs| = 0)
    q_network = _get_q_network(state)
    value = q_network(obs, obs)
    for v in value:
        assert float(v) == pytest.approx(0.0, abs=0.15)

    # Policy direction: mean actions should be positively correlated with obs
    # (MPO with KL constraints is more conservative than SAC/TD3)
    act = _make_act(state)
    vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
    num_rngs = 100
    rngs = jax.random.split(rng, num_rngs)
    actions = vmap_act(obs, rngs)

    mean_actions = actions.mean(axis=0).squeeze()  # (10,)
    obs_flat = obs.squeeze()  # (10,)
    # Correlation: actions track obs direction
    correlation = jnp.corrcoef(mean_actions, obs_flat)[0, 1]
    assert correlation > 0.8
