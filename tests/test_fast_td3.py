import jax
import pytest
from jax import numpy as jnp

from nexrl import FastTD3Config, train_fast_td3

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


# Reference algorithm defaults, scaled down to the analytic test envs
ARGS = {
    "env": "Pendulum-v1",  # overridden by the env= argument
    "obs_key": None,
    "policy_obs_dim": None,
    "num_envs": 4,
    "buffer_size_per_env": 2048,
    "batch_size": 128,
    "actor_hidden_layer_sizes": (64, 64),
    "critic_hidden_layer_sizes": (64, 64),
    "total_timesteps": 8192,
    "eval_freq": 8192,
    "skip_initial_evaluation": True,
}


def _train(env):
    config = FastTD3Config(**ARGS)
    return train_fast_td3(config, jax.random.PRNGKey(0), env=env)


def _q_fn(state):
    # The critic sees normalized obs (normalize_observations=True) and policy-space actions in
    # [-1, 1]; min over the ensemble, as in the FastTD3 policy loss
    from nexrl.algos.utils import normalize_obs

    def q_fn(obs, actions):
        return state["critic"].q_values(normalize_obs(state["obs_rms_state"], obs), actions).min(axis=0)

    return q_fn


def _make_act(state, env):
    # Same mapping as the eval callback of train_fast_td3: clip to [-1, 1], rescale to the env range
    from nexrl.algos.utils import make_eval_act

    model = state["actor"]
    space = env.action_space(env.default_params)
    low = jnp.asarray(space.low, dtype=jnp.float32)
    high = jnp.asarray(space.high, dtype=jnp.float32)

    def to_env_action(action):
        return low + 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0) * (high - low)

    return make_eval_act(lambda obs, rng: to_env_action(model.act(obs, rng)), FastTD3Config(**ARGS), state["obs_rms_state"])


def test_env1():
    env = TestEnv1Continuous()
    state, _ = _train(env)
    act = _make_act(state, env)

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    obs = jnp.zeros((10, 1))
    actions = jnp.expand_dims(jax.vmap(act)(obs, rngs), 1)

    for v in _q_fn(state)(obs, actions):
        assert v == pytest.approx(1.0, abs=0.1)


def test_env2():
    env = TestEnv2Continuous()
    state, _ = _train(env)
    act = _make_act(state, env)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    actions = jnp.expand_dims(jax.vmap(act)(obs, rngs), 1)

    for v, r in zip(_q_fn(state)(obs, actions), obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.1)


# The critic's nnx.vmap'd ensemble cannot run inside a jax.vmap that captures the module, so
# the tests below evaluate it on a batch (the eval action is deterministic, rng unused).


def test_env3():
    env = TestEnv3Continuous()
    state, _ = _train(env)
    act = _make_act(state, env)
    q_fn = _q_fn(state)
    gamma = FastTD3Config(**ARGS).gamma

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    for obs in jnp.array([[-1], [1]]):
        r = 1 * gamma if obs == -1 else 1
        obs_B1 = jnp.broadcast_to(obs, (10, 1))
        actions = jnp.expand_dims(jax.vmap(act)(obs_B1, rngs), 1)
        for v in q_fn(obs_B1, actions):
            assert v == pytest.approx(r, abs=0.1)


def test_env4():
    # Action range [-2, 2]: the reward is the env action = 2 * the policy-space action
    env = TestEnv4Continuous()
    state, _ = _train(env)
    act = _make_act(state, env)

    num_rngs = 100
    obs_B1 = jnp.zeros((num_rngs, 1))
    aa = jnp.expand_dims(jax.vmap(act)(obs_B1, jax.random.split(jax.random.PRNGKey(1), num_rngs)), 1)
    vv = _q_fn(state)(obs_B1, aa / 2.0)  # critic input is in policy space

    assert sum(aa.squeeze(-1) > 0.0) >= 0.9 * num_rngs
    for v, a in zip(vv, aa.squeeze(-1)):
        assert float(v) == pytest.approx(float(a), abs=0.1)


def test_env5():
    env = TestEnv5Continuous()
    state, _ = _train(env)

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    # Action range [-1, 1]: policy space equals env space
    for v in _q_fn(state)(obs, obs):
        assert float(v) == pytest.approx(0.0, abs=0.15)

    act = _make_act(state, env)
    vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
    num_rngs = 100
    actions = vmap_act(obs, jax.random.split(rng, num_rngs))

    for i in range(obs.shape[0]):
        assert float(actions[:, i].mean()) == pytest.approx(float(obs[i].squeeze()), abs=0.15)
        assert jnp.isclose(actions[:, i], obs[i], atol=0.5).sum() >= 0.9 * num_rngs
