import dataclasses

import jax
import numpy as np
import pytest
from flax import nnx
from jax import numpy as jnp

from nexrl import FastTD3Config, train_fast_td3
from nexrl.algos.fast_td3 import NStepTransition, make_fast_td3_networks, nstep_transitions
from nexrl.buffers import CircularBuffer

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


def _check_env3(state, env):
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


def test_env3():
    env = TestEnv3Continuous()
    state, _ = _train(env)
    _check_env3(state, env)


def test_env3_nstep():
    # 3-step windows stop at the terminal second step, so the values are those of one-step TD
    env = TestEnv3Continuous()
    state, _ = train_fast_td3(FastTD3Config(**{**ARGS, "n_step": 3}), jax.random.PRNGKey(0), env=env)
    _check_env3(state, env)


def test_nstep_transitions_windows():
    # Ring of 6 rows, 1 env, n_step 3, gamma 0.5. Row t has reward 2^t, next obs 100 + t.
    size = 6

    def buffer(*, index, full, terminated=(), truncated=()):
        rows_T1 = jnp.arange(size, dtype=jnp.float32)[:, None]

        def flags(rows):
            return jnp.zeros((size, 1), dtype=jnp.float32).at[jnp.array(rows, dtype=jnp.int32), 0].set(1.0)

        data = NStepTransition(
            obs=rows_T1[..., None],
            action=rows_T1[..., None],
            reward=2.0**rows_T1,
            terminated=flags(terminated),
            truncated=flags(truncated),
            next_obs=100.0 + rows_T1[..., None],
        )
        return CircularBuffer(size=size, data=data, index=index, full=full)

    def window(buf, start):
        batch = nstep_transitions(buf, jnp.array([start]), jnp.array([0]), n_step=3, gamma=0.5)
        return float(batch.reward[0]), float(batch.discount[0]), float(batch.terminated[0]), float(batch.next_obs[0, 0])

    # (n-step reward, discount, terminated flag of the last row, next obs of the last row)
    # Rows 0-4 written: a whole window, rows 0-2
    assert window(buffer(index=5, full=False), 0) == (1 + 0.5 * 2 + 0.25 * 4, 0.125, 0.0, 102.0)
    # Termination at row 1 ends the window; the critic drops the bootstrap
    assert window(buffer(index=5, full=False, terminated=[1]), 0) == (1 + 0.5 * 2, 0.25, 1.0, 101.0)
    # A time limit at row 1 ends the window and bootstraps from row 1's final obs
    assert window(buffer(index=5, full=False, truncated=[1]), 0) == (1 + 0.5 * 2, 0.25, 0.0, 101.0)
    # Termination in the first row
    assert window(buffer(index=5, full=False, terminated=[2]), 2) == (4.0, 0.5, 1.0, 102.0)
    # Full ring with newest row 1: the window stops there, before the oldest row 2
    assert window(buffer(index=2, full=True), 0) == (1 + 0.5 * 2, 0.25, 0.0, 101.0)
    # A window that starts at the newest row is a one-step transition
    assert window(buffer(index=2, full=True), 1) == (2.0, 0.5, 0.0, 101.0)
    # A window wraps around the end of the ring: rows 4, 5, 0
    assert window(buffer(index=2, full=True), 4) == (16 + 0.5 * 32 + 0.25 * 1, 0.125, 0.0, 100.0)


def test_invalid_configs_raise():
    env = TestEnv1Continuous()
    with pytest.raises(ValueError, match="n_step"):
        train_fast_td3(FastTD3Config(**{**ARGS, "n_step": 0}), jax.random.PRNGKey(0), env=env)
    with pytest.raises(ValueError, match="buffer_size_per_env"):
        train_fast_td3(FastTD3Config(**{**ARGS, "n_step": 3, "buffer_size_per_env": 2}), jax.random.PRNGKey(0), env=env)
    with pytest.raises(ValueError, match="learning_starts"):
        train_fast_td3(FastTD3Config(**{**ARGS, "n_step": 3, "learning_starts": 2}), jax.random.PRNGKey(0), env=env)
    with pytest.raises(ValueError, match="run_evaluation"):
        train_fast_td3(FastTD3Config(**ARGS), jax.random.PRNGKey(0), env=env, eval_fn=lambda **kwargs: {})


def test_make_networks_hook():
    calls = []

    def make_networks(**kwargs):
        calls.append(kwargs)
        config = dataclasses.replace(kwargs["config"], actor_hidden_layer_sizes=(8,))
        return make_fast_td3_networks(**{**kwargs, "config": config})

    config = FastTD3Config(**{**ARGS, "total_timesteps": 1024, "eval_freq": 1024})
    state, metrics = train_fast_td3(config, jax.random.PRNGKey(0), env=TestEnv1Continuous(), make_networks=make_networks)

    assert sorted(calls[0]) == ["action_dim", "config", "in_features", "num_critics", "policy_in_features", "rngs"]
    assert (calls[0]["policy_in_features"], calls[0]["in_features"], calls[0]["action_dim"], calls[0]["num_critics"]) == (
        1,
        1,
        1,
        2,
    )
    kernel_shapes = sorted(x.shape for x in jax.tree.leaves(nnx.state(state["actor"], nnx.Param)) if x.ndim == 2)
    assert kernel_shapes == [(1, 8), (8, 1)]
    assert np.isfinite(metrics["critic_loss"]).all() and (metrics["critic_loss"] != 0).all()


def test_eval_fn_hook_keeps_training_keys():
    def eval_fn(*, act_fn, actor, critic, obs_rms, rng):
        actions = jax.vmap(act_fn)(jnp.zeros((3, 1)), jax.random.split(rng, 3))
        return {"hook/action": actions.mean(), "hook/key": jax.random.uniform(rng)}

    config = FastTD3Config(
        **{**ARGS, "total_timesteps": 2048, "eval_freq": 1024, "run_evaluation": True, "skip_initial_evaluation": False}
    )
    env = TestEnv1Continuous()
    _, plain = train_fast_td3(config, jax.random.PRNGKey(0), env=env)
    _, hooked = train_fast_td3(config, jax.random.PRNGKey(0), env=env, eval_fn=eval_fn)

    assert set(hooked) == set(plain) | {"hook/action", "hook/key"}
    # The initial evaluation and 2 evaluation points, each with a fresh key
    assert len(set(np.asarray(hooked["hook/key"]).tolist())) == 3
    # The hook draws no training keys: every built-in metric matches the run without it
    for k in plain:
        np.testing.assert_allclose(hooked[k], plain[k], rtol=1e-5, atol=1e-6)


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
