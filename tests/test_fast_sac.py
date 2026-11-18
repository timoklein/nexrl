import dataclasses

import jax
import numpy as np
import pytest
from flax import nnx
from jax import numpy as jnp

from nexrl import FastSACConfig, train_fast_sac
from nexrl.algos.fast_sac import make_fast_sac_networks

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
    config = FastSACConfig(**ARGS)
    return train_fast_sac(config, jax.random.PRNGKey(0), env=env)


def _q_fn(state):
    # The critic sees normalized obs (normalize_observations=True); mean over the ensemble,
    # as in the FastSAC policy loss
    from nexrl.algos.utils import normalize_obs

    def q_fn(obs, actions):
        return state["critic"].q_values(normalize_obs(state["obs_rms_state"], obs), actions).mean(axis=0)

    return q_fn


def _make_act(state):
    from nexrl.algos.utils import make_eval_act

    model = state["actor"]
    return make_eval_act(lambda obs, rng: model.act(obs, rng), FastSACConfig(**ARGS), state["obs_rms_state"])


def test_env1():
    state, _ = _train(TestEnv1Continuous())
    act = _make_act(state)

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    obs = jnp.zeros((10, 1))
    actions = jnp.expand_dims(jax.vmap(act)(obs, rngs), 1)

    for v in _q_fn(state)(obs, actions):
        assert v == pytest.approx(1.0, abs=0.1)


def test_env2():
    state, _ = _train(TestEnv2Continuous())
    act = _make_act(state)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    actions = jnp.expand_dims(jax.vmap(act)(obs, rngs), 1)

    for v, r in zip(_q_fn(state)(obs, actions), obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.1)


# The critic's nnx.vmap'd ensemble cannot run inside a jax.vmap that captures the module, so
# the tests below evaluate it on a batch (the eval action is deterministic, rng unused).


def test_env3():
    state, _ = _train(TestEnv3Continuous())
    act = _make_act(state)
    q_fn = _q_fn(state)
    gamma = FastSACConfig(**ARGS).gamma

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    for obs in jnp.array([[-1], [1]]):
        r = 1 * gamma if obs == -1 else 1
        obs_B1 = jnp.broadcast_to(obs, (10, 1))
        actions = jnp.expand_dims(jax.vmap(act)(obs_B1, rngs), 1)
        for v in q_fn(obs_B1, actions):
            assert v == pytest.approx(r, abs=0.1)


def test_env4():
    state, _ = _train(TestEnv4Continuous())
    act = _make_act(state)

    num_rngs = 100
    obs_B1 = jnp.zeros((num_rngs, 1))
    aa = jnp.expand_dims(jax.vmap(act)(obs_B1, jax.random.split(jax.random.PRNGKey(1), num_rngs)), 1)
    vv = _q_fn(state)(obs_B1, aa)

    assert sum(aa.squeeze(-1) > 0.0) >= 0.9 * num_rngs
    for v, a in zip(vv, aa.squeeze(-1)):
        assert float(v) == pytest.approx(float(a), abs=0.1)


def test_env5():
    state, _ = _train(TestEnv5Continuous())

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    for v in _q_fn(state)(obs, obs):
        assert float(v) == pytest.approx(0.0, abs=0.15)

    act = _make_act(state)
    vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
    num_rngs = 100
    actions = vmap_act(obs, jax.random.split(rng, num_rngs))

    for i in range(obs.shape[0]):
        assert float(actions[:, i].mean()) == pytest.approx(float(obs[i].squeeze()), abs=0.15)
        assert jnp.isclose(actions[:, i], obs[i], atol=0.5).sum() >= 0.9 * num_rngs


def test_make_networks_hook():
    calls = []

    def make_networks(**kwargs):
        calls.append(kwargs)
        config = dataclasses.replace(kwargs["config"], actor_hidden_layer_sizes=(8,))
        return make_fast_sac_networks(**{**kwargs, "config": config})

    config = FastSACConfig(**{**ARGS, "total_timesteps": 1024, "eval_freq": 1024})
    state, metrics = train_fast_sac(config, jax.random.PRNGKey(0), env=TestEnv1Continuous(), make_networks=make_networks)

    assert sorted(calls[0]) == [
        "action_dim",
        "action_range",
        "config",
        "in_features",
        "num_critics",
        "policy_in_features",
        "rngs",
    ]
    assert calls[0]["action_range"] == (-1.0, 1.0) and calls[0]["num_critics"] == 2
    kernel_shapes = sorted(x.shape for x in jax.tree.leaves(nnx.state(state["actor"], nnx.Param)) if x.ndim == 2)
    assert kernel_shapes == [(1, 8), (8, 1), (8, 1)]  # hidden layer, mean head, log-std head
    assert np.isfinite(metrics["critic_loss"]).all() and (metrics["critic_loss"] != 0).all()


def test_eval_fn_hook_keeps_training_keys():
    def eval_fn(*, act_fn, actor, critic, obs_rms, rng):
        actions = jax.vmap(act_fn)(jnp.zeros((3, 1)), jax.random.split(rng, 3))
        return {"hook/action": actions.mean(), "hook/key": jax.random.uniform(rng)}

    config = FastSACConfig(
        **{**ARGS, "total_timesteps": 2048, "eval_freq": 1024, "run_evaluation": True, "skip_initial_evaluation": False}
    )
    env = TestEnv1Continuous()
    _, plain = train_fast_sac(config, jax.random.PRNGKey(0), env=env)
    _, hooked = train_fast_sac(config, jax.random.PRNGKey(0), env=env, eval_fn=eval_fn)

    assert set(hooked) == set(plain) | {"hook/action", "hook/key"}
    # The initial evaluation and 2 evaluation points, each with a fresh key
    assert len(set(np.asarray(hooked["hook/key"]).tolist())) == 3
    # The hook draws no training keys: every built-in metric matches the run without it
    for k in plain:
        np.testing.assert_allclose(hooked[k], plain[k], rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="run_evaluation"):
        train_fast_sac(FastSACConfig(**ARGS), jax.random.PRNGKey(0), env=env, eval_fn=eval_fn)
