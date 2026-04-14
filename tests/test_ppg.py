import jax
import pytest

from nexrl import PPGConfig, train_ppg

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


ARGS = {
    "num_envs": 64,
    "num_steps": 16,
    "num_epochs": 10,
    "learning_rate": 0.0003,
    "total_timesteps": 131072,
    "eval_freq": 131072,
    "skip_initial_evaluation": True,
    # PPG-specific: small values for fast tests
    "n_policy_iters": 4,
    "num_aux_epochs": 2,
    "aux_num_minibatches": 4,
}


def _train(env):
    config = PPGConfig(**ARGS)
    return train_ppg(config, jax.random.PRNGKey(0), env=env)


def _get_critic(state):
    ac = state["system"].ac
    return lambda obs: ac.value(obs)


def _make_act(state, config=None):
    from nexrl.algos.utils import make_eval_act

    cfg = config or PPGConfig(**ARGS)
    ac = state["system"].ac
    return make_eval_act(lambda obs, rng: ac.act(obs, rng), cfg, state.get("obs_rms_state"))


def test_env1():
    state, _ = _train(TestEnv1Discrete())
    critic = _get_critic(state)
    value = critic(jax.numpy.array([0]))
    assert value == pytest.approx(1.0, abs=0.2)


def test_env2():
    state, _ = _train(TestEnv2Discrete())
    critic = _get_critic(state)

    obs = jax.numpy.array([[-1], [1]])
    value = critic(obs)

    for v, r in zip(value, obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.2)


def test_env3():
    state, _ = _train(TestEnv3Discrete())
    critic = _get_critic(state)

    obs = jax.numpy.array([[-1], [1]])
    gamma = PPGConfig(**ARGS).gamma
    rew = [1 * gamma, 1]
    value = critic(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.2)


def test_env4():
    state, _ = _train(TestEnv4Discrete())
    critic = _get_critic(state)

    best_action = 1.0
    value = critic(jax.numpy.array([0]))
    assert value.item() == pytest.approx(best_action, abs=0.2)

    act = _make_act(state)
    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

    for a in actions:
        assert float(a) == pytest.approx(best_action, abs=0.1)


def test_env5():
    state, _ = _train(TestEnv5Discrete())

    rng = jax.random.PRNGKey(0)
    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

    act = _make_act(state)
    rngs = jax.random.split(rng, 10)
    actions = jax.vmap(act)(obs, rngs)

    for o, a in zip(obs.squeeze(-1), actions):
        assert (a > 0.5) == (o > 0)
