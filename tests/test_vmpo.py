import jax
import pytest

from nexrl import VMPOConfig, train_vmpo

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


DISCRETE_ARGS = {
    "num_envs": 64,
    "num_steps": 16,
    "num_epochs": 10,
    "learning_rate": 0.0003,
    "critic_learning_rate": 0.0003,
    "dual_learning_rate": 0.01,
    "total_timesteps": 131072,
    "eval_freq": 131072,
    "skip_initial_evaluation": True,
}


def _train(env):
    config = VMPOConfig(**DISCRETE_ARGS)
    return train_vmpo(config, jax.random.PRNGKey(0), env=env)


def _get_critic(state):
    return state["critic"]


def _make_act(state, config=None):
    from nexrl.algos.utils import make_eval_act

    cfg = config or VMPOConfig(**DISCRETE_ARGS)
    model = state["actor"]
    return make_eval_act(lambda obs, rng: model.act(obs, rng), cfg, state.get("obs_rms_state"))


def test_env1():
    state, _ = _train(TestEnv1Discrete())
    critic = _get_critic(state)
    value = critic(jax.numpy.array([0]))
    assert value == pytest.approx(1.0, abs=0.1)


def test_env2():
    state, _ = _train(TestEnv2Discrete())
    critic = _get_critic(state)

    obs = jax.numpy.array([[-1], [1]])
    value = critic(obs)

    for v, r in zip(value, obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.1)


def test_env3():
    state, _ = _train(TestEnv3Discrete())
    critic = _get_critic(state)

    obs = jax.numpy.array([[-1], [1]])
    gamma = VMPOConfig(**DISCRETE_ARGS).gamma
    rew = [1 * gamma, 1]
    value = critic(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.1)


def test_env4():
    state, _ = _train(TestEnv4Discrete())
    critic = _get_critic(state)

    best_action = 1.0
    value = critic(jax.numpy.array([0]))
    assert value.item() == pytest.approx(best_action, abs=0.1)

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

    for o, a in zip(obs, actions):
        assert (a > 0.5) == (o > 0)
