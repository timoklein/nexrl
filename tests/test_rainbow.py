import jax
import pytest

from nexrl import RainbowConfig, train_rainbow

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


ARGS = {
    "learning_rate": 0.0003,
    "total_timesteps": 16384,
    "eval_freq": 16384,
    "skip_initial_evaluation": True,
    "n_atoms": 11,
    "v_min": -5.0,
    "v_max": 5.0,
    "n_step": 1,
    "per_alpha": 0.5,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
}


def _train(env):
    config = RainbowConfig(**ARGS)
    return train_rainbow(config, jax.random.PRNGKey(0), env=env)


def _get_q_network(state):
    return state["q_network"]


def _make_act(state, config=None):
    from nexrl.algos.utils import make_eval_act

    cfg = config or RainbowConfig(**ARGS)
    model = state["q_network"]
    return make_eval_act(lambda obs, rng: model.act(obs, rng, epsilon=0.005), cfg, state.get("obs_rms_state"))


def test_env1():
    state, _ = _train(TestEnv1Discrete())
    q_network = _get_q_network(state)
    value = q_network.q_values(jax.numpy.array([0]))
    assert value == pytest.approx(1.0, abs=0.15)


def test_env2():
    state, _ = _train(TestEnv2Discrete())
    q_network = _get_q_network(state)

    obs = jax.numpy.array([[-1], [1]])
    rew = obs
    value = q_network.q_values(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.15)


def test_env3():
    state, _ = _train(TestEnv3Discrete())
    q_network = _get_q_network(state)

    obs = jax.numpy.array([[-1], [1]])
    gamma = RainbowConfig(**ARGS).gamma
    rew = [1 * gamma, 1]
    value = q_network.q_values(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.15)


def test_env4():
    state, _ = _train(TestEnv4Discrete())
    q_network = _get_q_network(state)

    best_action = 1
    value = q_network.q_values(jax.numpy.array([0]))
    assert value.argmax() == best_action

    act = _make_act(state)
    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

    for a in actions:
        assert a == pytest.approx(best_action, abs=0.1)


def test_env5():
    state, _ = _train(TestEnv5Discrete())

    rng = jax.random.PRNGKey(0)
    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

    act = _make_act(state)
    rngs = jax.random.split(rng, 10)
    actions = jax.vmap(act)(obs, rngs)

    for o, a in zip(obs, actions):
        assert (a > 0.5) == (o > 0)
