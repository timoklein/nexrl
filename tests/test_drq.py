import jax
import jax.numpy as jnp
import pytest

from nexrl import DrQConfig, train_drq

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


# Debug-env overrides: vector obs, so augmentation off (image-only); n_step=1 makes the
# ring buffer a plain 1-step transition; adam_eps pinned to the optax default because
# the Dopamine 1.5e-4 dampens updates too much for these short runs; epsilon_eval
# pinned low so greedy-action assertions are deterministic.
ARGS = {
    "learning_rate": 0.0003,
    "adam_eps": 1e-8,
    "total_timesteps": 16384,
    "eval_freq": 16384,
    "n_step": 1,
    "data_augmentation": False,
    "clip_rewards": False,
    "epsilon_train": 0.01,
    "epsilon_decay_period": 2000,
    "epsilon_eval": 0.001,
    "skip_initial_evaluation": True,
}


def _train(env):
    config = DrQConfig(**ARGS)
    return train_drq(config, jax.random.PRNGKey(0), env=env)


def _get_q_network(state):
    return state["q_network"]


def _make_act(state, config=None):
    from nexrl.algos.utils import make_eval_act

    cfg = config or DrQConfig(**ARGS)
    model = state["q_network"]
    return make_eval_act(lambda obs, rng: model.act(obs, rng, epsilon=cfg.epsilon_eval), cfg, state.get("obs_rms_state"))


def test_env1():
    state, _ = _train(TestEnv1Discrete())
    q_network = _get_q_network(state)
    value = q_network(jax.numpy.array([0]))
    assert value == pytest.approx(1.0, abs=0.1)


def test_env2():
    state, _ = _train(TestEnv2Discrete())
    q_network = _get_q_network(state)

    obs = jax.numpy.array([[-1], [1]])
    rew = obs
    value = q_network(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.1)


def test_env3():
    state, _ = _train(TestEnv3Discrete())
    q_network = _get_q_network(state)

    obs = jax.numpy.array([[-1], [1]])
    gamma = DrQConfig(**ARGS).gamma
    rew = [1 * gamma, 1]
    value = q_network(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.1)


def test_env4():
    state, _ = _train(TestEnv4Discrete())
    q_network = _get_q_network(state)

    best_action = 1
    value = q_network(jax.numpy.array([0]))
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


# ---------------------------------------------------------------------------
# Smoke test: end-to-end Atari with augmentation, n-step, and life-loss handling
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_drq_smoke_atari():
    """Run a very short DrQ training on Atari to verify end-to-end correctness.

    Exercises the full Atari path (uint8 obs, augmentation, n-step ring,
    terminal_on_life_loss, reward clipping). Checks: all metric keys present,
    all values finite.
    """
    pytest.importorskip("ale_py")
    from nexrl.configs import CNNConfig

    config = DrQConfig(
        env="atari/Breakout-v5",
        encoder_kwargs=CNNConfig(encoder="nature", features_dim=64),
        num_envs=1,
        buffer_size=500,
        fill_buffer=100,
        batch_size=16,
        n_step=3,
        total_timesteps=300,
        eval_freq=150,
        terminal_on_life_loss=True,
        run_evaluation=False,
        skip_initial_evaluation=True,
    )

    rng = jax.random.PRNGKey(0)
    _state, metrics = train_drq(config, rng)

    expected_keys = {
        "eval_lengths",
        "eval_returns",
        "train_return",
        "global_step",
        "td_loss",
        "mean_q_value",
        "epsilon",
    }
    assert expected_keys <= set(metrics.keys())
    for key, value in metrics.items():
        assert jnp.all(jnp.isfinite(value)), f"non-finite values in metric {key}"
    assert int(metrics["global_step"][-1]) > 0
