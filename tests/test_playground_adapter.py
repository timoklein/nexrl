import jax
import jax.numpy as jnp
import numpy as np
import pytest


mujoco_playground = pytest.importorskip("mujoco_playground")

from mujoco_playground import registry  # noqa: E402

from nexrl.compat.playground2gymnax import (  # noqa: E402
    Playground2GymnaxEnv,
    Playground2GymnaxEnvCachedReset,
    create_playground,
)


EPISODE_LENGTH = 5


def test_truncation_info_and_final_obs():
    env = Playground2GymnaxEnv(
        registry.load("CartpoleBalance", config_overrides={"impl": "jax"}), episode_length=EPISODE_LENGTH
    )
    params = env.default_params
    step = jax.jit(env.step)
    step_env = jax.jit(env.step_env)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    action = jnp.zeros(env.num_actions)
    for t in range(1, EPISODE_LENGTH + 1):
        key, k = jax.random.split(key)
        # Raw (non-resetting) step from the same state, for comparison with the auto-reset step.
        raw_obs, _, _, _, _ = step_env(k, state, action, params)
        obs, new_state, _, done, info = step(k, state, action, params)
        assert info["truncated"].dtype == jnp.bool_
        assert info["final_obs"].shape == obs.shape and info["final_obs"].dtype == obs.dtype
        if t < EPISODE_LENGTH:
            assert not bool(done) and not bool(info["truncated"])
            np.testing.assert_array_equal(info["final_obs"], obs)
        else:
            assert bool(done) and bool(info["truncated"])
            np.testing.assert_array_equal(info["final_obs"], raw_obs)
            assert not np.array_equal(info["final_obs"], obs)  # obs is the post-reset observation
        state = new_state


def test_action_repeat():
    inner_env = registry.load("CartpoleBalance", config_overrides={"impl": "jax"})
    env = Playground2GymnaxEnv(inner_env, episode_length=EPISODE_LENGTH, action_repeat=2)
    params = env.default_params
    # 5 inner steps -> ceil(5 / 2) = 3 agent steps
    assert params.max_steps_in_episode == 3
    step = jax.jit(env.step)
    inner_step = jax.jit(inner_env.step)

    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)
    action = 0.5 * jnp.ones(env.num_actions)
    for t in range(1, 4):
        # Two raw inner steps from the same state
        ref_first = inner_step(state.inner, action)
        ref_second = inner_step(ref_first, action)
        key, k = jax.random.split(key)
        _, new_state, reward, done, info = step(k, state, action, params)
        np.testing.assert_allclose(reward, ref_first.reward + ref_second.reward, rtol=1e-6)
        np.testing.assert_array_equal(info["final_obs"], ref_second.obs)
        assert bool(done) == (t == 3) and bool(info["truncated"]) == (t == 3)
        state = new_state


def test_cached_reset_restarts_from_first_state():
    inner_env = registry.load("CartpoleBalance", config_overrides={"impl": "jax"})
    env = Playground2GymnaxEnvCachedReset(inner_env, episode_length=EPISODE_LENGTH)
    params = env.default_params
    step = jax.jit(env.step)
    inner_step = jax.jit(inner_env.step)

    key = jax.random.PRNGKey(0)
    first_obs, first = env.reset(key, params)
    state = first
    action = 0.5 * jnp.ones(env.num_actions)
    for _ in range(EPISODE_LENGTH):
        key, k = jax.random.split(key)
        obs, state, _, done, info = step(k, state, action, params)
    # Truncated at the last step: back at the first state; info keeps the pre-reset obs
    assert bool(done) and bool(info["truncated"])
    np.testing.assert_array_equal(obs, first_obs)
    np.testing.assert_array_equal(state.inner.data.qpos, first.inner.data.qpos)
    np.testing.assert_array_equal(state.inner.data.qvel, first.inner.data.qvel)
    assert int(state.steps) == 0 and not bool(state.inner.done)
    assert not np.array_equal(info["final_obs"], first_obs)
    # The next episode continues from the first state
    key, k = jax.random.split(key)
    obs, state, reward, done, _ = step(k, state, action, params)
    ref = inner_step(first.inner, action)
    np.testing.assert_array_equal(obs, ref.obs)
    np.testing.assert_array_equal(reward, ref.reward)
    assert not bool(done) and int(state.steps) == 1


def test_cached_reset_is_opt_in():
    assert type(create_playground("CartpoleBalance")[0]) is Playground2GymnaxEnv
    assert type(create_playground("CartpoleBalance", cached_reset=True)[0]) is Playground2GymnaxEnvCachedReset


def test_cached_reset_rejected_outside_playground():
    from nexrl import FastTD3Config
    from nexrl.algos.utils import create_env

    with pytest.raises(ValueError, match="cached_reset"):
        create_env(FastTD3Config(env="Pendulum-v1", obs_key=None, policy_obs_dim=None, cached_reset=True))
    # An env object (here through env=) never reaches the adapter, so cached_reset would be ignored.
    env = create_playground("CartpoleBalance")[0]
    with pytest.raises(ValueError, match="cached_reset"):
        create_env(FastTD3Config(obs_key=None, policy_obs_dim=None, cached_reset=True), env=env)


def test_action_repeat_rejected_outside_playground_and_dmc():
    from nexrl import XQCConfig
    from nexrl.algos.utils import create_env

    with pytest.raises(ValueError, match="action_repeat"):
        create_env(XQCConfig(env="Pendulum-v1", action_repeat=2))


@pytest.mark.slow  # G1 model load + MJX compile takes ~1 min on CPU
def test_dict_obs_selection():
    env, params = create_playground("G1JoystickFlatTerrain", obs_key="privileged_state")
    expected = env.env.observation_size["privileged_state"]
    shape = env.observation_space(params).shape
    assert shape == tuple(expected)
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    assert obs.shape == shape
    assert env.get_obs(state).shape == shape


def test_dict_obs_requires_obs_key():
    with pytest.raises(ValueError, match="privileged_state"):
        create_playground("G1JoystickFlatTerrain")


def test_obs_key_rejected_for_array_obs():
    with pytest.raises(ValueError, match="obs_key"):
        create_playground("CartpoleBalance", obs_key="state")
