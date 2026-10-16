import jax
import jax.numpy as jnp
import numpy as np
import pytest


mujoco_playground = pytest.importorskip("mujoco_playground")

from mujoco_playground import registry  # noqa: E402

from nexrl.compat.playground2gymnax import Playground2GymnaxEnv, create_playground  # noqa: E402


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
