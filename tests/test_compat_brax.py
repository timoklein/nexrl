import jax
import jax.numpy as jnp
import numpy as np
import pytest


pytest.importorskip("brax")

from nexrl.compat.brax2gymnax import create_brax


# fmt: off
BRAX_ENVS = [
    "ant", "fast", "halfcheetah", "hopper", "humanoid", "humanoidstandup",
    "inverted_pendulum", "inverted_double_pendulum", "pusher", "reacher",
    "swimmer", "walker2d",
]
# fmt: on


@pytest.mark.parametrize("env_name", BRAX_ENVS)
def test_create_brax_environments(env_name):
    """Test creating and basic functionality of Brax environments."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_brax(env_name)
    except Exception as e:
        pytest.skip(f"Environment {env_name} not available: {e}")

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    obs, state = jitted_reset(rng, params)

    obs_space = env.observation_space(params)
    assert obs.dtype == obs_space.dtype
    assert obs.shape == obs_space.shape

    action_space = env.action_space(params)
    action = action_space.sample(rng)
    assert action.dtype == action_space.dtype
    assert action.shape == action_space.shape

    for _step in range(3):
        obs, state, reward, done, _info = jitted_step(rng, state, action, params)

        assert obs.dtype == obs_space.dtype
        assert obs.shape == obs_space.shape
        assert hasattr(reward, "dtype")
        assert hasattr(done, "dtype")

        action = action_space.sample(rng)


def test_brax_env_params():
    """Test that environment parameters work correctly."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_brax("ant")
    except Exception:
        pytest.skip("Ant environment not available")

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

    obs, state = env.reset(rng, params)
    assert obs is not None
    assert state is not None


def test_brax_truncation_info_and_final_obs():
    episode_length = 4
    env, params = create_brax("inverted_pendulum", episode_length=episode_length)
    step = jax.jit(env.step)
    step_env = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    action = jnp.zeros(env.num_actions)
    for _ in range(2 * episode_length):
        key, k = jax.random.split(key)
        raw_obs, _, _, _, _ = step_env(k, state, action, params)
        obs, new_state, _, done, info = step(k, state, action, params)
        assert info["truncated"].dtype == jnp.bool_
        assert info["final_obs"].shape == obs.shape and info["final_obs"].dtype == obs.dtype
        np.testing.assert_array_equal(info["final_obs"], raw_obs)
        at_limit = int(state.info["steps"]) + 1 >= episode_length
        # truncated only at the time limit; a done before the limit is a physical termination
        if bool(info["truncated"]):
            assert at_limit and bool(done)
        if bool(done) and not at_limit:
            assert not bool(info["truncated"])
        if bool(done):
            assert not np.array_equal(info["final_obs"], obs)  # obs is the post-reset observation
        else:
            np.testing.assert_array_equal(info["final_obs"], obs)
        state = new_state


def test_brax_no_episode_length_never_truncates():
    env, params = create_brax("inverted_pendulum", episode_length=None)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    _, _, _, _, info = env.step(key, state, jnp.zeros(env.num_actions), params)
    assert not bool(info["truncated"]) and info["final_obs"].shape == obs.shape
