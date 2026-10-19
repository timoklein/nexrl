import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nexrl.compat import create


def _test_create_env(env_name):
    """Helper: create env, reset, step, check outputs are not None."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create(env_name)
    except Exception as e:
        pytest.skip(f"Environment {env_name} not available: {e}")

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    obs, state = jitted_reset(rng, params)
    action = env.action_space(params).sample(rng)
    obs, state, reward, done, _info = jitted_step(rng, state, action, params)

    assert obs is not None
    assert reward is not None
    assert done is not None


# fmt: off
@pytest.mark.parametrize("env_name", ["CartPole-v1", "MountainCar-v0", "Pendulum-v1"])
def test_create_gymnax_environments(env_name):
    """Test that create function works with native Gymnax environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["brax/ant", "brax/halfcheetah", "brax/hopper"])
def test_create_brax_environments(env_name):
    """Test that create function works with Brax environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["navix/Navix-Empty-6x6-v0", "navix/Navix-FourRooms-v0"])
def test_create_navix_environments(env_name):
    """Test that create function works with Navix environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["jumanji/Snake-v1", "jumanji/Tetris-v0"])
def test_create_jumanji_environments(env_name):
    """Test that create function works with Jumanji environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["gymnasium/CartPole-v1", "gymnasium/Pendulum-v1"])
def test_create_gymnasium_environments(env_name):
    """Test that create function works with Gymnasium environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", [
    "kinetix/s/h0_weak_thrust", "kinetix/m/h1_car_left", "kinetix/l/h5_flappy_bird",
])
def test_create_kinetix_environments(env_name):
    """Test that create function works with Kinetix environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", [
    "craftax/Craftax-Symbolic-v1", "craftax/Craftax-Classic-Symbolic-v1",
])
def test_create_craftax_environments(env_name):
    """Test that create function works with Craftax environments."""
    _test_create_env(env_name)
# fmt: on


def test_create_invalid_prefix():
    """Test that create raises appropriate error for invalid prefix."""
    with pytest.raises(KeyError):
        create("invalid_prefix/some_env")


def test_create_invalid_environment():
    """Test that create handles invalid environment names gracefully."""
    with pytest.raises(ValueError):
        create("NonExistentEnvironment-v999")


def _rollout_check(env, params, n_steps, action_fn, key=jax.random.PRNGKey(0)):
    """Drive env.step; compare with the raw (non-resetting) step_env. Returns (n_trunc, n_physical)."""
    step = jax.jit(env.step)
    step_env = jax.jit(env.step_env)
    key, k = jax.random.split(key)
    obs, state = env.reset(k, params)
    n_trunc = n_physical = 0
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        action = action_fn(k)
        raw_obs, _, _, _, _ = step_env(k, state, action, params)
        obs, new_state, _, done, info = step(k, state, action, params)
        assert info["truncated"].dtype == jnp.bool_
        assert info["final_obs"].shape == obs.shape and info["final_obs"].dtype == obs.dtype
        assert "discount" in info
        np.testing.assert_array_equal(info["final_obs"], raw_obs)
        if bool(done):
            assert not np.array_equal(info["final_obs"], obs)  # obs is the post-reset observation
            if bool(info["truncated"]):
                n_trunc += 1
                assert int(state.time) + 1 >= params.max_steps_in_episode
            else:
                n_physical += 1
        else:
            assert not bool(info["truncated"])
            np.testing.assert_array_equal(info["final_obs"], obs)
        state = new_state
    return n_trunc, n_physical


def test_gymnax_pendulum_truncation_is_time_limit():
    env, params = create("Pendulum-v1")
    params = params.replace(max_steps_in_episode=5)
    n_trunc, n_physical = _rollout_check(env, params, 12, lambda k: jnp.zeros(1))
    assert n_trunc == 2 and n_physical == 0


def test_gymnax_cartpole_truncation_only_at_time_limit():
    env, params = create("CartPole-v1")
    params = params.replace(max_steps_in_episode=15)
    # Always pushing one way topples the pole (physical end) well before step 15 ...
    n_trunc, n_physical = _rollout_check(env, params, 30, lambda k: jnp.int32(1))
    assert n_physical > 0 and n_trunc == 0
    # ... while a limit of 3 steps is always hit first (time-limit end).
    params = params.replace(max_steps_in_episode=3)
    n_trunc, n_physical = _rollout_check(env, params, 9, lambda k: jnp.int32(1))
    assert n_trunc == 3 and n_physical == 0


def test_gymnax_wrapper_delegates_attributes():
    env, params = create("CartPole-v1")
    assert env.name == "CartPole-v1" and env.num_actions == 2
    assert env.obs_shape == (4,)  # delegated to the inner env
    assert env.action_space(params).n == 2
