import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nexrl.compat.gymnasium2gymnax import create_gymnasium


# fmt: off
GYMNASIUM_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]
DISCRETE_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
CONTINUOUS_ENVS = ["Pendulum-v1"]
# fmt: on


@pytest.mark.parametrize("env_name", GYMNASIUM_ENVS)
def test_create_gymnasium_environments(env_name):
    """Test creating and basic functionality of Gymnasium environments."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium(env_name)
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

    for _step in range(10):
        obs, state, reward, done, _info = jitted_step(rng, state, action, params)

        assert obs.dtype == obs_space.dtype
        assert obs.shape == obs_space.shape
        assert hasattr(reward, "dtype")
        assert hasattr(done, "dtype")

        if done:
            break

        action = action_space.sample(rng)


@pytest.mark.parametrize("env_name", DISCRETE_ENVS)
def test_gymnasium_discrete_environments(env_name):
    """Test discrete action environments specifically."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium(env_name)
    except Exception:
        pytest.skip(f"Environment {env_name} not available")

    action_space = env.action_space(params)
    from gymnax.environments.spaces import Discrete

    assert isinstance(action_space, Discrete)

    obs, state = env.reset(rng, params)
    action = action_space.sample(rng)
    obs, state, reward, done, _info = env.step(rng, state, action, params)

    assert obs is not None
    assert reward is not None
    assert done is not None


@pytest.mark.parametrize("env_name", CONTINUOUS_ENVS)
def test_gymnasium_continuous_environments(env_name):
    """Test continuous action environments specifically."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium(env_name)
    except Exception:
        pytest.skip(f"Environment {env_name} not available")

    action_space = env.action_space(params)
    from gymnax.environments.spaces import Box

    assert isinstance(action_space, Box)

    obs, state = env.reset(rng, params)
    action = action_space.sample(rng)
    obs, state, reward, done, _info = env.step(rng, state, action, params)

    assert obs is not None
    assert reward is not None
    assert done is not None


def test_gymnasium_env_params():
    """Test that environment parameters work correctly."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium("CartPole-v1")
    except Exception:
        pytest.skip("CartPole environment not available")

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

    obs, state = env.reset(rng, params)
    assert obs is not None
    assert state is not None


# --- Episode-end convention: info["truncated"], info["final_obs"], in-step reset -----------


@pytest.mark.parametrize(
    ("max_episode_steps", "expect_truncated"),
    [(8, True), (500, False)],  # 8-step limit: time-limit ends; constant push: the pole falls
)
def test_gymnasium_truncation_info_and_final_obs(max_episode_steps, expect_truncated):
    """`truncated` only at time-limit ends, `final_obs` is the pre-reset obs, `obs` on done is
    the reset obs, and no phantom follows a done (CartPole pays reward 1 on every real step)."""
    num_lanes = 2  # vmapped like training: one gymnasium env serves both lanes
    env, params = create_gymnasium("CartPole-v1", max_episode_steps=max_episode_steps)
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    step_env = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0, None)))
    obs, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(jax.random.split(jax.random.PRNGKey(0), 2), params)
    # Alternate pushes under the time limit (the pole stays up); push right otherwise.
    ep_len, num_done = np.zeros(num_lanes, dtype=int), 0
    for t in range(30):
        action = jnp.full((num_lanes,), t % 2 if expect_truncated else 1, dtype=jnp.int32)
        keys = jax.random.split(jax.random.PRNGKey(t + 1), num_lanes)
        raw_obs = step_env(keys, state, action, params)[0]  # non-resetting step from the same state
        obs, state, reward, done, info = step(keys, state, action, params)
        assert info["truncated"].dtype == jnp.bool_
        assert info["final_obs"].shape == obs.shape and info["final_obs"].dtype == obs.dtype
        np.testing.assert_array_equal(reward, 1.0)
        np.testing.assert_array_equal(info["final_obs"], raw_obs)
        ep_len += 1
        for i in range(num_lanes):
            if not done[i]:
                assert not info["truncated"][i]
                np.testing.assert_array_equal(info["final_obs"][i], obs[i])
                continue
            num_done += 1
            assert bool(info["truncated"][i]) == expect_truncated
            if expect_truncated:
                assert ep_len[i] == max_episode_steps
            assert not np.array_equal(info["final_obs"][i], obs[i])  # obs is the reset obs
            assert np.all(np.abs(np.asarray(obs[i])) <= 0.05)  # CartPole resets to U(-0.05, 0.05)
            assert int(state["_nexrl_elapsed_steps"][i]) == 0
            ep_len[i] = 0
    assert num_done >= num_lanes


@pytest.mark.parametrize(
    ("max_num_frames_per_episode", "expect_truncated"),
    [(100, True), (None, False)],  # 100 frames = 25 agent steps; default cap: game over first
)
def test_atari_truncation_info_and_final_obs(max_num_frames_per_episode, expect_truncated):
    """Same convention for the multi-instance Atari adapter. FrameStack pads a reset obs with
    copies of the reset frame, so a reset obs has 4 equal frames and the next obs continues it."""
    pytest.importorskip("ale_py")
    from nexrl.compat.gymnasium2gymnax import create_atari

    kwargs = {} if max_num_frames_per_episode is None else {"max_num_frames_per_episode": max_num_frames_per_episode}
    env, params = create_atari("Breakout-v5", num_envs=1, **kwargs)
    step = jax.jit(env.step)
    obs, state = jax.jit(env.reset_with_id)(jax.random.PRNGKey(0), jnp.int32(0), params)
    rng = np.random.default_rng(0)
    prev_obs, num_done, after_done = np.asarray(obs), 0, None
    for _ in range(2000):
        action = jnp.int32(rng.integers(env.num_actions))
        obs, state, _reward, done, info = step(jax.random.PRNGKey(0), state, action, params)
        obs, final_obs = np.asarray(obs), np.asarray(info["final_obs"])
        assert final_obs.shape == obs.shape and final_obs.dtype == obs.dtype
        np.testing.assert_array_equal(final_obs[:-1], prev_obs[1:])  # final_obs continues the episode
        if after_done is not None:  # no phantom: this step continued from the reset obs
            np.testing.assert_array_equal(obs[:-1], after_done[1:])
            after_done = None
        if done:
            num_done += 1
            assert bool(info["truncated"]) == expect_truncated
            assert all(np.array_equal(obs[0], obs[c]) for c in range(1, 4))  # obs is the reset obs
            if not expect_truncated:  # a 25-step episode can end on the static pre-launch screen
                assert not np.array_equal(final_obs, obs)
            after_done = obs
        else:
            assert not info["truncated"]
            np.testing.assert_array_equal(final_obs, obs)
        prev_obs = obs
        if num_done == 2 and after_done is None:
            break
    assert num_done == 2
