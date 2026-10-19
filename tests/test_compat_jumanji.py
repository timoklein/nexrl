from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct


jumanji = pytest.importorskip("jumanji")
from jumanji.env import Environment as JumanjiEnv  # noqa: E402
from jumanji.specs import DiscreteArray, MultiDiscreteArray  # noqa: E402
from jumanji.types import StepType, TimeStep  # noqa: E402

from nexrl.compat.jumanji2gymnax import FlattenObsWrapper, Jumanji2GymnaxEnv, create_jumanji  # noqa: E402


JUMANJI_ENVS = [env for env in jumanji.registered_environments() if not env.startswith("Sokoban")]


@pytest.mark.parametrize("env_name", JUMANJI_ENVS)
def test_create(env_name):
    rng = jax.random.PRNGKey(0)

    if len(jumanji.make(env_name).action_spec.shape) > 1 or isinstance(jumanji.make(env_name).action_spec, MultiDiscreteArray):
        with pytest.raises(NotImplementedError):
            create_jumanji(env_name)
        return

    env, params = create_jumanji(env_name)

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    obs, state = jitted_reset(rng, params)
    env.observation_space(params)

    a = env.action_space(params).sample(rng)

    for _ in range(3):
        obs, state, _reward, _done, _info = jitted_step(rng, state, a, params)

    assert obs.dtype == env.observation_space(params).dtype
    assert obs.shape == env.observation_space(params).shape
    assert a.dtype == env.action_space(params).dtype
    assert a.shape == env.action_space(params).shape


@pytest.mark.parametrize("env_name", JUMANJI_ENVS)
def test_jumanji_env_params(env_name):
    """Test that environment parameters have max_steps_in_episode."""
    try:
        _env, params = create_jumanji(env_name)
    except NotImplementedError:
        pytest.skip(f"Environment {env_name} not created.")

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)


class _ToyState(struct.PyTreeNode):
    count: jax.Array


class _ToyObs(NamedTuple):
    count: jax.Array


class _ToyEnv(JumanjiEnv):
    """Counter env. Action 1 terminates (discount 0); reaching `time_limit` truncates (discount 1)."""

    time_limit = 3

    def __init__(self):
        self._action_spec = DiscreteArray(2)  # built outside jit, like the real envs' cached specs
        super().__init__()

    def reset(self, key):
        state = _ToyState(count=jnp.int32(0))
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.float32(0),
            discount=jnp.float32(1),
            observation=_ToyObs(jnp.float32(0)),
            extras={},
        )
        return state, timestep

    def step(self, state, action):
        count = state.count + 1
        terminated = action == 1
        last = jnp.logical_or(terminated, count >= self.time_limit)
        step_type = jnp.where(last, StepType.LAST, StepType.MID).astype(jnp.int8)
        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.float32(0),
            discount=jnp.where(terminated, 0.0, 1.0),
            observation=_ToyObs(count.astype(jnp.float32)),
            extras={},
        )
        return _ToyState(count=count), timestep

    @property
    def observation_spec(self):
        return None

    @property
    def action_spec(self):
        return self._action_spec


def test_jumanji_truncation_info_and_final_obs():
    """discount != 0 at StepType.LAST is a truncation; discount == 0 is a termination."""
    env = FlattenObsWrapper(Jumanji2GymnaxEnv(_ToyEnv()))
    params = env.default_params
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key, params)
    for t in range(1, _ToyEnv.time_limit + 1):
        obs, state, _reward, done, info = env.step(key, state, jnp.int32(0), params)
        assert info["truncated"].dtype == jnp.bool_
        assert info["final_obs"].shape == obs.shape and info["final_obs"].dtype == obs.dtype
        if t < _ToyEnv.time_limit:
            assert not bool(done) and not bool(info["truncated"])
            np.testing.assert_array_equal(info["final_obs"], obs)
        else:
            assert bool(done) and bool(info["truncated"])
            np.testing.assert_array_equal(info["final_obs"], [float(t)])
            np.testing.assert_array_equal(obs, [0.0])  # obs is the post-reset observation

    obs, state = env.reset(key, params)
    obs, state, _reward, done, info = env.step(key, state, jnp.int32(1), params)
    assert bool(done) and not bool(info["truncated"])
    np.testing.assert_array_equal(info["final_obs"], [1.0])
    np.testing.assert_array_equal(obs, [0.0])
