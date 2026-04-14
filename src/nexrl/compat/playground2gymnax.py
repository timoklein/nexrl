import warnings
from copy import copy
from typing import Any

from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import numpy as jnp
from mujoco_playground import registry


def create_playground(env_name, *, impl="jax", **kwargs):
    """Create a MuJoCo Playground env wrapped in the gymnax interface.

    Args:
        env_name: Playground registry name (e.g. "CartpoleBalance", "CheetahRun").
        impl: MJX physics backend. "jax" (XLA, default) or "warp" (NVIDIA Warp).
        **kwargs: Passed as config_overrides to the playground registry.
    """
    overrides = {"impl": impl, **kwargs} if kwargs else {"impl": impl}
    env = registry.load(env_name, config_overrides=overrides)
    config = registry.get_default_config(env_name)
    episode_length = int(config.episode_length)
    wrapped = Playground2GymnaxEnv(env, episode_length)
    return wrapped, wrapped.default_params


@struct.dataclass
class PlaygroundEnvParams:
    # CAUTION: Passing params with a different value than on init has no effect
    max_steps_in_episode: int = 1000


@struct.dataclass
class PlaygroundState:
    """Wraps playground State with a step counter for episode truncation.

    Playground envs only set done=True for physical termination (e.g. NaN).
    Episode length truncation is handled by the step counter.
    """

    inner: Any  # mujoco_playground State (flax struct, valid pytree)
    steps: int = 0


class Playground2GymnaxEnv(GymnaxEnv):
    """Wraps a MuJoCo Playground MjxEnv into the gymnax GymnaxEnv protocol.

    Playground envs are fully JAX-native (jittable, vmappable), so this wrapper
    is lightweight — no jax.pure_callback needed (unlike dm_control).
    """

    def __init__(self, env, episode_length: int):
        self.env = env
        self.episode_length = episode_length

    @property
    def default_params(self):
        return PlaygroundEnvParams(max_steps_in_episode=self.episode_length)

    def step_env(self, key, state, action, params):
        inner = self.env.step(state.inner, action)
        steps = state.steps + 1
        truncated = steps >= self.episode_length
        done = jnp.logical_or(inner.done.astype(bool), truncated)
        new_state = PlaygroundState(inner=inner, steps=steps)
        return inner.obs, new_state, inner.reward, done, {}

    def reset_env(self, key, params):
        inner = self.env.reset(key)
        state = PlaygroundState(inner=inner, steps=jnp.int32(0))
        return inner.obs, state

    def get_obs(self, state):
        return state.inner.obs

    def is_terminal(self, state):
        return jnp.logical_or(
            state.inner.done.astype(bool),
            state.steps >= self.episode_length,
        )

    @property
    def name(self):
        return type(self.env).__name__

    def action_space(self, params):
        # DMC suite envs use ctrlrange="-1 1"
        return spaces.Box(low=-1, high=1, shape=(self.env.action_size,))

    def observation_space(self, params):
        obs_size = self.env.observation_size
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(obs_size,))

    @property
    def num_actions(self) -> int:
        return self.env.action_size

    def __deepcopy__(self, memo):
        warnings.warn(
            f"Trying to deepcopy {type(self).__name__}, which contains a MuJoCo Playground env. "
            "MuJoCo Playground envs may throw an error when deepcopying, so a shallow copy is returned.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return copy(self)
