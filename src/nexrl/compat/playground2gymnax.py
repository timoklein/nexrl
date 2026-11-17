import warnings
from collections.abc import Mapping
from copy import copy
from functools import partial
from typing import Any

import jax
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import numpy as jnp
from mujoco_playground import registry


def create_playground(
    env_name,
    *,
    impl="jax",
    obs_key: str | None = None,
    action_repeat: int = 1,
    cached_reset: bool = False,
    **kwargs,
):
    """Create a MuJoCo Playground env wrapped in the gymnax interface.

    Args:
        env_name: Playground registry name (e.g. "CartpoleBalance", "CheetahRun").
        impl: MJX physics backend. "jax" (XLA, default) or "warp" (NVIDIA Warp).
        obs_key: Entry to select when the env returns a dict observation
            (e.g. "state" or "privileged_state" for locomotion envs). Required for
            dict observations, rejected otherwise. Not forwarded to config_overrides.
        action_repeat: Inner env steps per agent step, rewards summed (as brax's
            EpisodeWrapper, which playground's own training wrapper applies). The
            env config's action_repeat field is not read by the env itself, so this
            is not forwarded to config_overrides.
        cached_reset: Reset a finished episode to the env's first state instead of
            calling the env's reset (see Playground2GymnaxEnvCachedReset). Not
            forwarded to config_overrides.
        **kwargs: Passed as config_overrides to the playground registry.
    """
    overrides = {"impl": impl, **kwargs} if kwargs else {"impl": impl}
    env = registry.load(env_name, config_overrides=overrides)
    config = registry.get_default_config(env_name)
    episode_length = int(config.episode_length)
    env_cls = Playground2GymnaxEnvCachedReset if cached_reset else Playground2GymnaxEnv
    wrapped = env_cls(env, episode_length, obs_key=obs_key, action_repeat=action_repeat)
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

    def __init__(self, env, episode_length: int, obs_key: str | None = None, action_repeat: int = 1):
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {action_repeat}.")
        self.env = env
        # episode_length counts inner env steps; steps in the state count them too
        self.episode_length = episode_length
        self.obs_key = obs_key
        self.action_repeat = action_repeat
        obs_size = env.observation_size
        if isinstance(obs_size, Mapping):
            if obs_key is None:
                raise ValueError(
                    f"{type(env).__name__} returns a dict observation; pass obs_key as one of {sorted(obs_size.keys())}."
                )
            if obs_key not in obs_size:
                raise ValueError(f"obs_key={obs_key!r} not in available keys {sorted(obs_size.keys())}.")
        elif obs_key is not None:
            raise ValueError(f"{type(env).__name__} has a non-dict observation; obs_key={obs_key!r} is not allowed.")

    def _select_obs(self, obs):
        if isinstance(obs, Mapping):
            return obs[self.obs_key]
        return obs

    @property
    def default_params(self):
        # Agent steps per episode
        return PlaygroundEnvParams(max_steps_in_episode=-(-self.episode_length // self.action_repeat))

    def step_env(self, key, state, action, params):
        if self.action_repeat == 1:
            inner = self.env.step(state.inner, action)
        else:
            # As brax's EpisodeWrapper: repeat the action, sum the rewards
            def repeat_step(inner_state, _):
                next_state = self.env.step(inner_state, action)
                return next_state, next_state.reward

            inner, rewards = jax.lax.scan(repeat_step, state.inner, None, length=self.action_repeat)
            inner = inner.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.steps + self.action_repeat
        truncated = steps >= self.episode_length
        inner_done = inner.done.astype(bool)
        done = jnp.logical_or(inner_done, truncated)
        new_state = PlaygroundState(inner=inner, steps=steps)
        obs = self._select_obs(inner.obs)
        # gymnax's step() auto-resets and swaps in the reset obs on done, so the pre-reset
        # obs is only available via info. "truncated" = time limit without physical termination.
        info = {
            "truncated": jnp.logical_and(truncated, jnp.logical_not(inner_done)),
            "final_obs": obs,
        }
        return obs, new_state, inner.reward, done, info

    def reset_env(self, key, params):
        inner = self.env.reset(key)
        state = PlaygroundState(inner=inner, steps=jnp.int32(0))
        return self._select_obs(inner.obs), state

    def get_obs(self, state):
        return self._select_obs(state.inner.obs)

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
        if isinstance(obs_size, Mapping):
            obs_size = obs_size[self.obs_key]
        shape = tuple(obs_size) if isinstance(obs_size, (tuple, list)) else (obs_size,)
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=shape)

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


@struct.dataclass
class PlaygroundFirstState(PlaygroundState):
    """PlaygroundState plus the data and observation of the env's first reset."""

    first_data: Any = None  # mjx.Data after the first reset
    first_obs: Any = None  # observation after the first reset (dict or array, before obs_key selection)


class Playground2GymnaxEnvCachedReset(Playground2GymnaxEnv):
    """Resets a finished episode to the env's first state instead of calling the env's reset.

    As playground's BraxAutoResetWrapper with full_reset=False, the default of its training
    wrapper (wrap_for_brax_training, which the official FastTD3 uses): each env keeps the data
    and observation of its first reset, and a finished episode restarts from them. info is
    kept, as there. gymnax's step instead computes a fresh reset at every step and selects it
    on done, so every step pays for the env's reset (CheetahRun's reset runs 200 physics steps).
    Only the autoreset changes: reset() and reset_env() still reset fully.
    """

    def reset_env(self, key, params):
        obs, state = super().reset_env(key, params)
        first = PlaygroundFirstState(
            inner=state.inner, steps=state.steps, first_data=state.inner.data, first_obs=state.inner.obs
        )
        return obs, first

    def step_env(self, key, state, action, params):
        obs, new_state, reward, done, info = super().step_env(key, state, action, params)
        carried = PlaygroundFirstState(
            inner=new_state.inner, steps=new_state.steps, first_data=state.first_data, first_obs=state.first_obs
        )
        return obs, carried, reward, done, info

    @partial(jax.jit, static_argnames=("self",))
    def step(self, key, state, action, params=None):
        if params is None:
            params = self.default_params
        # gymnax's step hands step_env the first half of its key
        key_step, _ = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key_step, state, action, params)
        inner_re = state_st.inner.replace(
            data=state_st.first_data, obs=state_st.first_obs, done=jnp.zeros_like(state_st.inner.done)
        )
        state_re = state_st.replace(inner=inner_re, steps=jnp.zeros_like(state_st.steps))
        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, self._select_obs(state_st.first_obs), obs_st)
        return obs, state, reward, done, info
