from functools import partial
from typing import Any

import chex
import jax
import navix as nx
from gymnax.environments import environment
from jax import numpy as jnp
from navix.environments import wrappers
from navix.environments.wrappers import GymnaxState


def create_navix(env_name, **kwargs):
    env = nx.make(env_name, **kwargs)
    env = Navix2GymnaxEnv(env)
    return env, env.default_params


def Navix2GymnaxEnv(env: nx.Environment) -> environment.Environment:  # noqa:  N802
    env = wrappers.ToGymnax(env)
    env = FloatObsWrapper(env)
    return env


class FloatObsWrapper(environment.Environment):
    def __init__(self, env):
        self.env = env

    def __getattribute__(self, name: str) -> Any:
        if name in ["env", "reset", "step", "observation_space"]:
            return super().__getattribute__(name)
        return self.env.__getattribute__(name)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[chex.Array, environment.EnvState, float, bool, dict]:
        # navix's own `step` resets lazily (on the step AFTER a done), which would insert a phantom
        # transition. Use its non-resetting `_step` and reset in-step, like gymnax's Environment.step.
        navix_env = self.env.env
        timestep = navix_env._step(state.timestep, action)
        done = timestep.is_done()
        # StepType.TRUNCATION is time limit without termination (termination wins).
        truncated = timestep.is_truncation()
        obs_st = timestep.observation.astype(jnp.float32)
        # The reset timestep has step_type TRANSITION, so navix does not reset again on the next step.
        reset_timestep = navix_env.reset(timestep.state.key, timestep.state.cache)
        new_timestep = jax.tree.map(lambda reset, cur: jax.lax.select(done, reset, cur), reset_timestep, timestep)
        obs = new_timestep.observation.astype(jnp.float32)
        info = {**timestep.info, "truncated": truncated, "final_obs": obs_st}
        return obs, GymnaxState(time=new_timestep.t, timestep=new_timestep), timestep.reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: environment.EnvParams | None = None) -> tuple[chex.Array, environment.EnvState]:
        obs, state = self.env.reset(key, params)
        obs = obs.astype(jnp.float32)
        return obs, state

    def observation_space(self, params):
        space = self.env.observation_space(params)
        space.dtype = jnp.float32
        return space
