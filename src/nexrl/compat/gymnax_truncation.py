import warnings

import jax.numpy as jnp
from gymnax.environments.environment import Environment as GymnaxEnv


_INT32_MAX = int(jnp.iinfo(jnp.int32).max)


class TruncationInfoWrapper(GymnaxEnv):
    """Adds `info["truncated"]` and `info["final_obs"]` to a native gymnax env.

    gymnax's `is_terminal` ORs a physical condition with `state.time >= max_steps_in_episode`.
    Evaluating it with the time limit pushed out of reach leaves only the physical condition,
    so `truncated = done & ~physical`. `final_obs` is the obs before gymnax's autoreset swaps in
    the reset obs. obs, reward and done are untouched.
    """

    def __init__(self, env: GymnaxEnv):
        self.env = env
        self._has_time_limit = hasattr(env.default_params, "max_steps_in_episode")
        if not self._has_time_limit:
            warnings.warn(
                f"{type(env).__name__} params have no max_steps_in_episode; info['truncated'] is always False.",
                category=RuntimeWarning,
                stacklevel=2,
            )

    def __getattr__(self, name):
        # Only called when normal lookup fails: delegate everything else (obs_shape, ...) to the inner env.
        if name == "env":  # not yet set (copy/pickle): avoid infinite recursion
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def default_params(self):
        return self.env.default_params

    def step_env(self, key, state, action, params):
        obs, state_st, reward, done, info = self.env.step_env(key, state, action, params)
        if self._has_time_limit:
            no_limit = params.replace(max_steps_in_episode=_INT32_MAX)
            physical = jnp.asarray(self.env.is_terminal(state_st, no_limit)).astype(bool)
            truncated = jnp.logical_and(done, jnp.logical_not(physical))
        else:
            truncated = jnp.zeros_like(done, dtype=bool)
        return obs, state_st, reward, done, {**info, "truncated": truncated, "final_obs": obs}

    def reset_env(self, key, params):
        return self.env.reset_env(key, params)

    def get_obs(self, state, params=None, key=None):
        return self.env.get_obs(state, params, key)

    def is_terminal(self, state, params):
        return self.env.is_terminal(state, params)

    @property
    def name(self):
        return self.env.name

    @property
    def num_actions(self):
        return self.env.num_actions

    def action_space(self, params):
        return self.env.action_space(params)

    def observation_space(self, params):
        return self.env.observation_space(params)

    def state_space(self, params):
        return self.env.state_space(params)
