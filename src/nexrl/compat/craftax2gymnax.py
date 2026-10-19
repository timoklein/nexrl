import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParams
from craftax.craftax_env import make_craftax_env_from_name

from nexrl.compat._env_params import make_env_params_class


EnvParams = make_env_params_class(CraftaxEnvParams)

_INT32_MAX = int(jnp.iinfo(jnp.int32).max)


def _add_truncation_info(env):
    """Make `env.step_env` also return `info["truncated"]` and `info["final_obs"]`.

    Craftax's `is_terminal` is `timestep >= max_timesteps | death | (boss defeated)`. Evaluating it with
    `max_timesteps` out of reach leaves only the physical condition, so `truncated = done & ~physical`.
    The env has no autoreset (the algos reset), so the obs returned by step is already the pre-reset obs.
    """

    class CraftaxTruncationEnv(type(env)):
        def step_env(self, key, state, action, params):
            obs, state, reward, done, info = super().step_env(key, state, action, params)
            physical = self.is_terminal(state, params.replace(max_timesteps=_INT32_MAX))
            truncated = jnp.logical_and(done, jnp.logical_not(physical))
            return obs, state, reward, done, {**info, "truncated": truncated, "final_obs": obs}

    CraftaxTruncationEnv.__name__ = CraftaxTruncationEnv.__qualname__ = type(env).__name__
    env.__class__ = CraftaxTruncationEnv
    return env


def create_craftax(env_name, auto_reset=False):
    env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
    if not auto_reset:
        env = _add_truncation_info(env)
    # Wrap params to add max_steps_in_episode
    env_params = EnvParams(_base=env.default_params, max_steps_in_episode=env.default_params.max_timesteps)
    return env, env_params
