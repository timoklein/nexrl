import jax
import jax.numpy as jnp
from kinetix.environment import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.environment.spaces import EnvParams as KinetixEnvParams
from kinetix.util import load_from_json_file

from nexrl.compat._env_params import make_env_params_class


EnvParams = make_env_params_class(KinetixEnvParams)

_INT32_MAX = int(jnp.iinfo(jnp.int32).max)


def _add_truncation_info(env):
    """Make `env.step_env` also return `info["truncated"]` and `info["final_obs"]`.

    Kinetix's `engine_step` computes `done = physical | (timestep >= max_timesteps)` and has no separate
    physical flag. Running it once with `max_timesteps` out of reach gives the physical `done`; the time
    limit is then applied here, including the `last_distance = -1` reset that `engine_step` does on done.
    Physics runs once, so obs, reward and done are unchanged. `step()` swaps in the reset obs on done, so
    the pre-reset obs is only available via info.
    """

    class KinetixTruncationEnv(type(env)):
        def step_env(self, rng, state, action, env_params):
            action_processed = self.action_type.process_action(action, state, self.static_env_params)
            no_limit = env_params.replace(max_timesteps=_INT32_MAX)
            obs, state, reward, physical, info = self.engine_step(state, action_processed, no_limit)
            time_up = state.timestep >= env_params.max_timesteps
            done = jnp.logical_or(physical, time_up)
            state = state.replace(last_distance=jax.lax.select(done, -1.0, state.last_distance))
            truncated = jnp.logical_and(time_up, jnp.logical_not(physical))
            return obs, state, reward, done, {**info, "truncated": truncated, "final_obs": obs}

    KinetixTruncationEnv.__name__ = KinetixTruncationEnv.__qualname__ = type(env).__name__
    env.__class__ = KinetixTruncationEnv
    return env


def create_kinetix(env_name, **kwargs):
    """Create a Kinetix environment. The `env_name` argument is the name of the
    level file, e.g. `"l/grasp_easy"`.
    """
    level, static_env_params, env_params = load_from_json_file(f"{env_name}.json")

    default_kwargs = {
        "action_type": ActionType.CONTINUOUS,
        "observation_type": ObservationType.SYMBOLIC_FLAT,
        "reset_fn": lambda rng: level,
        "env_params": env_params,
        "static_env_params": static_env_params,
    }
    default_kwargs.update(kwargs)

    env = make_kinetix_env(**default_kwargs)
    env = _add_truncation_info(env)

    env_params = EnvParams(_base=env_params, max_steps_in_episode=env_params.max_timesteps)
    return env, env_params
