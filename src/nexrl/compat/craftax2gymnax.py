from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParams
from craftax.craftax_env import make_craftax_env_from_name

from nexrl.compat._env_params import make_env_params_class


EnvParams = make_env_params_class(CraftaxEnvParams)


def create_craftax(env_name, auto_reset=False):
    env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
    # Wrap params to add max_steps_in_episode
    env_params = EnvParams(_base=env.default_params, max_steps_in_episode=env.default_params.max_timesteps)
    return env, env_params
