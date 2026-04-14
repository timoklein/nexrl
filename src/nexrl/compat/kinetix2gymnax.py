from kinetix.environment import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.environment.spaces import EnvParams as KinetixEnvParams
from kinetix.util import load_from_json_file

from nexrl.compat._env_params import make_env_params_class


EnvParams = make_env_params_class(KinetixEnvParams)


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

    env_params = EnvParams(_base=env_params, max_steps_in_episode=env_params.max_timesteps)
    return env, env_params
