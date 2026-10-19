import importlib

import jax
from gymnax import make as _gymnax_make

from nexrl.compat.gymnax_truncation import TruncationInfoWrapper


_create_fns = {
    "brax": ("nexrl.compat.brax2gymnax", "create_brax"),
    "navix": ("nexrl.compat.navix2gymnax", "create_navix"),
    "jumanji": ("nexrl.compat.jumanji2gymnax", "create_jumanji"),
    "gymnasium": ("nexrl.compat.gymnasium2gymnax", "create_gymnasium"),
    "kinetix": ("nexrl.compat.kinetix2gymnax", "create_kinetix"),
    "craftax": ("nexrl.compat.craftax2gymnax", "create_craftax"),
    "atari": ("nexrl.compat.gymnasium2gymnax", "create_atari"),
    "dm_control": ("nexrl.compat.gymnasium2gymnax", "create_dmc"),
    "playground": ("nexrl.compat.playground2gymnax", "create_playground"),
}


_ENVS_ACCEPTING_NUM_ENVS = {"atari", "dm_control"}


def create(env, **kwargs):
    if len(split := env.split("/", 1)) == 1:
        kwargs.pop("num_envs", None)  # gymnax.make doesn't accept num_envs
        gymnax_env, params = _gymnax_make(env, **kwargs)
        return TruncationInfoWrapper(gymnax_env), params

    prefix, env_name = split
    if prefix not in _ENVS_ACCEPTING_NUM_ENVS:
        kwargs.pop("num_envs", None)
    module, create_fn = _create_fns[prefix]
    # ensure_compile_time_eval so that module-level jnp ops in third-party
    # packages (e.g. Craftax constants.py) produce concrete arrays, not tracers,
    # even when this import is first triggered during JIT tracing.
    with jax.ensure_compile_time_eval():
        module = importlib.import_module(module)
        create_fn = getattr(module, create_fn)
        return create_fn(env_name, **kwargs)


__all__ = ["create"]
