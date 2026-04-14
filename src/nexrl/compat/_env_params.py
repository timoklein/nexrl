"""Shared EnvParams wrapper for environments that lack max_steps_in_episode."""

from typing import Any

from flax import struct


def make_env_params(max_steps: int = 1000):
    """Create a simple EnvParams with just max_steps_in_episode."""

    @struct.dataclass
    class SimpleEnvParams:
        max_steps_in_episode: int = struct.field(pytree_node=False, default=1000)

    return SimpleEnvParams(max_steps_in_episode=max_steps)


def make_env_params_class(base_type: type) -> type:
    """Create an EnvParams wrapper class for a given base params type.

    The wrapper adds ``max_steps_in_episode`` (pytree_node=False) and delegates
    attribute access to the underlying ``_base``. The ``replace`` method
    auto-syncs ``max_steps_in_episode`` <-> ``max_timesteps``.
    """

    @struct.dataclass
    class EnvParams:
        _base: Any  # typed loosely; actual type enforced at call site
        max_steps_in_episode: int = struct.field(pytree_node=False, default=None)

        def __getattr__(self, name):
            try:
                return getattr(self._base, name)
            except AttributeError as err:
                raise AttributeError(f"{type(self).__name__} has no attribute {name!r}") from err

        def __repr__(self):
            base_repr = ", ".join(f"{k}={v!r}" for k, v in vars(self._base).items())
            return f"{type(self).__name__}({base_repr}, max_steps_in_episode={self.max_steps_in_episode!r})"

    _original_replace = EnvParams.replace

    def _custom_replace(self, **updates):
        """Custom replace with auto-sync between max_steps_in_episode and max_timesteps."""
        top = {"_base", "max_steps_in_episode"}
        base_updates = {k: v for k, v in updates.items() if k not in top}
        top_updates = {k: v for k, v in updates.items() if k in top}

        if "max_steps_in_episode" in top_updates:
            base_updates["max_timesteps"] = top_updates["max_steps_in_episode"]
        if "max_timesteps" in base_updates:
            top_updates["max_steps_in_episode"] = base_updates["max_timesteps"]

        base = self._base
        if base_updates:
            if hasattr(base, "replace"):
                base = base.replace(**base_updates)
            else:
                d = dict(vars(base))
                d.update(base_updates)
                base = base.__class__(**d)

        final_updates = top_updates.copy()
        final_updates["_base"] = base
        return _original_replace(self, **final_updates)

    EnvParams.replace = _custom_replace
    return EnvParams
