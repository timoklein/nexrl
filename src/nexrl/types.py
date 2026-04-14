"""Shared type aliases for nexrl RL algorithms."""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import jax


# Type aliases
TrainState = dict[str, Any]
EvalMetrics = tuple[jax.Array, jax.Array]  # (lengths, returns)
Metrics = dict[str, jax.Array]
TrainFn = Callable[..., tuple[TrainState, Metrics]]
ActFn = Callable[[jax.Array, jax.Array], jax.Array]
EnvParams = Any  # gymnax params have no common base


class ActorCriticProtocol(Protocol):
    """Structural protocol for actor-critic modules.

    ActorCritic, SharedDiscreteActorCritic, and SharedMLPDiscreteActorCritic
    all conform without explicit inheritance.
    """

    def get_action_and_value(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]: ...

    def get_pi_and_value(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]: ...

    def value(self, obs: jax.Array) -> jax.Array: ...

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array: ...


@runtime_checkable
class GymnaxEnv(Protocol):
    """Structural type for gymnax-compatible environments.

    Both gymnax.Environment and FloatObsWrapper satisfy this protocol.
    """

    def reset(self, key: jax.Array, params: Any) -> tuple[jax.Array, Any]: ...

    def step(
        self, key: jax.Array, state: Any, action: jax.Array, params: Any
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]]: ...

    def action_space(self, params: Any) -> Any: ...

    def observation_space(self, params: Any) -> Any: ...

    @property
    def default_params(self) -> Any: ...
