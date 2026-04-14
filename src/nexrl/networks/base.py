"""Base network components: MLP and frozen parameter utilities.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    H: hidden layer size
"""

from collections.abc import Callable, Sequence

import jax
from flax import nnx


class FrozenParam(nnx.Variable):
    """Variable type excluded from optimizer updates.

    Used for RND target network weights. ``wrt=nnx.Param`` in the optimizer
    skips these because FrozenParam is not a subclass of ``nnx.Param``.
    """

    pass


def _freeze_params(module: nnx.Module) -> None:
    """Convert all ``nnx.Param`` variables in the module tree to ``FrozenParam``."""
    for name in list(vars(module)):
        if not isinstance(name, str):
            continue  # skip nnx.List integer indices
        attr = getattr(module, name)
        if isinstance(attr, nnx.Param):
            setattr(module, name, FrozenParam(attr[...]))
        elif isinstance(attr, nnx.List):
            for item in attr:
                if isinstance(item, nnx.Module):
                    _freeze_params(item)
        elif isinstance(attr, nnx.Module):
            _freeze_params(attr)


class MLP(nnx.Module):
    """Multi-layer perceptron with configurable hidden layers and activation."""

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        layers = []
        current_in = in_features
        for size in hidden_layer_sizes:
            layers.append(nnx.Linear(current_in, size, rngs=rngs))
            current_in = size
        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape((x.shape[0], -1))
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
