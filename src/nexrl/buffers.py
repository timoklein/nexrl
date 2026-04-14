from functools import partial
from typing import NamedTuple

import chex
import jax
from flax import struct
from gymnax.environments import spaces
from jax import numpy as jnp


class CircularBuffer(struct.PyTreeNode):
    size: int = struct.field(pytree_node=False)
    data: chex.ArrayTree
    index: int
    full: bool

    @classmethod
    def empty(cls, size: int, data: chex.ArrayTree) -> "CircularBuffer":
        return cls(size=size, data=data, index=0, full=False)

    @property
    def num_entries(self):
        return jnp.where(self.full, self.size, self.index)

    @jax.jit
    def append(self, a: chex.ArrayTree) -> "CircularBuffer":
        data = jax.tree.map(lambda arr, a_: arr.at[self.index].set(a_), self.data, a)
        next_index = (self.index + 1) % self.size
        full = jnp.logical_or(self.full, next_index == 0)
        return self.replace(data=data, index=next_index, full=full)

    @jax.jit
    def extend(self, batch: chex.ArrayTree) -> "CircularBuffer":
        batch_flat, _ = jax.tree.flatten(batch)
        batch_size = batch_flat[0].shape[0]

        idx = self.index + jnp.arange(batch_size)
        idx = idx % self.size
        data = jax.tree.map(lambda arr, b: arr.at[idx].set(b), self.data, batch)

        next_index = (self.index + batch_size) % self.size
        full = jnp.logical_or(self.full, next_index == 0)
        return self.replace(data=data, index=next_index, full=full)


class Minibatch(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array


class NStepState(NamedTuple):
    """Ring buffer for n-step return accumulation in carry."""

    obs: jax.Array  # (n_step, num_envs, *obs_shape)
    action: jax.Array  # (n_step, num_envs, *action_shape)
    reward: jax.Array  # (n_step, num_envs)
    done: jax.Array  # (n_step, num_envs)
    next_obs: jax.Array  # (n_step, num_envs, *obs_shape)
    write_idx: int  # scalar: next write position in ring
    count: jax.Array  # (num_envs,): number of valid entries per env


def create_nstep_state(
    n_step: int,
    num_envs: int,
    obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    obs_dtype: jnp.dtype = jnp.float32,
    action_dtype: jnp.dtype = jnp.int32,
) -> NStepState:
    return NStepState(
        obs=jnp.zeros((n_step, num_envs, *obs_shape), dtype=obs_dtype),
        action=jnp.zeros((n_step, num_envs, *action_shape), dtype=action_dtype),
        reward=jnp.zeros((n_step, num_envs), dtype=jnp.float32),
        done=jnp.zeros((n_step, num_envs), dtype=bool),
        next_obs=jnp.zeros((n_step, num_envs, *obs_shape), dtype=obs_dtype),
        write_idx=0,
        count=jnp.zeros(num_envs, dtype=jnp.int32),
    )


class PrioritizedReplayBuffer(CircularBuffer):
    """Circular buffer with proportional prioritized experience replay."""

    data: Minibatch
    priorities: chex.Array  # (size,)

    @classmethod
    def empty(
        cls,
        size: int,
        obs_space: spaces.Discrete | spaces.Box,
        action_space: spaces.Discrete | spaces.Box,
    ) -> "PrioritizedReplayBuffer":
        data = Minibatch(
            obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
            action=jnp.empty((size, *action_space.shape)).astype(action_space.dtype),
            reward=jnp.empty(size),
            done=jnp.empty(size).astype(bool),
            next_obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
        )
        return cls(size=size, data=data, index=0, full=False, priorities=jnp.ones(size, dtype=jnp.float32))

    def extend_with_priorities(self, batch: Minibatch, priorities: chex.Array) -> "PrioritizedReplayBuffer":
        batch_flat, _ = jax.tree.flatten(batch)
        batch_size = batch_flat[0].shape[0]

        idx = (self.index + jnp.arange(batch_size)) % self.size
        data = jax.tree.map(lambda arr, b: arr.at[idx].set(b), self.data, batch)
        new_priorities = self.priorities.at[idx].set(priorities)

        next_index = (self.index + batch_size) % self.size
        full = jnp.logical_or(self.full, next_index == 0)
        return self.replace(data=data, index=next_index, full=full, priorities=new_priorities)

    @partial(jax.jit, static_argnames=("num",))
    def sample(self, num: int, rng: chex.PRNGKey, beta: float) -> tuple[Minibatch, chex.Array, chex.Array]:
        """Proportional PER sampling via cumsum + searchsorted (stratified).

        Returns (minibatch, indices, importance_weights).
        """
        n = self.num_entries
        # Mask out unused slots (beyond num_entries) by zeroing their priorities
        valid_mask = jnp.arange(self.size) < n
        probs = self.priorities * valid_mask
        total = probs.sum()
        probs_norm = probs / jnp.maximum(total, 1e-8)

        # Stratified sampling
        cumsum = jnp.cumsum(probs_norm)
        segment_size = 1.0 / num
        offsets = jnp.arange(num, dtype=jnp.float32) * segment_size
        u = jax.random.uniform(rng, (num,), minval=0.0, maxval=segment_size)
        sample_points = offsets + u
        indices = jnp.searchsorted(cumsum, sample_points)
        indices = jnp.clip(indices, 0, n - 1)

        # Importance sampling weights (clamp to avoid 0^(-beta) = inf)
        sampled_probs = jnp.maximum(probs_norm[indices], 1e-8)
        weights = (n * sampled_probs) ** (-beta)
        weights = weights / jnp.maximum(weights.max(), 1e-8)

        mb = jax.tree.map(lambda arr: arr[indices], self.data)
        return mb, indices, weights

    def update_priorities(self, indices: chex.Array, new_priorities: chex.Array) -> "PrioritizedReplayBuffer":
        return self.replace(priorities=self.priorities.at[indices].set(new_priorities))


class ReplayBuffer(CircularBuffer):
    """
    Circular buffer for storing transitions. Implements appending and sampling
    while being `jit`-able.
    """

    data: Minibatch

    @classmethod
    def empty(
        cls,
        size: int,
        obs_space: spaces.Discrete | spaces.Box,
        action_space: spaces.Discrete | spaces.Box,
    ) -> "ReplayBuffer":
        """Returns an empty replay buffer with the given size and shapes.

        Args:
            size (int): Maximum number of transitions to store.
            obs_shape (chex.Shape): Shape of the observations.
            action_shape (chex.Shape): Shape of the actions.

        Returns:
            ReplayBuffer: The initialized replay buffer.
        """
        # Skip checking sizes as we know they are correct here
        data = Minibatch(
            obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
            action=jnp.empty((size, *action_space.shape)).astype(action_space.dtype),
            reward=jnp.empty(size),
            done=jnp.empty(size).astype(bool),
            next_obs=jnp.empty((size, *obs_space.shape)).astype(obs_space.dtype),
        )
        return cls(size=size, data=data, index=0, full=False)

    def __getattr__(self, name):
        if name in self.data._fields:
            return getattr(self.data, name)

    @partial(jax.jit, static_argnames=("num"))
    def sample(self, num: int, rng: chex.PRNGKey) -> Minibatch:
        """Samples a minibatch of transitions from the buffer, without replacement.
        Note that this function does not check if enough transitions are stored, and
        might return the same transition multiple times.

        Args:
            num (int): The size of the minibatch to sample.
            rng (chex.PRNGKey): The random number generator key.

        Returns:
            Minibatch: A minibatch of randomly sampled transitions.
        """
        minibatch_index = jax.random.randint(rng, (num,), 0, self.num_entries)
        return jax.tree.map(lambda arr: arr[minibatch_index], self.data)
