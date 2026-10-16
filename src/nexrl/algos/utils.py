"""Shared utilities for RL algorithms."""

import dataclasses
from functools import partial
from typing import Any

import jax
import optax
from flax import nnx, struct
from jax import numpy as jnp

from nexrl.compat import create
from nexrl.configs import CNNConfig
from nexrl.networks.encoder import ImpalaCNNEncoder, NatureCNNEncoder
from nexrl.types import ActFn, EnvParams, GymnaxEnv


class RMSState(struct.PyTreeNode):
    """Running mean and standard deviation state for normalization."""

    mean: jax.Array | float
    var: jax.Array | float
    count: jax.Array | float

    @classmethod
    def create(cls, shape: tuple[int, ...]) -> "RMSState":
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=1e-4,
        )


class RewardRMSState(RMSState):
    """RMS state for reward normalization with episodic returns tracking."""

    returns: jax.Array

    @classmethod
    def create(cls, batch_size: int) -> "RewardRMSState":  # type: ignore[override]
        return cls(mean=0, var=1, count=1e-4, returns=jnp.zeros(batch_size))


class FloatObsWrapper:
    """Environment wrapper that converts observations to float."""

    def __init__(self, env: Any):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name in ["env", "reset", "step"]:
            raise AttributeError(name)
        return getattr(self.env, name)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: Any, action: jax.Array, params: Any
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = obs.astype(jnp.float32)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params: Any) -> tuple[jax.Array, Any]:
        obs, state = self.env.reset(key, params)
        obs = obs.astype(jnp.float32)
        return obs, state


_OPTIMISTIC_RESET_RATIO = 16


def create_vec_env_fns(env: GymnaxEnv, config: Any) -> tuple[Any, Any]:
    """Create vectorized step/reset functions, with optimistic reset for Craftax.

    Returns (vmap_step, vmap_reset) matching the calling convention:
      vmap_step(rngs_B, state, action, params) -> (obs, state, reward, done, info)
      vmap_reset(rngs_B, params) -> (obs, state)
    """
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Multi-instance envs (Atari/DMC) hold N independent gymnasium instances and
    # carry each lane's state as an integer env_idx. Their base reset can't recover
    # its vmap lane inside a pure_callback, so it hardcodes instance 0 — aliasing
    # every lane to one instance and corrupting transitions for num_envs>1. Route
    # reset through `reset_with_id`, mapping over an explicit per-lane env_idx, so
    # each lane binds to its own instance. step already routes via env_idx=state.
    reset_with_id = getattr(env, "reset_with_id", None)
    if reset_with_id is not None:
        # reset_with_id is a bound method of MultiInstanceGymnasium2GymnaxEnv; its
        # __self__ is the env, which knows how many instances it holds.
        env_ids = jnp.arange(reset_with_id.__self__._num_envs, dtype=jnp.int32)

        def vmap_reset_multi(rngs_B, params):
            return jax.vmap(reset_with_id, in_axes=(0, 0, None))(rngs_B, env_ids, params)

        return vmap_step, vmap_reset_multi

    is_craftax = isinstance(config.env, str) and config.env.startswith("craftax/")
    if not is_craftax:
        return vmap_step, vmap_reset

    num_envs = config.num_envs
    num_resets = num_envs // _OPTIMISTIC_RESET_RATIO

    def optimistic_step(rngs_B, state, action, params):
        obs_st, state_st, reward, done, info = vmap_step(rngs_B, state, action, params)

        rng_reset = jax.random.fold_in(rngs_B[0], jnp.int32(1))
        rng_choice = jax.random.fold_in(rngs_B[0], jnp.int32(2))

        reset_rngs = jax.random.split(rng_reset, num_resets)
        obs_re, state_re = vmap_reset(reset_rngs, params)

        # Build index mapping: each env maps to one of the pre-computed resets
        reset_indexes = jnp.arange(num_resets).repeat(_OPTIMISTIC_RESET_RATIO)
        done_f = done.astype(jnp.float32)
        safe_p = jnp.where(done_f.sum() > 0, done_f, jnp.ones_like(done_f))

        being_reset = jax.random.choice(
            rng_choice,
            jnp.arange(num_envs),
            shape=(num_resets,),
            p=safe_p,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        def _auto_reset(d, s_re, s_st, o_re, o_st):
            s = jax.tree.map(lambda r, s: jax.lax.select(d, r, s), s_re, s_st)
            o = jax.lax.select(d, o_re, o_st)
            return s, o

        state, obs = jax.vmap(_auto_reset)(done, state_re, state_st, obs_re, obs_st)
        return obs, state, reward, done, info

    return optimistic_step, vmap_reset


def update_rms(rms_state: RMSState, x: jax.Array, batched: bool = True) -> RMSState:
    """Update running mean and standard deviation (Welford's online algorithm)."""
    batch = x if batched else jnp.expand_dims(x, 0)

    batch_count = batch.shape[0]
    batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)

    delta = batch_mean - rms_state.mean
    tot_count = rms_state.count + batch_count

    new_mean = rms_state.mean + delta * batch_count / tot_count
    m_a = rms_state.var * rms_state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * rms_state.count * batch_count / tot_count
    new_var = m2 / tot_count
    new_count = tot_count

    return rms_state.replace(mean=new_mean, var=new_var, count=new_count)


def normalize_obs(rms_state: RMSState, x: jax.Array) -> jax.Array:
    """Normalize observations using running statistics."""
    return (x - rms_state.mean) / jnp.sqrt(rms_state.var + 1e-8)


def update_and_normalize_obs(rms_state: RMSState, x: jax.Array, batched: bool = True) -> tuple[RMSState, jax.Array]:
    """Update RMS state and normalize observations."""
    rms_state = update_rms(rms_state, x, batched)
    return rms_state, normalize_obs(rms_state, x)


def normalize_rew(rms_state: RewardRMSState, r: jax.Array) -> jax.Array:
    """Normalize rewards using running statistics."""
    return r / jnp.sqrt(rms_state.var + 1e-8)


def update_rew_rms(
    rms_state: RewardRMSState,
    rewards: jax.Array,
    dones: jax.Array,
    discount: float,
    batched: bool = True,
) -> RewardRMSState:
    """Update reward RMS state using episodic returns."""
    returns = rewards + (1 - dones) * discount * rms_state.returns
    rms_state = rms_state.replace(returns=returns)
    return update_rms(rms_state, returns, batched=batched)  # type: ignore[return-value]


def update_and_normalize_rew(
    rms_state: RewardRMSState,
    r: jax.Array,
    done: jax.Array,
    discount: float,
    batched: bool = True,
) -> tuple[RewardRMSState, jax.Array]:
    """Update RMS state and normalize rewards."""
    rms_state = update_rew_rms(rms_state, r, done, discount, batched=batched)
    return rms_state, normalize_rew(rms_state, r)


def shuffle_and_split(data: Any, rng: jax.Array, iteration_size: int, num_minibatches: int) -> Any:
    """Shuffle and split data into minibatches."""
    permutation = jax.random.permutation(rng, iteration_size)

    def _shuffle_and_split(x: jax.Array) -> jax.Array:
        x = x.reshape((iteration_size, *x.shape[2:]))
        x = jnp.take(x, permutation, axis=0)
        return x.reshape(num_minibatches, -1, *x.shape[1:])

    return jax.tree.map(_shuffle_and_split, data)


# ---------------------------------------------------------------------------
# Shared helpers for algorithm files
# ---------------------------------------------------------------------------


def create_env(
    config: Any,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
) -> tuple[GymnaxEnv, EnvParams]:
    """Create environment from config, return (env, env_params).

    Handles string-based env creation, optional env/env_params override,
    and FloatObsWrapper for observation normalization.
    Skips FloatObsWrapper when a CNN encoder is configured (encoder handles normalization).
    """
    has_encoder = getattr(config, "encoder_kwargs", None) is not None
    if env is None:
        if isinstance(config.env, str):
            # terminal_on_life_loss is Atari-only (BBF); pass it through only when set
            # so non-Atari envs (which would reject the kwarg) are never handed it.
            extra_env_kwargs = {}
            if getattr(config, "terminal_on_life_loss", False):
                extra_env_kwargs["terminal_on_life_loss"] = True
            # noop_max is Atari-only (DrQ); gate on the namespace so non-Atari envs
            # (e.g. the CartPole test configs) never see the kwarg.
            noop_max = getattr(config, "noop_max", None)
            if noop_max is not None and config.env.startswith("atari/"):
                extra_env_kwargs["noop_max"] = noop_max
            # obs_key selects one entry of a dict observation; playground-only (locomotion envs).
            obs_key = getattr(config, "obs_key", None)
            if obs_key is not None and config.env.startswith("playground/"):
                extra_env_kwargs["obs_key"] = obs_key
            env, env_params = create(config.env, num_envs=getattr(config, "num_envs", 1), **extra_env_kwargs)
        else:
            env = config.env
            env_params = getattr(config, "env_params", None) or env.default_params  # type: ignore[union-attr]
    else:
        if env_params is None:
            env_params = env.default_params
    if config.normalize_observations and not has_encoder:
        env = FloatObsWrapper(env)  # type: ignore[assignment]  # delegates via __getattr__
    return env, env_params  # type: ignore[return-value]


def make_eval_env(env: Any, enabled: bool = True) -> Any:
    """Return an env that evaluation can drive without disturbing training.

    Multi-instance adapters (Atari, DMC) are stateful: their JAX "state" is just an
    int index into a Python list of live gymnasium instances, mutated in place inside
    `pure_callback`. `evaluate` drives the env it is handed (`reset_with_id` + `step`),
    so pointing it at the training env destroys one in-progress training episode per
    env per eval point and leaves training to resume from a stale `last_obs`. Cloning
    gives eval its own instances, matching the reference's separate eval envs.

    Pure-JAX envs (gymnax, brax, craftax, ...) are returned unchanged: their state is a
    real pytree carried by the caller, so `evaluate` runs independent vmapped seeds and
    sharing the env object is harmless.

    `FloatObsWrapper` needs the explicit recursion. It forwards `reset_with_id` through
    `__getattr__`, so `evaluate` still takes the multi-lane branch through it — but
    `getattr(wrapper, "clone")` would resolve to the *inner* env's method and hand back
    an unwrapped env, silently dropping the float cast.

    Args:
        env: The training env, as returned by `create_env`.
        enabled: Pass `config.run_evaluation` — when eval is off, skip building a second
            set of emulator instances.
    """
    if not enabled:
        return env
    if isinstance(env, FloatObsWrapper):
        return FloatObsWrapper(make_eval_env(env.env, True))
    clone = getattr(env, "clone", None)
    return env if clone is None else clone()


def create_encoder(
    encoder_config: CNNConfig,
    obs_shape: tuple[int, ...],
    rngs: nnx.Rngs,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    return_spatial: bool = False,
    kernel_init: nnx.Initializer | None = None,
) -> NatureCNNEncoder | ImpalaCNNEncoder:
    """Create CNN encoder from CNNConfig.

    Args:
        encoder_config: CNN encoder configuration.
        obs_shape: Observation shape in NCHW format (channels, height, width).
        rngs: NNX RNG container.
        compute_dtype: Forward-pass dtype (e.g. jnp.bfloat16). Master weights
            remain fp32. Defaults to fp32 to preserve behavior for non-BBF callers.
        return_spatial: If True (IMPALA only), return the (B, H', W', C) spatial
            latent instead of flat features (BBF convolutional SPR path).
        kernel_init: Conv/Dense kernel initializer. None (default) keeps the NNX
            default (lecun_normal) so PPO/SGF/etc. callers are unchanged; BBF passes
            xavier_uniform to match Schwarzer's reference; DrQ passes it for the
            nature encoder (Dopamine xavier-everywhere).
    """
    in_channels, height, width = obs_shape[0], obs_shape[1], obs_shape[2]
    if encoder_config.encoder == "nature":
        if return_spatial:
            raise ValueError("return_spatial is only supported for the IMPALA encoder.")
        return NatureCNNEncoder(
            in_channels=in_channels,
            height=height,
            width=width,
            features_dim=encoder_config.features_dim,
            rngs=rngs,
            compute_dtype=compute_dtype,
            kernel_init=kernel_init,
            padding=encoder_config.padding,
        )
    elif encoder_config.encoder == "impala":
        return ImpalaCNNEncoder(
            in_channels=in_channels,
            height=height,
            width=width,
            features_dim=encoder_config.features_dim,
            rngs=rngs,
            width_scale=encoder_config.width_scale,
            compute_dtype=compute_dtype,
            return_spatial=return_spatial,
            kernel_init=kernel_init,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_config.encoder!r}. Expected 'nature' or 'impala'.")


def resolve_network_kwargs(config_field: Any, default_activation: str = "swish") -> dict[str, Any]:
    """Extract network kwargs from a config field (e.g. agent_kwargs, actor_kwargs).

    Converts activation string to nnx function, hidden_layer_sizes to tuple.
    Returns a dict ready to be unpacked into a network constructor.
    """
    kwargs = {}
    if config_field is not None:
        kwargs = dataclasses.asdict(config_field)
    activation = kwargs.pop("activation", default_activation)
    kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = kwargs.pop("hidden_layer_sizes", (64, 64))
    kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
    return kwargs


def _polyak_update(online_model: nnx.Module, target_model: nnx.Module, polyak: float) -> None:
    """Polyak-average target params toward online params."""
    online_params = nnx.state(online_model, nnx.Param)
    target_params = nnx.state(target_model, nnx.Param)
    updated = jax.tree.map(
        lambda o, t: polyak * t + (1 - polyak) * o,
        online_params,
        target_params,
    )
    nnx.update(target_model, updated)


def maybe_update_targets(
    online_models: nnx.Module | list[nnx.Module],
    target_models: nnx.Module | list[nnx.Module],
    polyak: float,
    freq: int,
    old_step: int | jax.Array,
    new_step: int | jax.Array,
) -> nnx.Module | list[nnx.Module]:
    """Update target networks — polyak averaging if freq==1, hard copy otherwise.

    Args:
        online_models: single model or list of models (online)
        target_models: single model or list of models (target)
        polyak: polyak coefficient (only used when freq==1)
        freq: target update frequency
        old_step: global step before this iteration
        new_step: global step after this iteration
    """
    if not isinstance(online_models, list):
        online_models = [online_models]
        target_models = [target_models]  # type: ignore[assignment]

    assert isinstance(online_models, list) and isinstance(target_models, list)

    if freq == 1:
        for online, target in zip(online_models, target_models):
            _polyak_update(online, target, polyak)
    else:
        do_update = new_step % freq <= old_step % freq

        def _update(tgts):
            for online, tgt in zip(online_models, tgts):
                nnx.update(tgt, nnx.state(online, nnx.Param))
            return tgts

        def _no_update(tgts):
            return tgts

        target_models = nnx.cond(do_update, _update, _no_update, target_models)

    assert isinstance(target_models, list)
    return target_models if len(target_models) > 1 else target_models[0]


def normalize_minibatch(mb: Any, config: Any, obs_rms: RMSState, rew_rms: RewardRMSState) -> Any:
    """Normalize obs/next_obs/reward on a sampled replay buffer minibatch."""
    if config.normalize_observations:
        mb = mb._replace(
            obs=normalize_obs(obs_rms, mb.obs),
            next_obs=normalize_obs(obs_rms, mb.next_obs),
        )
    if config.normalize_rewards:
        mb = mb._replace(reward=normalize_rew(rew_rms, mb.reward))
    return mb


def calculate_gae(
    rewards_TB: jax.Array,
    values_TB: jax.Array,
    dones_TB: jax.Array,
    last_val_B: jax.Array,
    gamma: float,
    gae_lambda: float,
    episodic: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Generalized Advantage Estimation (GAE-lambda).

    Args:
        rewards_TB: rewards, shape (T, B).
        values_TB: value estimates, shape (T, B).
        dones_TB: episode done flags, shape (T, B).
        last_val_B: bootstrap value for the last step, shape (B,).
        gamma: discount factor.
        gae_lambda: GAE lambda parameter.
        episodic: if False, done flags are zeroed (for non-episodic intrinsic rewards).

    Returns:
        (advantages_TB, targets_TB) where targets = advantages + values.
    """

    def get_advantages(advantage_and_next_value, transition_slice):
        advantage_B, next_value_B = advantage_and_next_value
        reward_B, value_B, done_B = transition_slice
        effective_done_B = done_B if episodic else jnp.zeros_like(done_B)
        delta_B = reward_B + gamma * next_value_B * (1 - effective_done_B) - value_B
        advantage_B = delta_B + gamma * gae_lambda * (1 - effective_done_B) * advantage_B
        return (advantage_B, value_B), advantage_B

    _, advantages_TB = jax.lax.scan(
        get_advantages,
        (jnp.zeros_like(last_val_B), last_val_B),
        (rewards_TB, values_TB, dones_TB),
        reverse=True,
        unroll=16,
    )
    return advantages_TB, advantages_TB + values_TB


def make_eval_act(
    model_fn: ActFn,
    config: Any,
    obs_rms: RMSState | None = None,
) -> ActFn:
    """Build eval policy closure.

    Args:
        model_fn: callable (obs, rng) -> action (batched, single-env obs expanded inside)
        config: algo config (uses normalize_observations)
        obs_rms: optional RMSState for obs normalization
    """

    def act(obs, rng):
        if config.normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        obs = jnp.expand_dims(obs, 0)
        return jnp.squeeze(model_fn(obs, rng))

    return act


def ppo_lr_schedule(
    learning_rate: float,
    anneal_lr: bool,
    total_timesteps: int,
    num_steps: int,
    num_envs: int,
    num_minibatches: int,
    num_epochs: int,
) -> float | optax.Schedule:
    """Create PPO learning rate (constant or linearly annealed)."""
    if not anneal_lr:
        return learning_rate
    num_updates = total_timesteps // num_steps // num_envs

    def schedule(count: jax.Array) -> jax.Array:
        frac = 1.0 - (count // (num_minibatches * num_epochs)) / num_updates
        return learning_rate * frac

    return schedule
