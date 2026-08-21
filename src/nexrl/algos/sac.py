"""Soft Actor-Critic (SAC) — standalone training function.

"Soft Actor-Critic Algorithms and Applications" (Haarnoja et al., 2018)

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (continuous action_dim)
    C: num critics (2)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

from nexrl.algos.utils import (
    RewardRMSState,
    RMSState,
    create_encoder,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    make_eval_env,
    maybe_update_targets,
    normalize_minibatch,
    normalize_obs,
    resolve_network_kwargs,
    update_rew_rms,
    update_rms,
)
from nexrl.buffers import Minibatch, ReplayBuffer
from nexrl.configs import SACConfig
from nexrl.evaluate import evaluate
from nexrl.networks import QNetwork, SquashedGaussianPolicy
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


NUM_CRITICS = 2

_SAC_METRIC_NAMES = ("actor_loss", "critic_loss", "alpha", "entropy")


class SACCarry(NamedTuple):
    """Non-module carry state for SAC training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    log_alpha: chex.Array
    alpha_opt_state: Any
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _create_networks(
    config: SACConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[SquashedGaussianPolicy, nnx.Optimizer, list[QNetwork], list[nnx.Optimizer], list[QNetwork], int]:
    """Create actor, twin critics, and twin critic targets."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
    log_std_range = agent_kwargs.pop("log_std_range", (-10, 2))

    actor_kwargs = {
        "in_features": in_features,
        "action_dim": action_dim,
        "action_range": action_range,
        "log_std_range": log_std_range,
        **agent_kwargs,
    }
    critic_kwargs = {
        "obs_dim": in_features,
        "action_dim": action_dim,
        **agent_kwargs,
    }

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )

    rngs = nnx.Rngs(rng)

    actor_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    actor = SquashedGaussianPolicy(**actor_kwargs, rngs=rngs, encoder=actor_encoder)
    actor_opt = nnx.Optimizer(actor, tx, wrt=nnx.Param)

    critics = []
    critic_opts = []
    for _ in range(NUM_CRITICS):
        critic_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
        critic = QNetwork(**critic_kwargs, rngs=rngs, encoder=critic_encoder)
        critics.append(critic)
        critic_opts.append(nnx.Optimizer(critic, tx, wrt=nnx.Param))

    critic_targets = []
    for i in range(NUM_CRITICS):
        critic_target_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
        target = QNetwork(**critic_kwargs, rngs=rngs, encoder=critic_target_encoder)
        nnx.update(target, nnx.state(critics[i], nnx.Param))
        critic_targets.append(target)

    return actor, actor_opt, critics, critic_opts, critic_targets, action_dim


def train_sac(
    config: SACConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a SAC agent.

    Soft Actor-Critic with automatic entropy tuning for continuous control.

    Args:
        config: SACConfig with hyperparameters and environment specification.
        rng: PRNG key for initialization and stochasticity. Use
            ``jax.vmap`` over different keys for independent seeds.
        env: Optional gymnax-compatible environment. Overrides ``config.env``
            when provided.
        env_params: Optional environment parameters. Uses defaults if ``None``.
        log_fn: Optional callback invoked at each evaluation point via
            ``jax.debug.callback``. Receives ``(step, metrics_dict)``.

    Returns:
        A ``(state, metrics)`` tuple where ``state`` is a dict of trained
        parameters and ``metrics`` is a dict of JAX arrays with keys
        including ``eval_returns``, ``eval_lengths``, and ``global_step``.

    Example::

        from nexrl import get_train_fn, SACConfig

        config = SACConfig.from_yaml("configs/sac/gymnax/pendulum.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("sac"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()

    # --- Setup ---
    env, env_params = create_env(config, env, env_params)
    # Dedicated eval instances — sharing stateful Atari/DMC envs corrupts training (see make_eval_env).
    eval_env = make_eval_env(env, config.run_evaluation)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    actor, actor_optimizer, critics, critic_opts, critic_targets, action_dim = _create_networks(
        config, env, env_params, rngs()
    )

    # Compute target entropy
    target_entropy = float(-action_dim)

    # Alpha optimizer (pure optax, not nnx.Optimizer since it's a scalar)
    alpha_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    log_alpha = jnp.array(0.0)
    alpha_opt_state = alpha_tx.init(log_alpha)

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _SAC_METRIC_NAMES}

    carry = SACCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect transitions ---
    def collect_transitions(carry, actor, rngs):
        last_obs = carry.last_obs
        if config.normalize_observations:
            last_obs = normalize_obs(carry.obs_rms, last_obs)

        action_BA = actor.act(last_obs, rngs())

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_BA, env_params)

        # Track episode returns using raw rewards
        new_running_B = carry.running_return + reward_B
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * done_B)
        completed_count = carry.completed_returns_count + jnp.sum(done_B.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - done_B)

        obs_rms = carry.obs_rms
        rew_rms = carry.rew_rms
        if config.normalize_observations:
            obs_rms = update_rms(obs_rms, next_obs)
        if config.normalize_rewards:
            rew_rms = update_rew_rms(rew_rms, reward_B, done_B, config.gamma)

        mb = Minibatch(obs=carry.last_obs, action=action_BA, reward=reward_B, next_obs=next_obs, done=done_B)
        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry, mb

    # --- Update actor ---
    def update_actor(carry, mb, actor, actor_optimizer, critics, rngs):
        alpha = jnp.exp(carry.log_alpha)
        action_rng = rngs()

        def actor_loss_fn(actor_model):
            action_BA, log_prob_B = actor_model.action_log_prob(mb.obs, action_rng)
            qs_CB = jnp.stack([critic(mb.obs, action_BA) for critic in critics])  # (C, B)
            q_min_B = jnp.min(qs_CB, axis=0)
            loss = (alpha * log_prob_B - q_min_B).mean()
            return loss, log_prob_B

        (actor_loss, log_prob_B), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)
        actor_optimizer.update(actor, grads)
        entropy = -log_prob_B.mean()
        return log_prob_B, actor_loss, entropy

    # --- Update critics ---
    def update_critics(carry, mb, actor, critics, critic_opts, critic_targets, rngs):
        alpha = jnp.exp(carry.log_alpha)

        next_action_BA, next_log_prob_B = actor.action_log_prob(mb.next_obs, rngs())

        qs_target_CB = jnp.stack([critic(mb.next_obs, next_action_BA) for critic in critic_targets])  # (C, B)
        q_target_B = jnp.min(qs_target_CB, axis=0)
        q_target_B = q_target_B - alpha * next_log_prob_B
        target_B = mb.reward + (1 - mb.done) * config.gamma * q_target_B

        total_critic_loss = jnp.float32(0.0)
        for critic_model, critic_opt in zip(critics, critic_opts):

            def critic_loss_fn(c_model):
                q_B = c_model(mb.obs, mb.action)
                return optax.l2_loss(q_B, target_B).mean()

            loss, grads = nnx.value_and_grad(critic_loss_fn)(critic_model)
            critic_opt.update(critic_model, grads)
            total_critic_loss = total_critic_loss + loss
        return total_critic_loss / NUM_CRITICS

    # --- Update alpha ---
    def update_alpha(carry, logprob):
        def alpha_loss_fn(log_alpha_val):
            alpha = jnp.exp(log_alpha_val)
            loss = -alpha * (logprob + target_entropy)
            return loss.mean()

        _loss, grads = jax.value_and_grad(alpha_loss_fn)(carry.log_alpha)
        updates, new_alpha_opt_state = alpha_tx.update(grads, carry.alpha_opt_state, carry.log_alpha)
        new_log_alpha = optax.apply_updates(carry.log_alpha, updates)
        return carry._replace(log_alpha=new_log_alpha, alpha_opt_state=new_alpha_opt_state)

    # --- Train iteration ---
    def train_iteration(carry, actor, actor_optimizer, critics, critic_opts, critic_targets, rngs):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        carry, batch = collect_transitions(carry, actor, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        zero_update_metrics = {
            "actor_loss": jnp.float32(0.0),
            "critic_loss": jnp.float32(0.0),
            "entropy": jnp.float32(0.0),
        }

        def update_iteration(_, state):
            carry, act, act_opt, crs, c_opts, c_tgts, rngs, metrics_sum = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)

            logprob, actor_loss, entropy = update_actor(carry, minibatch, act, act_opt, crs, rngs)
            critic_loss = update_critics(carry, minibatch, act, crs, c_opts, c_tgts, rngs)
            carry = update_alpha(carry, logprob)
            step_metrics = {"actor_loss": actor_loss, "critic_loss": critic_loss, "entropy": entropy}
            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, step_metrics)
            return (carry, act, act_opt, crs, c_opts, c_tgts, rngs, metrics_sum)

        def do_updates(carry, act, act_opt, crs, c_opts, c_tgts, rngs):
            carry, act, act_opt, crs, c_opts, c_tgts, rngs, metrics_sum = nnx.fori_loop(
                0,
                config.num_epochs,
                update_iteration,
                (carry, act, act_opt, crs, c_opts, c_tgts, rngs, zero_update_metrics),
            )
            iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry, act, act_opt, crs, c_opts, c_tgts, rngs, iter_metrics

        def no_updates(carry, act, act_opt, crs, c_opts, c_tgts, rngs):
            return carry, act, act_opt, crs, c_opts, c_tgts, rngs, zero_update_metrics

        carry, actor, actor_optimizer, critics, critic_opts, critic_targets, rngs, update_metrics = nnx.cond(
            start_training,
            do_updates,
            no_updates,
            carry,
            actor,
            actor_optimizer,
            critics,
            critic_opts,
            critic_targets,
            rngs,
        )

        # Update target networks
        critic_targets = maybe_update_targets(
            critics, critic_targets, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )

        # Accumulate metrics (only when training)
        iter_metrics = {**update_metrics, "alpha": jnp.exp(carry.log_alpha)}
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, iter_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, actor, actor_optimizer, critics, critic_opts, critic_targets, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), eval_env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, actor, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _SAC_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, act, act_opt, crs, c_opts, c_tgts, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, act, act_opt, crs, c_opts, c_tgts, rngs = nnx.fori_loop(
            0, steps_per_iter, train_body, (carry, act, act_opt, crs, c_opts, c_tgts, rngs)
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, act, rngs)
        else:
            eval_lengths = jnp.zeros(128)
            eval_returns = jnp.zeros(128)

        metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": train_return,
            "global_step": jnp.int32(carry.global_step),
            **avg_tm,
        }

        if log_fn is not None:
            jax.debug.callback(log_fn, metrics)

        # Reset accumulator (use traced ops to preserve nnx.scan carry references)
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda x: x * 0, carry.train_metrics),
            metrics_count=carry.metrics_count * 0,
            completed_returns_sum=carry.completed_returns_sum * 0,
            completed_returns_count=carry.completed_returns_count * 0,
        )

        return (carry, act, act_opt, crs, c_opts, c_tgts, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor, actor_optimizer, critics, critic_opts, critic_targets, rngs), all_metrics = scan_eval(
        (carry, actor, actor_optimizer, critics, critic_opts, critic_targets, rngs), jnp.zeros(num_evals)
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critics": critics,
        "critic_optimizers": critic_opts,
        "critic_targets": critic_targets,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "log_alpha": carry.log_alpha,
        "global_step": carry.global_step,
    }
    return state, all_metrics
