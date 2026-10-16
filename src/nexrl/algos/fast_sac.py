"""FastSAC — standalone training function.

"Learning Sim-to-Real Humanoid Locomotion in 15 Minutes" (Seo et al., 2025), arXiv:2512.01996.
Port of RL-X's fully jitted JAX FastSAC (nico-bohlinger/RL-X, rl_x/algorithms/fastsac/flax_full_jit).

SAC tuned for massively parallel simulation: LayerNorm+SiLU networks, a twin distributional
(C51) critic where each critic learns from its own target, AdamW for actor, critic and alpha,
several critic updates per vectorized env step on large batches, observation normalization
updated from the replay batch, and bootstrapping through time-limit truncations.

Dimension key:
    T: buffer rows per env (buffer_size_per_env)
    B: batch (num_envs during collection, batch_size during updates)
    D: observation dimension
    A: action dimension
    C: num critics (2)
    N: atoms of the categorical value distribution
    U: critic updates per env step (num_critic_updates_per_policy_update * num_policy_updates_per_step)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import jax
import numpy as np
import optax
import rlax
from flax import nnx
from jax import numpy as jnp

from nexrl.algos.utils import (
    RMSState,
    _polyak_update,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    make_eval_env,
    normalize_obs,
    update_rms,
)
from nexrl.buffers import CircularBuffer
from nexrl.configs import FastSACConfig
from nexrl.evaluate import evaluate
from nexrl.networks import FastSACCritic, FastSACPolicy
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


NUM_CRITICS = 2

_FAST_SAC_METRIC_NAMES = ("actor_loss", "critic_loss", "alpha_loss", "alpha", "entropy", "q_max", "q_min")
_CRITIC_METRIC_NAMES = ("critic_loss", "alpha_loss", "entropy", "q_max", "q_min")


class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    terminated: chex.Array  # physical termination only; truncated steps bootstrap
    next_obs: chex.Array  # pre-reset observation


class FastSACCarry(NamedTuple):
    """Non-module carry state for FastSAC training."""

    env_state: Any
    last_obs: chex.Array
    global_step: int
    replay_buffer: CircularBuffer
    obs_rms: RMSState
    log_alpha: chex.Array
    alpha_opt_state: Any
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _adamw(config: FastSACConfig) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=config.learning_rate,
        b1=config.adam_beta1,
        b2=config.adam_beta2,
        weight_decay=config.weight_decay,
    )


def train_fast_sac(
    config: FastSACConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a FastSAC agent.

    Args:
        config: FastSACConfig with hyperparameters and environment specification.
        rng: PRNG key for initialization and stochasticity. Use
            ``jax.vmap`` over different keys for independent seeds.
        env: Optional gymnax-compatible environment. Overrides ``config.env``
            when provided.
        env_params: Optional environment parameters. Uses defaults if ``None``.
        log_fn: Optional callback invoked at each evaluation point via
            ``jax.debug.callback``. Receives the metrics dict.

    Returns:
        A ``(state, metrics)`` tuple where ``state`` is a dict of trained
        parameters and ``metrics`` is a dict of JAX arrays with keys
        including ``eval_returns``, ``eval_lengths``, and ``global_step``.

    Example::

        from nexrl import get_train_fn, FastSACConfig

        config = FastSACConfig.from_yaml("configs/fast_sac/gymnax/pendulum.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("fast_sac"), in_axes=(None, 0)),
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
    obs_shape = obs_space.shape
    in_features = int(np.prod(obs_shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))
    max_steps = env_params.max_steps_in_episode
    num_critic_updates = config.num_critic_updates_per_policy_update * config.num_policy_updates_per_step

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    actor = FastSACPolicy(
        in_features=in_features,
        action_dim=action_dim,
        action_range=action_range,
        hidden_layer_sizes=config.actor_hidden_layer_sizes,
        log_std_range=(config.log_std_min, config.log_std_max),
        rngs=rngs,
    )
    actor_optimizer = nnx.Optimizer(actor, _adamw(config), wrt=nnx.Param)
    critic = FastSACCritic(
        obs_dim=in_features,
        action_dim=action_dim,
        num_atoms=config.num_atoms,
        v_min=config.v_min,
        v_max=config.v_max,
        hidden_layer_sizes=config.critic_hidden_layer_sizes,
        rngs=rngs,
        num_critics=NUM_CRITICS,
    )
    critic_optimizer = nnx.Optimizer(critic, _adamw(config), wrt=nnx.Param)
    critic_target = nnx.clone(critic)

    # Alpha: scalar log-temperature with its own AdamW (weight decay included, as in the reference)
    alpha_tx = _adamw(config)
    log_alpha = jnp.log(jnp.float32(config.initial_alpha))
    alpha_opt_state = alpha_tx.init(log_alpha)

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization (float32 count so the carry dtype is stable across updates)
    obs_rms = RMSState.create(obs_shape).replace(count=jnp.float32(1e-4))

    # Init replay buffer: one ring of buffer_size_per_env rows, each row holding all envs
    rows = config.buffer_size_per_env
    buf = CircularBuffer.empty(
        rows,
        Transition(
            obs=jnp.zeros((rows, num_envs, *obs_shape), dtype=jnp.float32),  # (T, B, D)
            action=jnp.zeros((rows, num_envs, action_dim), dtype=jnp.float32),  # (T, B, A)
            reward=jnp.zeros((rows, num_envs), dtype=jnp.float32),  # (T, B)
            terminated=jnp.zeros((rows, num_envs), dtype=jnp.float32),  # (T, B)
            next_obs=jnp.zeros((rows, num_envs, *obs_shape), dtype=jnp.float32),  # (T, B, D)
        ),
    )

    zero_train_metrics = {k: jnp.float32(0.0) for k in _FAST_SAC_METRIC_NAMES}

    carry = FastSACCarry(
        env_state=env_state,
        last_obs=obs,
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
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

        # Stochastic policy action, also during warm-up (the zero-init heads give mean 0)
        action_BA, _ = actor(last_obs, rngs())

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, reward_B, done_B, info = vmap_step(rng_steps, carry.env_state, action_BA, env_params)

        # Bootstrap through time-limit truncations. Adapters that report them also report the
        # pre-reset obs, which gymnax's autoreset overwrites; elsewhere every done is terminal.
        truncated_B = info.get("truncated", jnp.zeros_like(done_B))
        final_obs = info.get("final_obs", next_obs)
        terminated_B = jnp.logical_and(done_B, jnp.logical_not(truncated_B))

        # Track episode returns using raw rewards
        new_running_B = carry.running_return + reward_B
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * done_B)
        completed_count = carry.completed_returns_count + jnp.sum(done_B.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - done_B)

        transition = Transition(
            obs=carry.last_obs.astype(jnp.float32),
            action=action_BA,
            reward=reward_B.astype(jnp.float32),
            terminated=terminated_B.astype(jnp.float32),
            next_obs=final_obs.astype(jnp.float32),
        )
        return carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            replay_buffer=carry.replay_buffer.append(transition),
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )

    # --- Sample all batches of one env step ---
    def sample_batches(replay_buffer, rng):
        rng_t, rng_e = jax.random.split(rng)
        t_UB = jax.random.randint(rng_t, (num_critic_updates, config.batch_size), 0, replay_buffer.num_entries)
        e_UB = jax.random.randint(rng_e, (num_critic_updates, config.batch_size), 0, num_envs)
        return jax.tree.map(lambda x: x[t_UB, e_UB], replay_buffer.data)  # leaves (U, B, ...)

    # --- Update critic and alpha (one loss, one gradient call) ---
    def update_critic_and_alpha(carry, mb, actor, critic, critic_optimizer, critic_target, rngs):
        alpha = jnp.exp(carry.log_alpha)
        next_action_BA, next_log_prob_B = actor(mb.next_obs, rngs())

        # Soft distributional target: the entropy bonus shifts the reward, then project onto the support
        support_N = critic.support
        discount_B = config.gamma * (1.0 - mb.terminated)
        target_support_BN = (mb.reward - discount_B * alpha * next_log_prob_B)[:, None] + discount_B[:, None] * support_N
        target_support_BN = jnp.clip(target_support_BN, config.v_min, config.v_max)
        target_probs_CBN = jax.nn.softmax(critic_target(mb.next_obs, next_action_BA), axis=-1)
        project_BN = jax.vmap(rlax.categorical_l2_project, in_axes=(0, 0, None))
        # Each critic learns from the projection of its own target critic
        projected_CBN = jax.vmap(project_BN, in_axes=(None, 0, None))(target_support_BN, target_probs_CBN, support_N)
        projected_CBN = jax.lax.stop_gradient(projected_CBN)

        def loss_fn(critic_model, log_alpha):
            logits_CBN = critic_model(mb.obs, mb.action)
            critic_loss = -(projected_CBN * jax.nn.log_softmax(logits_CBN, axis=-1)).sum(axis=-1).sum(axis=0).mean()
            # Alpha loss on the next-state log-probs of this critic step, as in the reference
            alpha_loss = (jnp.exp(log_alpha) * (-next_log_prob_B - config.target_entropy)).mean()
            return critic_loss + alpha_loss, (critic_loss, alpha_loss)

        (_, (critic_loss, alpha_loss)), (critic_grads, log_alpha_grad) = nnx.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(critic, carry.log_alpha)
        critic_optimizer.update(critic, critic_grads)
        updates, alpha_opt_state = alpha_tx.update(log_alpha_grad, carry.alpha_opt_state, carry.log_alpha)
        carry = carry._replace(log_alpha=optax.apply_updates(carry.log_alpha, updates), alpha_opt_state=alpha_opt_state)

        target_q_CB = (projected_CBN * support_N).sum(axis=-1)
        metrics = {
            "critic_loss": critic_loss,
            "alpha_loss": alpha_loss,
            "entropy": -next_log_prob_B.mean(),
            "q_max": target_q_CB.max(axis=0).mean(),
            "q_min": target_q_CB.min(axis=0).mean(),
        }
        return carry, metrics

    # --- Update actor ---
    def update_actor(carry, obs_BD, actor, actor_optimizer, critic, rngs):
        alpha = jnp.exp(carry.log_alpha)
        action_rng = rngs()

        # The critic is an explicit argument (not a closure): its nnx.vmap'd ensemble cannot read
        # a module captured from an outer trace level. Only the actor (argnums=0) is differentiated.
        def actor_loss_fn(actor_model, critic_model):
            action_BA, log_prob_B = actor_model(obs_BD, action_rng)
            q_B = critic_model.q_values(obs_BD, action_BA).mean(axis=0)  # mean over critics
            return (alpha * log_prob_B - q_B).mean()

        actor_loss, grads = nnx.value_and_grad(actor_loss_fn)(actor, critic)
        actor_optimizer.update(actor, grads)
        return actor_loss

    # --- Train iteration ---
    zero_update_metrics = {k: jnp.float32(0.0) for k in _FAST_SAC_METRIC_NAMES if k != "alpha"}

    def train_iteration(carry, actor, actor_optimizer, critic, critic_optimizer, critic_target, rngs):
        carry = collect_transitions(carry, actor, rngs)
        # Warm-up: learning_starts vectorized steps fill the buffer before the first update
        start_training = carry.global_step > config.learning_starts * num_envs

        def do_updates(carry, act, act_opt, cr, cr_opt, cr_tgt, rngs):
            batch = sample_batches(carry.replay_buffer, rngs())
            if config.normalize_observations:
                # The normalizer is updated from the sampled obs and next_obs of all U batches
                obs_rms = update_rms(
                    carry.obs_rms,
                    jnp.concatenate([batch.obs.reshape(-1, *obs_shape), batch.next_obs.reshape(-1, *obs_shape)]),
                )
                batch = batch._replace(obs=normalize_obs(obs_rms, batch.obs), next_obs=normalize_obs(obs_rms, batch.next_obs))
                carry = carry._replace(obs_rms=obs_rms)

            def policy_iteration(p, state):
                carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum = state

                def critic_iteration(c, critic_state):
                    carry, cr, cr_opt, cr_tgt, rngs, metrics_sum = critic_state
                    u = p * config.num_critic_updates_per_policy_update + c
                    mb = jax.tree.map(lambda x: x[u], batch)
                    carry, step_metrics = update_critic_and_alpha(carry, mb, act, cr, cr_opt, cr_tgt, rngs)
                    _polyak_update(cr, cr_tgt, config.polyak)
                    metrics_sum = {**metrics_sum, **{k: metrics_sum[k] + step_metrics[k] for k in step_metrics}}
                    return carry, cr, cr_opt, cr_tgt, rngs, metrics_sum

                carry, cr, cr_opt, cr_tgt, rngs, metrics_sum = nnx.fori_loop(
                    0,
                    config.num_critic_updates_per_policy_update,
                    critic_iteration,
                    (carry, cr, cr_opt, cr_tgt, rngs, metrics_sum),
                )
                # The policy step reuses the states of the last critic batch in this group
                last_u = (p + 1) * config.num_critic_updates_per_policy_update - 1
                actor_loss = update_actor(carry, batch.obs[last_u], act, act_opt, cr, rngs)
                metrics_sum = {**metrics_sum, "actor_loss": metrics_sum["actor_loss"] + actor_loss}
                return carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum

            carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum = nnx.fori_loop(
                0,
                config.num_policy_updates_per_step,
                policy_iteration,
                (carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, zero_update_metrics),
            )
            iter_metrics = {
                k: v / (config.num_policy_updates_per_step if k == "actor_loss" else num_critic_updates)
                for k, v in metrics_sum.items()
            }
            return carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, iter_metrics

        def no_updates(carry, act, act_opt, cr, cr_opt, cr_tgt, rngs):
            return carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, zero_update_metrics

        carry, actor, actor_optimizer, critic, critic_optimizer, critic_target, rngs, update_metrics = nnx.cond(
            start_training,
            do_updates,
            no_updates,
            carry,
            actor,
            actor_optimizer,
            critic,
            critic_optimizer,
            critic_target,
            rngs,
        )

        # Accumulate metrics (only when training)
        iter_metrics = {**update_metrics, "alpha": jnp.exp(carry.log_alpha)}
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, iter_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, actor, actor_optimizer, critic, critic_optimizer, critic_target, rngs

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
            **{k: jnp.float32(0.0) for k in _FAST_SAC_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, act, act_opt, cr, cr_opt, cr_tgt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, act, act_opt, cr, cr_opt, cr_tgt, rngs = nnx.fori_loop(
            0, steps_per_iter, train_body, (carry, act, act_opt, cr, cr_opt, cr_tgt, rngs)
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

        return (carry, act, act_opt, cr, cr_opt, cr_tgt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor, actor_optimizer, critic, critic_optimizer, critic_target, rngs), all_metrics = scan_eval(
        (carry, actor, actor_optimizer, critic, critic_optimizer, critic_target, rngs), jnp.zeros(num_evals)
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
        "critic": critic,
        "critic_optimizer": critic_optimizer,
        "critic_target": critic_target,
        "obs_rms_state": carry.obs_rms,
        "log_alpha": carry.log_alpha,
        "global_step": carry.global_step,
    }
    return state, all_metrics
