"""FastTD3 — standalone training function.

"FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control" (Seo et al., 2025),
arXiv:2505.22642. Port of RL-X's fully jitted JAX FastTD3 (nico-bohlinger/RL-X,
rl_x/algorithms/fasttd3/flax_full_jit).

TD3 tuned for massively parallel simulation: a deterministic policy with per-env exploration
noise scales resampled at episode ends, a twin distributional (C51) critic whose target is the
projection of the critic with the lower expected value, target policy smoothing with the current
policy (no target policy), AdamW for actor and critic, several critic updates per vectorized env
step on large batches, observation normalization updated from the replay batch, and
bootstrapping through time-limit truncations.

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
from nexrl.configs import FastTD3Config
from nexrl.evaluate import evaluate
from nexrl.networks import FastTD3Critic, FastTD3Policy
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


NUM_CRITICS = 2

_FAST_TD3_METRIC_NAMES = ("actor_loss", "critic_loss", "q_max", "q_min")


class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array  # unclipped policy action + exploration noise, in policy space
    reward: chex.Array
    terminated: chex.Array  # physical termination only; truncated steps bootstrap
    next_obs: chex.Array  # pre-reset observation


class FastTD3Carry(NamedTuple):
    """Non-module carry state for FastTD3 training."""

    env_state: Any
    last_obs: chex.Array
    global_step: int
    replay_buffer: CircularBuffer
    obs_rms: RMSState
    noise_scales: chex.Array  # (B, 1) per-env exploration noise std
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def train_fast_td3(
    config: FastTD3Config,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a FastTD3 agent.

    Args:
        config: FastTD3Config with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, FastTD3Config

        config = FastTD3Config(env="Pendulum-v1", obs_key=None, num_envs=128)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("fast_td3"), in_axes=(None, 0)),
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
    action_low = jnp.asarray(action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(action_space.high, dtype=jnp.float32)
    max_steps = env_params.max_steps_in_episode
    num_critic_updates = config.num_critic_updates_per_policy_update * config.num_policy_updates_per_step

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    def to_env_action(action):
        """Clip a policy-space action to [-1, 1] and rescale it to the env's action range."""
        return action_low + 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0) * (action_high - action_low)

    def sample_noise_scales(rng):
        return jax.random.uniform(rng, (num_envs, 1), minval=config.noise_std_min, maxval=config.noise_std_max)

    # Create networks
    rngs = nnx.Rngs(rng)
    actor = FastTD3Policy(
        in_features=in_features,
        action_dim=action_dim,
        hidden_layer_sizes=config.actor_hidden_layer_sizes,
        rngs=rngs,
    )
    actor_optimizer = nnx.Optimizer(actor, optax.adamw(config.learning_rate, weight_decay=config.weight_decay), wrt=nnx.Param)
    critic = FastTD3Critic(
        obs_dim=in_features,
        action_dim=action_dim,
        num_atoms=config.num_atoms,
        v_min=config.v_min,
        v_max=config.v_max,
        hidden_layer_sizes=config.critic_hidden_layer_sizes,
        rngs=rngs,
        num_critics=NUM_CRITICS,
    )
    critic_optimizer = nnx.Optimizer(
        critic, optax.adamw(config.learning_rate, weight_decay=config.weight_decay), wrt=nnx.Param
    )
    critic_target = nnx.clone(critic)

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

    zero_train_metrics = {k: jnp.float32(0.0) for k in _FAST_TD3_METRIC_NAMES}

    carry = FastTD3Carry(
        env_state=env_state,
        last_obs=obs,
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        noise_scales=sample_noise_scales(rngs()),  # (B, 1)
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect transitions ---
    def collect_transitions(carry, actor, rngs):
        last_obs_BD = carry.last_obs
        if config.normalize_observations:
            last_obs_BD = normalize_obs(carry.obs_rms, last_obs_BD)

        # Noisy deterministic policy, also during warm-up; the stored action is NOT clipped
        action_BA = actor(last_obs_BD)
        action_BA = action_BA + jax.random.normal(rngs(), action_BA.shape) * carry.noise_scales

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs_BD, new_env_state, reward_B, done_B, info = vmap_step(
            rng_steps, carry.env_state, to_env_action(action_BA), env_params
        )

        # Bootstrap through time-limit truncations. Adapters that report them also report the
        # pre-reset obs, which gymnax's autoreset overwrites; elsewhere every done is terminal.
        truncated_B = info.get("truncated", jnp.zeros_like(done_B))
        final_obs_BD = info.get("final_obs", next_obs_BD)
        terminated_B = jnp.logical_and(done_B, jnp.logical_not(truncated_B))

        # Resample the exploration noise std of every env whose episode ended
        noise_scales_B1 = jnp.where(done_B[:, None], sample_noise_scales(rngs()), carry.noise_scales)

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
            next_obs=final_obs_BD.astype(jnp.float32),
        )
        return carry._replace(
            last_obs=next_obs_BD,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            replay_buffer=carry.replay_buffer.append(transition),
            noise_scales=noise_scales_B1,
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

    # --- Update critic ---
    def update_critic(mb, actor, critic, critic_optimizer, critic_target, rngs):
        # Target policy smoothing with the current policy (no target policy)
        clipped_noise_BA = jnp.clip(
            jax.random.normal(rngs(), mb.action.shape) * config.smoothing_noise_std,
            -config.smoothing_noise_clip,
            config.smoothing_noise_clip,
        )
        next_action_BA = jnp.clip(actor(mb.next_obs) + clipped_noise_BA, -1.0, 1.0)

        # Distributional target on the shared support, projected for each target critic
        support_N = critic.support
        discount_B = config.gamma * (1.0 - mb.terminated)
        target_support_BN = jnp.clip(mb.reward[:, None] + discount_B[:, None] * support_N, config.v_min, config.v_max)
        target_probs_CBN = jax.nn.softmax(critic_target(mb.next_obs, next_action_BA), axis=-1)
        project_BN = jax.vmap(rlax.categorical_l2_project, in_axes=(0, 0, None))
        projected_CBN = jax.vmap(project_BN, in_axes=(None, 0, None))(target_support_BN, target_probs_CBN, support_N)
        projected_CBN = jnp.asarray(jax.lax.stop_gradient(projected_CBN))

        # Both critics learn from the projection with the lower expected value (ties pick the second critic)
        target_q_CB = (projected_CBN * support_N).sum(axis=-1)
        pick_first_B1 = jnp.expand_dims(target_q_CB[0] < target_q_CB[1], -1)
        target_dist_BN = jnp.where(pick_first_B1, projected_CBN[0], projected_CBN[1])

        def loss_fn(critic_model):
            logits_CBN = critic_model(mb.obs, mb.action)
            loss_CB = -(target_dist_BN[None] * jax.nn.log_softmax(logits_CBN, axis=-1)).sum(axis=-1)
            return loss_CB.sum(axis=0).mean()

        critic_loss, critic_grads = nnx.value_and_grad(loss_fn)(critic)
        critic_optimizer.update(critic, critic_grads)

        metrics = {
            "critic_loss": critic_loss,
            "q_max": target_q_CB.max(axis=0).mean(),
            "q_min": target_q_CB.min(axis=0).mean(),
        }
        return metrics

    # --- Update actor ---
    def update_actor(obs_BD, actor, actor_optimizer, critic):
        # The critic is an explicit argument (not a closure): its nnx.vmap'd ensemble cannot read
        # a module captured from an outer trace level. Only the actor (argnums=0) is differentiated.
        def actor_loss_fn(actor_model, critic_model):
            action_BA = actor_model(obs_BD)
            q_CB = critic_model.q_values(obs_BD, action_BA)
            return -jnp.min(q_CB, axis=0).mean()

        actor_loss, grads = nnx.value_and_grad(actor_loss_fn)(actor, critic)
        actor_optimizer.update(actor, grads)
        return actor_loss

    # --- Train iteration ---
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
                act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum = state

                def critic_iteration(c, critic_state):
                    cr, cr_opt, cr_tgt, rngs, metrics_sum = critic_state
                    u = p * config.num_critic_updates_per_policy_update + c
                    mb = jax.tree.map(lambda x: x[u], batch)
                    step_metrics = update_critic(mb, act, cr, cr_opt, cr_tgt, rngs)
                    _polyak_update(cr, cr_tgt, config.polyak)
                    metrics_sum = {**metrics_sum, **{k: metrics_sum[k] + step_metrics[k] for k in step_metrics}}
                    return cr, cr_opt, cr_tgt, rngs, metrics_sum

                cr, cr_opt, cr_tgt, rngs, metrics_sum = nnx.fori_loop(
                    0,
                    config.num_critic_updates_per_policy_update,
                    critic_iteration,
                    (cr, cr_opt, cr_tgt, rngs, metrics_sum),
                )
                # The policy step reuses the states of the last critic batch in this group
                last_u = (p + 1) * config.num_critic_updates_per_policy_update - 1
                actor_loss = update_actor(batch.obs[last_u], act, act_opt, cr)
                metrics_sum = {**metrics_sum, "actor_loss": metrics_sum["actor_loss"] + actor_loss}
                return act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum

            act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum = nnx.fori_loop(
                0,
                config.num_policy_updates_per_step,
                policy_iteration,
                (act, act_opt, cr, cr_opt, cr_tgt, rngs, zero_train_metrics),
            )
            iter_metrics = {
                k: v / (config.num_policy_updates_per_step if k == "actor_loss" else num_critic_updates)
                for k, v in metrics_sum.items()
            }
            return carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, iter_metrics

        def no_updates(carry, act, act_opt, cr, cr_opt, cr_tgt, rngs):
            return carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, zero_train_metrics

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
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, update_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, actor, actor_optimizer, critic, critic_optimizer, critic_target, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor, rngs):
        # The policy acts in [-1, 1]; clip and rescale to the env range exactly as in collection
        act_fn = make_eval_act(lambda obs, rng: to_env_action(actor.act(obs, rng)), config, carry.obs_rms)
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
            **{k: jnp.float32(0.0) for k in _FAST_TD3_METRIC_NAMES},
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
        "global_step": carry.global_step,
    }
    return state, all_metrics
