"""FlashSAC — standalone training function.

"FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control"
(Kim, Lee et al., 2026). Port of RL-X's fully jitted JAX FlashSAC
(nico-bohlinger/RL-X, rl_x/algorithms/flashsac/flax_full_jit), default path only.

SAC for massively parallel simulation with residual BatchNorm networks whose weights are projected
back to unit norm after every gradient step. Policy and critic forward passes in train mode run on
concat(s, s') so BatchNorm sees both halves. The twin distributional (C51) critic is trained
against the target critic with the lower expected value. Rewards are scaled by a running return
statistic, and exploration noise is held for a zeta-distributed number of steps.

Dimension key:
    T: buffer rows per env (buffer_size // num_envs)
    B: batch (num_envs during collection, batch_size during updates)
    D: observation dimension
    A: action dimension
    C: num critics (2)
    N: atoms of the categorical value distribution
"""

import math
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
    RewardRMSState,
    _polyak_update,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    make_eval_env,
    update_rew_rms,
)
from nexrl.buffers import CircularBuffer
from nexrl.configs import FlashSACConfig
from nexrl.evaluate import evaluate
from nexrl.networks import FlashSACCritic, FlashSACPolicy, project_params
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


NUM_CRITICS = 2

_POLICY_METRIC_NAMES = ("actor_loss", "alpha_loss", "entropy", "policy_q")
_CRITIC_METRIC_NAMES = ("critic_loss", "target_q")
_FLASH_SAC_METRIC_NAMES = (*_POLICY_METRIC_NAMES, *_CRITIC_METRIC_NAMES, "alpha")


class Transition(NamedTuple):
    obs: chex.Array
    action: chex.Array  # policy action in [-1, 1], before rescaling to the env range
    reward: chex.Array  # raw reward; scaled at sample time
    terminated: chex.Array  # physical termination only; truncated steps bootstrap
    next_obs: chex.Array  # pre-reset observation


class FlashSACCarry(NamedTuple):
    """Non-module carry state for FlashSAC training."""

    env_state: Any
    last_obs: chex.Array
    global_step: int
    replay_buffer: CircularBuffer
    rew_rms: RewardRMSState
    g_max: chex.Array  # largest |discounted return| seen so far
    noise: chex.Array  # (num_envs, A) held exploration noise
    noise_count: chex.Array  # steps the current noise has been held
    noise_hold: chex.Array  # steps to hold the current noise
    update_count: chex.Array
    log_alpha: chex.Array
    alpha_opt_state: Any
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _zeta_cdf(mu: float, max_n: int) -> jax.Array:
    """CDF of a zeta(mu) distribution truncated to {1, ..., max_n}."""
    ns = jnp.arange(1, max_n + 1, dtype=jnp.float32)
    pmf = ns ** (-mu)
    return jnp.cumsum(pmf / jnp.sum(pmf))


def _step_noise(
    noise_BA: jax.Array, count: jax.Array, hold: jax.Array, rng: jax.Array, cdf: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Keep the noise for ``hold`` steps, then draw new noise and a new hold length."""
    noise_rng, hold_rng = jax.random.split(rng)
    new_noise_BA = jax.random.normal(noise_rng, noise_BA.shape, dtype=jnp.float32)
    u = jax.random.uniform(hold_rng, shape=())
    new_hold = jnp.argmax((u < cdf).astype(jnp.int32)) + 1

    reinit = (count == 0) | (count >= hold)
    noise_BA = jnp.where(reinit, new_noise_BA, noise_BA)
    hold = jnp.where(reinit, new_hold, hold)
    count = jnp.where(reinit, 0, count) + 1
    return noise_BA, count, hold


def train_flash_sac(
    config: FlashSACConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a FlashSAC agent.

    Args:
        config: FlashSACConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, FlashSACConfig

        config = FlashSACConfig.from_yaml("configs/flash_sac/gymnax/pendulum.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("flash_sac"), in_axes=(None, 0)),
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
    target_entropy = 0.5 * action_dim * math.log(2.0 * math.pi * math.e * config.target_entropy_sigma**2)
    noise_cdf = _zeta_cdf(config.noise_zeta_mu, config.noise_zeta_max)

    def to_env_action(action):
        # Policy actions live in [-1, 1]; the env sees them mapped to [low, high]
        return action_low + 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0) * (action_high - action_low)

    if config.policy_obs_dim is not None and config.policy_obs_dim >= in_features:
        raise ValueError(
            f"policy_obs_dim={config.policy_obs_dim} must be smaller than the observation dimension {in_features}. "
            "Set policy_obs_dim=None for envs without a privileged observation."
        )
    policy_in_features = in_features if config.policy_obs_dim is None else config.policy_obs_dim

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # One cosine decay for all three optimizers; each counts its own steps, so the policy and
    # alpha (updated every policy_delay critic steps) decay only part of the way, as in the reference.
    total_updates = max(1, (config.total_timesteps // num_envs) * config.updates_per_step)
    lr_schedule = optax.cosine_decay_schedule(
        config.learning_rate, total_updates, alpha=config.learning_rate_end / config.learning_rate
    )

    # Create networks, projected to unit norm before the first step
    rngs = nnx.Rngs(rng)
    actor = FlashSACPolicy(
        in_features=policy_in_features,
        action_dim=action_dim,
        hidden_dim=config.actor_hidden_dim,
        num_blocks=config.actor_num_blocks,
        rngs=rngs,
        policy_obs_dim=config.policy_obs_dim,
    )
    project_params(actor)
    actor_optimizer = nnx.Optimizer(actor, optax.adam(lr_schedule), wrt=nnx.Param)
    critic = FlashSACCritic(
        obs_dim=in_features,
        action_dim=action_dim,
        num_atoms=config.num_atoms,
        v_min=config.v_min,
        v_max=config.v_max,
        hidden_dim=config.critic_hidden_dim,
        num_blocks=config.critic_num_blocks,
        rngs=rngs,
        num_critics=NUM_CRITICS,
    )
    project_params(critic)
    critic_optimizer = nnx.Optimizer(critic, optax.adam(lr_schedule), wrt=nnx.Param)
    critic_target = nnx.clone(critic)

    # Alpha: scalar log-temperature with its own Adam
    alpha_tx = optax.adam(lr_schedule)
    log_alpha = jnp.log(jnp.float32(config.initial_alpha))
    alpha_opt_state = alpha_tx.init(log_alpha)

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init replay buffer: one ring of buffer_size // num_envs rows, each row holding all envs
    rows = config.buffer_size // num_envs
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

    # float32 scalars so the carry dtypes are stable across updates
    rew_rms = RewardRMSState(mean=jnp.float32(0.0), var=jnp.float32(1.0), count=jnp.float32(1e-4), returns=jnp.zeros(num_envs))

    zero_train_metrics = {k: jnp.float32(0.0) for k in _FLASH_SAC_METRIC_NAMES}

    carry = FlashSACCarry(
        env_state=env_state,
        last_obs=obs,
        global_step=0,
        replay_buffer=buf,
        rew_rms=rew_rms,
        g_max=jnp.float32(0.0),
        noise=jax.random.normal(rngs(), (num_envs, action_dim), dtype=jnp.float32),
        noise_count=jnp.int32(0),
        noise_hold=jnp.int32(1),
        update_count=jnp.int32(0),
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
        last_obs_BD = carry.last_obs.astype(jnp.float32)

        # Uniform random actions until learning_starts timesteps, then the policy with held noise.
        # The noise state advances only when the policy acts.
        use_random = carry.global_step < config.learning_starts
        random_action_BA = jax.random.uniform(rngs(), (num_envs, action_dim), minval=-1.0, maxval=1.0)
        noise_BA, noise_count, noise_hold = _step_noise(carry.noise, carry.noise_count, carry.noise_hold, rngs(), noise_cdf)
        policy_action_BA = actor.act_with_noise(last_obs_BD, noise_BA)
        action_BA = jnp.where(use_random, random_action_BA, policy_action_BA)
        noise_BA = jnp.where(use_random, carry.noise, noise_BA)
        noise_count = jnp.where(use_random, carry.noise_count, noise_count)
        noise_hold = jnp.where(use_random, carry.noise_hold, noise_hold)

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs_BD, new_env_state, reward_B, done_B, info = vmap_step(
            rng_steps, carry.env_state, to_env_action(action_BA), env_params
        )

        # Bootstrap through time-limit truncations from the pre-reset obs, which gymnax's
        # autoreset overwrites. Every compat adapter reports both keys.
        truncated_B = info["truncated"]
        final_obs_BD = info["final_obs"]
        terminated_B = jnp.logical_and(done_B, jnp.logical_not(truncated_B))

        # Reward scale statistics: running variance of the discounted return, and its largest magnitude
        rew_rms = carry.rew_rms
        g_max = carry.g_max
        if config.normalize_rewards:
            rew_rms = update_rew_rms(rew_rms, reward_B, done_B, config.gamma)
            g_max = jnp.maximum(g_max, jnp.max(jnp.abs(rew_rms.returns)))

        # Track episode returns using raw rewards
        new_running_B = carry.running_return + reward_B
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * done_B)
        completed_count = carry.completed_returns_count + jnp.sum(done_B.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - done_B)

        transition = Transition(
            obs=last_obs_BD,
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
            rew_rms=rew_rms,
            g_max=g_max,
            noise=noise_BA,
            noise_count=noise_count,
            noise_hold=noise_hold,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )

    # --- Sample one batch ---
    def sample_batch(carry, rng):
        rng_t, rng_e = jax.random.split(rng)
        t_B = jax.random.randint(rng_t, (config.batch_size,), 0, carry.replay_buffer.num_entries)
        e_B = jax.random.randint(rng_e, (config.batch_size,), 0, num_envs)
        mb = jax.tree.map(lambda x: x[t_B, e_B], carry.replay_buffer.data)
        if config.normalize_rewards:
            rms = carry.rew_rms
            scale = jnp.maximum(jnp.sqrt(rms.var + 1e-8), carry.g_max / config.normalized_g_max)
            mb = mb._replace(reward=mb.reward / scale)
        return mb

    # --- Update policy and alpha ---
    def update_policy_and_alpha(carry, mb, actor, actor_optimizer, critic, rng):
        alpha = jnp.exp(carry.log_alpha)
        batch_size = mb.obs.shape[0]

        # The critic is an explicit argument (not a closure): its nnx.vmap'd ensemble cannot read
        # a module captured from an outer trace level. Only the actor (argnums=0) is differentiated.
        def actor_loss_fn(actor_model, critic_model):
            # Train-mode BatchNorm sees concat(s, s'); only the s half enters the loss
            obs_all = jnp.concatenate([mb.obs, mb.next_obs], axis=0)  # (2B, D)
            action_all, log_prob_all = actor_model(obs_all, rng, train=True)
            action_BA, log_prob_B = action_all[:batch_size], log_prob_all[:batch_size]
            q_B = critic_model.q_values(mb.obs, action_BA, train=False).min(axis=0)
            loss = (alpha * log_prob_B - q_B).mean()
            return loss, (-log_prob_B.mean(), q_B.mean())

        (actor_loss, (entropy, policy_q)), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor, critic)
        actor_optimizer.update(actor, grads)
        project_params(actor)

        # Alpha moves the entropy of this policy step toward the target entropy
        def alpha_loss_fn(log_alpha):
            return jnp.exp(log_alpha) * (entropy - target_entropy)

        alpha_loss, log_alpha_grad = jax.value_and_grad(alpha_loss_fn)(carry.log_alpha)
        updates, alpha_opt_state = alpha_tx.update(log_alpha_grad, carry.alpha_opt_state, carry.log_alpha)
        carry = carry._replace(log_alpha=optax.apply_updates(carry.log_alpha, updates), alpha_opt_state=alpha_opt_state)

        metrics = {"actor_loss": actor_loss, "alpha_loss": alpha_loss, "entropy": entropy, "policy_q": policy_q}
        return carry, metrics

    # --- Update critic ---
    def update_critic(carry, mb, actor, critic, critic_optimizer, critic_target, rng):
        alpha = jnp.exp(carry.log_alpha)
        batch_size = mb.obs.shape[0]
        next_action_BA, next_log_prob_B = actor(mb.next_obs, rng, train=False)
        obs_all = jnp.concatenate([mb.obs, mb.next_obs], axis=0)  # (2B, D)
        action_all = jnp.concatenate([mb.action, next_action_BA], axis=0)  # (2B, A)
        support_N = critic.support

        # The target is a module argument so its train-mode BatchNorm statistics update propagates.
        # Only the online critic (argnums=0) is differentiated.
        def critic_loss_fn(critic_model, target_model):
            # Target: train-mode forward on concat(s, s'), keep the s' half
            next_probs_CBN = jax.nn.softmax(target_model(obs_all, action_all, train=True)[:, batch_size:], axis=-1)
            next_q_CB = (next_probs_CBN * support_N).sum(axis=-1)
            # Clipped double-Q: project the distribution of the critic with the lower expected value
            min_idx_B = jnp.argmin(next_q_CB, axis=0)
            next_probs_BN = jnp.take_along_axis(next_probs_CBN, min_idx_B[None, :, None], axis=0)[0]

            discount_B = config.gamma * (1.0 - mb.terminated)
            target_support_BN = mb.reward[:, None] + discount_B[:, None] * (
                support_N[None, :] - (alpha * next_log_prob_B)[:, None]
            )
            target_support_BN = jnp.clip(target_support_BN, config.v_min, config.v_max)
            projected_BN = jax.vmap(rlax.categorical_l2_project, in_axes=(0, 0, None))(
                target_support_BN, next_probs_BN, support_N
            )
            projected_BN = jnp.asarray(jax.lax.stop_gradient(projected_BN))

            # Online: train-mode forward on the same concat, keep the s half
            log_probs_CBN = jax.nn.log_softmax(critic_model(obs_all, action_all, train=True)[:, :batch_size], axis=-1)
            loss = -(projected_BN[None] * log_probs_CBN).sum(axis=-1).mean()
            return loss, next_q_CB.mean()

        (critic_loss, target_q), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(critic, critic_target)
        critic_optimizer.update(critic, grads)
        project_params(critic)
        return {"critic_loss": critic_loss, "target_q": target_q}

    # --- Train iteration ---
    zero_policy_metrics = {k: jnp.float32(0.0) for k in _POLICY_METRIC_NAMES}
    zero_update_metrics = {k: jnp.float32(0.0) for k in (*_POLICY_METRIC_NAMES, *_CRITIC_METRIC_NAMES)}

    def train_iteration(carry, actor, actor_optimizer, critic, critic_optimizer, critic_target, rngs):
        carry = collect_transitions(carry, actor, rngs)
        start_training = carry.global_step >= config.learning_starts

        def do_updates(carry, act, act_opt, cr, cr_opt, cr_tgt, rngs):
            def update_step(_, state):
                carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum, num_policy_updates = state
                mb = sample_batch(carry, rngs())
                policy_rng, critic_rng = rngs(), rngs()

                # Policy and alpha first (every policy_delay steps), then the critic, as in the reference
                def policy_branch(carry, act, act_opt, cr):
                    return update_policy_and_alpha(carry, mb, act, act_opt, cr, policy_rng)

                def skip_branch(carry, act, act_opt, cr):
                    return carry, zero_policy_metrics

                do_policy = carry.update_count % config.policy_delay == 0
                carry, policy_metrics = nnx.cond(do_policy, policy_branch, skip_branch, carry, act, act_opt, cr)
                critic_metrics = update_critic(carry, mb, act, cr, cr_opt, cr_tgt, critic_rng)
                _polyak_update(cr, cr_tgt, config.polyak)

                carry = carry._replace(update_count=carry.update_count + 1)
                step_metrics = {**policy_metrics, **critic_metrics}
                metrics_sum = {k: metrics_sum[k] + step_metrics[k] for k in metrics_sum}
                num_policy_updates = num_policy_updates + do_policy.astype(jnp.float32)
                return carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum, num_policy_updates

            carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, metrics_sum, num_policy_updates = nnx.fori_loop(
                0,
                config.updates_per_step,
                update_step,
                (carry, act, act_opt, cr, cr_opt, cr_tgt, rngs, zero_update_metrics, jnp.float32(0.0)),
            )
            iter_metrics = {
                k: v / (jnp.maximum(num_policy_updates, 1.0) if k in _POLICY_METRIC_NAMES else config.updates_per_step)
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
        act_fn = make_eval_act(lambda obs, rng: to_env_action(actor.act(obs, rng)), config)
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
            **{k: jnp.float32(0.0) for k in _FLASH_SAC_METRIC_NAMES},
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
        "log_alpha": carry.log_alpha,
        "global_step": carry.global_step,
    }
    return state, all_metrics
