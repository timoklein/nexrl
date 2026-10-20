"""REPPO — standalone training function.

"Relative Entropy Pathwise Policy Optimization" (Voelcker, Brunnbauer et al., 2026), arXiv:2507.11019.
Port of RL-X's fully jitted JAX REPPO (nico-bohlinger/RL-X, rl_x/algorithms/reppo/flax_full_jit).

On-policy actor-critic that trains the policy with pathwise (reparameterized) gradients of a
learned Q-function instead of likelihood-ratio gradients. Each rollout stores soft rewards
(reward minus the discounted entropy bonus of the next action) and next-state values; the
HL-Gauss categorical critic regresses TD(lambda) targets built from them, with an auxiliary
next-feature and reward prediction loss. The policy maximizes Q plus entropy while the KL to the
pre-update policy stays below a bound; past the bound the loss switches to minimizing the KL.
Entropy and KL coefficients are learned Lagrange multipliers on the policy.

Dimension key:
    T: rollout steps (num_steps)
    B: batch (num_envs during collection, minibatch size during updates)
    D: observation dimension
    A: action dimension
    H: critic feature dimension (critic_hidden_dim)
    N: HL-Gauss bins (num_bins)
    K: KL samples (num_kl_samples)
"""

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

from nexrl.algos.utils import (
    RMSState,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    make_eval_env,
    normalize_obs,
    shuffle_and_split,
    update_rms,
)
from nexrl.configs import REPPOConfig
from nexrl.evaluate import evaluate
from nexrl.networks import REPPOCritic, REPPOPolicy
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_REPPO_METRIC_NAMES = (
    "actor_loss",
    "critic_loss",
    "aux_loss",
    "entropy",
    "entropy_coefficient",
    "kl",
    "kl_coefficient",
    "q_mean",
    "policy_q",
    "explained_variance",
)


class Transition(NamedTuple):
    obs: jax.Array  # normalized observation the policy acted on
    action: jax.Array  # policy-space action in (-1, 1)
    reward: jax.Array  # raw env reward
    soft_reward: jax.Array  # reward - gamma * alpha * log pi(a' | s')
    next_features: jax.Array  # critic features of (s', a'), target of the auxiliary head
    next_value: jax.Array  # critic value of (s', a')
    terminated: jax.Array  # physical termination only
    truncated: jax.Array  # time-limit truncation


class Minibatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    target: jax.Array  # TD(lambda) soft value target
    next_features: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class REPPOCarry(NamedTuple):
    """Non-module carry state for REPPO training."""

    env_state: Any
    last_obs: jax.Array  # raw (unnormalized) observation
    global_step: int
    obs_rms: RMSState
    train_metrics: dict
    metrics_count: int
    running_return: jax.Array
    completed_returns_sum: jax.Array
    completed_returns_count: jax.Array


def _optimizer(config: REPPOConfig) -> optax.GradientTransformation:
    return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))


def _td_lambda_targets(
    soft_reward_TB: jax.Array,
    next_value_TB: jax.Array,
    terminated_TB: jax.Array,
    truncated_TB: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Soft TD(lambda) targets, scanned backward from the last next-state value."""

    def td_lambda_step(lambda_return_B, step):
        soft_reward_B, next_value_B, terminated_B, truncated_B = step
        lambda_sum_B = gae_lambda * lambda_return_B + (1 - gae_lambda) * next_value_B
        # Truncated steps bootstrap from the next-state value alone
        delta_B = gamma * jnp.where(truncated_B > 0.5, next_value_B, (1.0 - terminated_B) * lambda_sum_B)
        lambda_return_B = soft_reward_B + delta_B
        return lambda_return_B, lambda_return_B

    _, target_TB = jax.lax.scan(
        td_lambda_step,
        next_value_TB[-1],
        (soft_reward_TB, next_value_TB, terminated_TB, truncated_TB),
        reverse=True,
    )
    return target_TB


def train_reppo(
    config: REPPOConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a REPPO agent.

    Args:
        config: REPPOConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, REPPOConfig

        config = REPPOConfig()
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("reppo"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()

    # --- Setup ---
    env, env_params = create_env(config, env, env_params)
    # Dedicated eval instances — sharing stateful Atari/DMC envs corrupts training (see make_eval_env).
    eval_env = make_eval_env(env, config.run_evaluation)

    num_envs = config.num_envs
    num_steps = config.num_steps
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    obs_shape = obs_space.shape
    in_features = int(np.prod(obs_shape))
    action_dim = int(np.prod(action_space.shape))
    action_low = jnp.asarray(action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(action_space.high, dtype=jnp.float32)
    max_steps = env_params.max_steps_in_episode
    target_entropy = action_dim * config.target_entropy_multiplier

    if config.policy_obs_dim is not None and config.policy_obs_dim >= in_features:
        raise ValueError(
            f"policy_obs_dim={config.policy_obs_dim} must be smaller than the observation dimension {in_features}. "
            "Set policy_obs_dim=None for envs without a privileged observation."
        )
    policy_in_features = in_features if config.policy_obs_dim is None else config.policy_obs_dim

    iteration_size = num_envs * num_steps
    if iteration_size % config.num_minibatches != 0:
        raise ValueError(
            f"num_envs * num_steps = {iteration_size} must be divisible by num_minibatches={config.num_minibatches}."
        )

    def to_env_action(action):
        # Policy actions live in [-1, 1]; the env sees them mapped to [low, high]
        return action_low + 0.5 * (jnp.clip(action, -1.0, 1.0) + 1.0) * (action_high - action_low)

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    policy = REPPOPolicy(
        policy_in_features,
        action_dim,
        config.actor_hidden_dim,
        rngs,
        policy_obs_dim=config.policy_obs_dim,
        min_std=config.policy_min_std,
        init_entropy_coefficient=config.init_entropy_coefficient,
        init_kl_coefficient=config.init_kl_coefficient,
    )
    # The entropy and KL coefficients are policy Params: one optimizer, one global-norm clip, as in the reference
    policy_optimizer = nnx.Optimizer(policy, _optimizer(config), wrt=nnx.Param)
    critic = REPPOCritic(
        in_features,
        action_dim,
        config.critic_hidden_dim,
        config.num_bins,
        config.v_min,
        config.v_max,
        rngs,
    )
    critic_optimizer = nnx.Optimizer(critic, _optimizer(config), wrt=nnx.Param)
    # Pre-update policy of each rollout, the anchor of the KL constraint
    old_policy = nnx.clone(policy)

    centers_N = critic.centers
    support_N1 = critic.support  # (N + 1,) bin edges
    sigma = critic.sigma

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization (float32 count so the carry dtype is stable across updates)
    obs_rms = RMSState.create(obs_shape).replace(count=jnp.float32(1e-4))

    zero_train_metrics = {k: jnp.float32(0.0) for k in _REPPO_METRIC_NAMES}

    carry = REPPOCarry(
        env_state=env_state,
        last_obs=obs,
        global_step=0,
        obs_rms=obs_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect rollout (modules read-only) ---
    def collect_rollout(carry, policy, critic, rngs):
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_steps, act_key, next_act_key = jax.random.split(step_key, 3)
            rng_steps = jax.random.split(rng_steps, num_envs)

            # The normalizer is updated from the raw obs before acting on it
            obs_rms = carry.obs_rms
            if config.normalize_observations:
                obs_rms = update_rms(obs_rms, carry.last_obs)
                obs_BD = normalize_obs(obs_rms, carry.last_obs)
            else:
                obs_BD = carry.last_obs

            loc_BA, log_std_BA = policy(obs_BD)
            action_BA, _ = policy.sample_and_log_prob(loc_BA, log_std_BA, act_key)

            next_obs_BD, new_env_state, reward_B, done_B, info = vmap_step(
                rng_steps, carry.env_state, to_env_action(action_BA), env_params
            )

            # Bootstrap from the pre-reset obs, which gymnax's autoreset overwrites.
            # Every compat adapter reports both keys.
            truncated_B = info["truncated"]
            terminated_B = jnp.logical_and(done_B, jnp.logical_not(truncated_B))
            final_obs_BD = info["final_obs"]
            if config.normalize_observations:
                final_obs_BD = normalize_obs(obs_rms, final_obs_BD)

            next_loc_BA, next_log_std_BA = policy(final_obs_BD)
            next_action_BA, next_log_prob_B = policy.sample_and_log_prob(next_loc_BA, next_log_std_BA, next_act_key)
            next_features_BH, next_logits_BN, _, _ = critic(final_obs_BD, next_action_BA)
            next_value_B = jax.nn.softmax(next_logits_BN, axis=-1) @ centers_N

            # Entropy bonus of the next action, not masked at terminals (reference behavior)
            alpha = jnp.exp(policy.log_entropy_coefficient[...]).squeeze()
            reward_B = reward_B.astype(jnp.float32)
            soft_reward_B = reward_B - config.gamma * next_log_prob_B * alpha

            # Track episode returns using raw rewards
            new_running_B = carry.running_return + reward_B
            completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * done_B)
            completed_count = carry.completed_returns_count + jnp.sum(done_B.astype(jnp.int32))
            reset_running_B = new_running_B * (1 - done_B)

            transition = Transition(
                obs=obs_BD.astype(jnp.float32),
                action=action_BA,
                reward=reward_B,
                soft_reward=soft_reward_B,
                next_features=next_features_BH,
                next_value=next_value_B,
                terminated=terminated_B.astype(jnp.float32),
                truncated=truncated_B.astype(jnp.float32),
            )
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs_BD,
                global_step=carry.global_step + num_envs,
                obs_rms=obs_rms,
                running_return=reset_running_B,
                completed_returns_sum=completed_sum,
                completed_returns_count=completed_count,
            )
            return carry, transition

        return jax.lax.scan(env_step, carry, step_keys)

    # --- Update critic ---
    def update_critic(mb, critic, critic_optimizer):
        # HL-Gauss target distribution: Gaussian (std sigma) mass of each bin around the clipped target
        target_B = jnp.clip(mb.target, config.v_min, config.v_max)
        cdf_BN1 = jax.scipy.special.erf((support_N1 - target_B[:, None]) / (math.sqrt(2.0) * sigma))
        target_dist_BN = (cdf_BN1[:, 1:] - cdf_BN1[:, :-1]) / (cdf_BN1[:, -1:] - cdf_BN1[:, :1])

        def loss_fn(critic_model):
            _, logits_BN, pred_features_BH, pred_reward_B1 = critic_model(mb.obs, mb.action)
            critic_loss_B = optax.softmax_cross_entropy(logits_BN, target_dist_BN)
            aux_loss_B = jnp.mean(
                jnp.concatenate(
                    [(pred_features_BH - mb.next_features) ** 2, (pred_reward_B1 - mb.reward[:, None]) ** 2], axis=-1
                ),
                axis=-1,
            )
            critic_loss_B = (1.0 - mb.truncated) * critic_loss_B
            aux_loss_B = (1.0 - mb.truncated) * (1.0 - mb.terminated) * aux_loss_B
            critic_loss = jnp.mean(critic_loss_B)
            aux_loss = jnp.mean(aux_loss_B)
            loss = critic_loss + config.aux_loss_coefficient * aux_loss

            value_B = jax.nn.softmax(logits_BN, axis=-1) @ centers_N
            metrics = {
                "critic_loss": critic_loss,
                "aux_loss": aux_loss,
                "q_mean": value_B.mean(),
                "explained_variance": 1 - jnp.var(mb.target - value_B) / (jnp.var(mb.target) + 1e-8),
            }
            return loss, metrics

        (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(critic)
        critic_optimizer.update(critic, grads)
        return metrics

    # --- Update policy and its entropy / KL coefficients ---
    def update_policy(obs_BD, policy, policy_optimizer, critic, old_policy, rngs):
        sample_key, kl_key = rngs(), rngs()

        # Critic and old policy are explicit module arguments; only the policy (argnums=0) is differentiated.
        def loss_fn(policy_model, critic_model, old_policy_model):
            loc_BA, log_std_BA = policy_model(obs_BD)
            new_action_BA, new_log_prob_B = policy_model.sample_and_log_prob(loc_BA, log_std_BA, sample_key)
            value_B = critic_model.value(obs_BD, new_action_BA)

            # Monte-Carlo KL(old || new) from num_kl_samples old-policy actions per state
            old_loc_BA, old_log_std_BA = old_policy_model(obs_BD)
            old_actions_KBA, old_log_probs_KB = old_policy_model.sample_and_log_prob(
                old_loc_BA, old_log_std_BA, kl_key, sample_shape=(config.num_kl_samples,)
            )
            new_log_probs_at_old_KB = policy_model.log_prob(loc_BA, log_std_BA, old_actions_KBA)
            kl_B = jnp.mean(old_log_probs_KB - new_log_probs_at_old_KB, axis=0)

            alpha = jnp.exp(policy_model.log_entropy_coefficient[...]).squeeze()
            beta = jnp.exp(policy_model.log_kl_coefficient[...]).squeeze()
            # Inside the KL bound: soft Q maximization; outside: pull the policy back toward the old one
            clipped_loss_B = jnp.where(
                kl_B < config.kl_bound,
                new_log_prob_B * jax.lax.stop_gradient(alpha) - value_B,
                kl_B * jax.lax.stop_gradient(beta),
            )
            entropy_B = -new_log_prob_B
            entropy_coefficient_loss_B = alpha * jax.lax.stop_gradient(target_entropy + entropy_B)
            kl_coefficient_loss_B = -beta * jax.lax.stop_gradient(kl_B - config.kl_bound)
            loss = jnp.mean(clipped_loss_B) + jnp.mean(entropy_coefficient_loss_B) + jnp.mean(kl_coefficient_loss_B)

            metrics = {
                "actor_loss": clipped_loss_B.mean(),
                "entropy": entropy_B.mean(),
                "entropy_coefficient": alpha,
                "kl": kl_B.mean(),
                "kl_coefficient": beta,
                "policy_q": value_B.mean(),
            }
            return loss, metrics

        (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy, critic, old_policy)
        policy_optimizer.update(policy, grads)
        return metrics

    # --- Train iteration: one rollout, then num_epochs passes over it ---
    def train_iteration(carry, policy, policy_optimizer, critic, critic_optimizer, old_policy, rngs):
        carry, rollout = collect_rollout(carry, policy, critic, rngs)
        target_TB = _td_lambda_targets(
            rollout.soft_reward,
            rollout.next_value,
            rollout.terminated,
            rollout.truncated,
            config.gamma,
            config.gae_lambda,
        )

        data = Minibatch(
            obs=rollout.obs,
            action=rollout.action,
            reward=rollout.reward,
            target=target_TB,
            next_features=rollout.next_features,
            terminated=rollout.terminated,
            truncated=rollout.truncated,
        )

        # Snapshot the pre-update policy for the KL constraint
        nnx.update(old_policy, nnx.state(policy, nnx.Param))

        def update_epoch(_, state):
            pol, pol_opt, cr, cr_opt, old_pol, rngs, metrics_sum = state
            minibatches = shuffle_and_split(data, rngs(), iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def update_step(state, mb):
                pol, pol_opt, cr, cr_opt, old_pol, rngs = state
                # Critic first; the policy step then uses the just-updated critic
                critic_metrics = update_critic(mb, cr, cr_opt)
                policy_metrics = update_policy(mb.obs, pol, pol_opt, cr, old_pol, rngs)
                return (pol, pol_opt, cr, cr_opt, old_pol, rngs), {**critic_metrics, **policy_metrics}

            (pol, pol_opt, cr, cr_opt, old_pol, rngs), mb_metrics = update_step(
                (pol, pol_opt, cr, cr_opt, old_pol, rngs), minibatches
            )
            metrics_sum = {k: metrics_sum[k] + jnp.mean(mb_metrics[k]) for k in metrics_sum}
            return pol, pol_opt, cr, cr_opt, old_pol, rngs, metrics_sum

        policy, policy_optimizer, critic, critic_optimizer, old_policy, rngs, epoch_metrics_sum = nnx.fori_loop(
            0,
            config.num_epochs,
            update_epoch,
            (policy, policy_optimizer, critic, critic_optimizer, old_policy, rngs, zero_train_metrics),
        )

        # Average over epochs and accumulate in the carry
        iter_metrics = {k: v / config.num_epochs for k, v in epoch_metrics_sum.items()}
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, iter_metrics),
            metrics_count=carry.metrics_count + 1,
        )
        return carry, policy, policy_optimizer, critic, critic_optimizer, old_policy, rngs

    # --- Eval callback ---
    def eval_callback(carry, policy, rngs):
        act_fn = make_eval_act(lambda obs, rng: to_env_action(policy.act(obs, rng)), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), eval_env, env_params, 128, max_steps)

    # --- Outer training loop ---
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_size))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, policy, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _REPPO_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, pol, pol_opt, cr, cr_opt, old_pol, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, pol, pol_opt, cr, cr_opt, old_pol, rngs = nnx.fori_loop(
            0, num_iters_per_eval, train_body, (carry, pol, pol_opt, cr, cr_opt, old_pol, rngs)
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, pol, rngs)
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

        return (carry, pol, pol_opt, cr, cr_opt, old_pol, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, policy, policy_optimizer, critic, critic_optimizer, old_policy, rngs), all_metrics = scan_eval(
        (carry, policy, policy_optimizer, critic, critic_optimizer, old_policy, rngs), jnp.zeros(num_evals)
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "actor": policy,
        "actor_optimizer": policy_optimizer,
        "critic": critic,
        "critic_optimizer": critic_optimizer,
        "obs_rms_state": carry.obs_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
