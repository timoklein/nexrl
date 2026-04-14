"""V-MPO (On-Policy Maximum a-Posteriori Policy Optimization) — continuous action spaces.

On-policy with GAE advantages. Uses a two-step EM procedure: E-step selects
top-k advantages to compute non-parametric weights, M-step fits the parametric
actor via weighted maximum likelihood with decomposed KL trust-region constraints
(separate constraints for mean and std) enforced by Lagrange dual variables.
Data is collected using a target actor that is periodically hard-copied from
the online actor.

Reference: Song et al., "V-MPO: On-Policy Maximum a Posteriori Policy Optimization
for Discrete and Continuous Control" (2020).

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    A: action dimension
    T: trajectory length (num_steps)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import jax
import numpy as np
import optax
from flax import nnx, struct
from jax import numpy as jnp

from nexrl.algos.utils import (
    RewardRMSState,
    RMSState,
    calculate_gae,
    create_encoder,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    resolve_network_kwargs,
    shuffle_and_split,
    update_and_normalize_obs,
    update_and_normalize_rew,
)
from nexrl.configs import VMPOConfig
from nexrl.evaluate import evaluate
from nexrl.networks import GaussianPolicy, VNetwork
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


class Trajectory(struct.PyTreeNode):
    """Container for trajectory data collected during rollout."""

    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    done: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    """Container for minibatch data with computed advantages."""

    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


_METRIC_NAMES = ("actor_loss", "critic_loss", "temperature", "alpha_mean", "alpha_stddev", "entropy")


class VMPOContinuousCarry(NamedTuple):
    """Non-module carry state for continuous V-MPO training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    obs_rms: RMSState
    rew_rms: RewardRMSState
    dual_params: dict
    dual_opt_state: Any
    epoch_count: int
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _create_networks(
    config: VMPOConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[GaussianPolicy, nnx.Optimizer, GaussianPolicy, VNetwork, nnx.Optimizer]:
    """Create continuous actor (online + target), critic, and optimizers."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    actor_kwargs = resolve_network_kwargs(config.actor_kwargs)
    critic_kwargs = resolve_network_kwargs(config.critic_kwargs)

    rngs = nnx.Rngs(rng)

    actor_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    target_actor_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    critic_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None

    actor = GaussianPolicy(
        in_features=in_features, action_dim=action_dim, action_range=action_range, **actor_kwargs, rngs=rngs, encoder=actor_enc
    )
    target_actor = GaussianPolicy(
        in_features=in_features,
        action_dim=action_dim,
        action_range=action_range,
        **actor_kwargs,
        rngs=rngs,
        encoder=target_actor_enc,
    )
    nnx.update(target_actor, nnx.state(actor, nnx.Param))

    actor_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate=config.learning_rate))
    actor_opt = nnx.Optimizer(actor, actor_tx, wrt=nnx.Param)

    critic = VNetwork(in_features=in_features, **critic_kwargs, rngs=rngs, encoder=critic_enc)
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate=config.critic_learning_rate)
    )
    critic_opt = nnx.Optimizer(critic, critic_tx, wrt=nnx.Param)

    return actor, actor_opt, target_actor, critic, critic_opt


def train_vmpo_continuous(
    config: VMPOConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a continuous V-MPO agent.

    On-policy V-MPO with decomposed KL constraints for continuous actions.

    Args:
        config: VMPOConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, VMPOConfig

        config = VMPOConfig.from_yaml("configs/vmpo_continuous/gymnax/pendulum.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("vmpo_continuous"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    num_steps = config.num_steps
    action_dim = int(np.prod(env.action_space(env_params).shape))
    action_space = env.action_space(env_params)
    action_low = action_space.low
    action_high = action_space.high
    max_steps = env_params.max_steps_in_episode

    iteration_size = num_envs * num_steps
    assert iteration_size % config.num_minibatches == 0

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    rngs = nnx.Rngs(rng)
    actor, actor_opt, target_actor, critic, critic_opt = _create_networks(config, env, env_params, rngs())

    # Dual params: per-dimension alpha for mean and std when per_dim_constraining
    alpha_shape = (action_dim,) if config.per_dim_constraining else ()
    dual_params: dict[str, jax.Array] = {
        "log_temperature": jnp.array(config.init_log_temperature),
        "log_alpha_mean": jnp.full(alpha_shape, config.init_log_alpha_mean),
        "log_alpha_stddev": jnp.full(alpha_shape, config.init_log_alpha_stddev),
    }
    dual_tx = optax.adam(config.dual_learning_rate)
    dual_opt_state = dual_tx.init(dual_params)

    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)
    obs_rms = RMSState.create(env.observation_space(env_params).shape)
    rew_rms = RewardRMSState.create(num_envs)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _METRIC_NAMES}

    carry = VMPOContinuousCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        dual_params=dual_params,
        dual_opt_state=dual_opt_state,
        epoch_count=0,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect trajectories using target actor (read-only) ---
    def collect_trajectories(carry, target_actor, rngs):
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_steps, rng_action = jax.random.split(step_key, 2)
            rng_steps = jax.random.split(rng_steps, num_envs)

            # Use target actor for data collection (V-MPO key difference from PPO)
            action_BA, log_prob_B, _ = target_actor(carry.last_obs, rng_action)
            env_action = jnp.clip(action_BA, action_low, action_high)

            next_obs_BD, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, env_action, env_params)

            new_running_B = carry.running_return + reward_B
            completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * done_B)
            completed_count = carry.completed_returns_count + jnp.sum(done_B.astype(jnp.int32))
            reset_running_B = new_running_B * (1 - done_B)

            obs_rms = carry.obs_rms
            rew_rms = carry.rew_rms
            if config.normalize_observations:
                obs_rms, next_obs_BD = update_and_normalize_obs(obs_rms, next_obs_BD)
            if config.normalize_rewards:
                rew_rms, reward_B = update_and_normalize_rew(rew_rms, reward_B, done_B, config.gamma)

            # Store raw (unclipped) action for correct log_prob in update
            transition = Trajectory(carry.last_obs, action_BA, log_prob_B, reward_B, done_B)
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs_BD,
                last_done=done_B,
                global_step=carry.global_step + num_envs,
                obs_rms=obs_rms,
                rew_rms=rew_rms,
                running_return=reset_running_B,
                completed_returns_sum=completed_sum,
                completed_returns_count=completed_count,
            )
            return carry, transition

        carry, trajectories = jax.lax.scan(env_step, carry, step_keys)
        return carry, trajectories

    # --- V-MPO update (actor + dual + critic) per minibatch ---
    def update_vmpo(carry, batch, actor, actor_opt, target_actor, critic, critic_opt, rngs):
        obs_flat = batch.trajectories.obs  # (MB, D)
        action_flat = batch.trajectories.action  # (MB, A)
        advantages_flat = batch.advantages  # (MB,)
        targets_flat = batch.targets  # (MB,)
        dual_params = carry.dual_params

        # --- E-step: compute non-parametric weights from top-k advantages ---
        temperature = jax.nn.softplus(dual_params["log_temperature"]) + 1e-8
        scaled_adv = advantages_flat / temperature

        # Top-k selection
        num_samples = scaled_adv.shape[0]
        k = jnp.maximum(jnp.int32(config.top_k_fraction * num_samples), 1)
        sorted_adv = jnp.sort(scaled_adv)[::-1]
        top_k_threshold = sorted_adv[k - 1]
        mask = (scaled_adv >= top_k_threshold).astype(jnp.float32)

        # Numerically stable softmax over selected advantages
        max_adv = jnp.max(scaled_adv)
        unnorm_weights = mask * jnp.exp(scaled_adv - max_adv)
        sum_weights = jnp.sum(unnorm_weights) + 1e-10
        num_selected = jnp.sum(mask) + 1e-10
        weights = jax.lax.stop_gradient(unnorm_weights / sum_weights)

        # --- M-step: actor loss with decomposed KL ---
        target_mean_BA, target_std_BA = jax.lax.stop_gradient(target_actor.get_mean_std(obs_flat))

        def actor_loss_fn(actor_model):
            log_prob_B, entropy_B = actor_model.log_prob_entropy(obs_flat, action_flat)

            # Weighted negative log-likelihood
            policy_loss = -jnp.sum(weights * log_prob_B)

            # Decomposed KL constraints
            online_mean_BA, online_std_BA = actor_model.get_mean_std(obs_flat)

            # Per-dim KL for mean (fixed std = target_std)
            kl_mean_BA = 0.5 * (target_mean_BA - online_mean_BA) ** 2 / (target_std_BA**2 + 1e-8)
            # Per-dim KL for std (fixed mean = target_mean)
            kl_std_BA = jnp.log(online_std_BA / target_std_BA) + (target_std_BA**2) / (2 * online_std_BA**2 + 1e-8) - 0.5

            alpha_mean = jax.nn.softplus(dual_params["log_alpha_mean"]) + 1e-8
            alpha_stddev = jax.nn.softplus(dual_params["log_alpha_stddev"]) + 1e-8

            loss_kl_mean = jnp.sum(jax.lax.stop_gradient(alpha_mean) * jnp.mean(kl_mean_BA, axis=0))
            loss_kl_std = jnp.sum(jax.lax.stop_gradient(alpha_stddev) * jnp.mean(kl_std_BA, axis=0))

            total_loss = policy_loss + loss_kl_mean + loss_kl_std
            return total_loss, (jnp.mean(kl_mean_BA), jnp.mean(kl_std_BA), jnp.mean(entropy_B))

        (actor_loss, (_mean_kl_mean, _mean_kl_std, mean_entropy)), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(
            actor
        )
        actor_opt.update(actor, grads)

        # --- Critic loss ---
        def critic_loss_fn(critic_model):
            value_B = critic_model(obs_flat)
            return config.vf_coef * optax.l2_loss(value_B, jax.lax.stop_gradient(targets_flat)).mean()

        critic_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(critic)
        critic_opt.update(critic, critic_grads)

        # --- Dual loss ---
        # Recompute KL stats with stop_gradient for dual update
        online_mean_BA_sg, online_std_BA_sg = jax.lax.stop_gradient(actor.get_mean_std(obs_flat))
        kl_mean_BA = 0.5 * (target_mean_BA - online_mean_BA_sg) ** 2 / (target_std_BA**2 + 1e-8)
        kl_std_BA = jnp.log(online_std_BA_sg / target_std_BA) + (target_std_BA**2) / (2 * online_std_BA_sg**2 + 1e-8) - 0.5

        def dual_loss_fn(dp):
            temp = jax.nn.softplus(dp["log_temperature"]) + 1e-8
            # Temperature loss
            scaled = advantages_flat / temp
            max_s = jnp.max(scaled)
            unnorm_w = mask * jnp.exp(scaled - max_s)
            sw = jnp.sum(unnorm_w) + 1e-10
            log_mean_weights = jnp.log(sw) + max_s - jnp.log(num_selected)
            t_loss = temp * (config.epsilon + log_mean_weights)

            # Alpha losses for mean and std
            a_mean = jax.nn.softplus(dp["log_alpha_mean"]) + 1e-8
            a_loss_mean = jnp.sum(a_mean * (config.epsilon_mean - jax.lax.stop_gradient(jnp.mean(kl_mean_BA, axis=0))))

            a_std = jax.nn.softplus(dp["log_alpha_stddev"]) + 1e-8
            a_loss_std = jnp.sum(a_std * (config.epsilon_stddev - jax.lax.stop_gradient(jnp.mean(kl_std_BA, axis=0))))

            return t_loss + a_loss_mean + a_loss_std

        _dual_loss, dual_grads = jax.value_and_grad(dual_loss_fn)(dual_params)
        dual_updates, new_dual_opt_state = dual_tx.update(dual_grads, carry.dual_opt_state, dual_params)
        new_dual_params = optax.apply_updates(dual_params, dual_updates)
        new_dual_params = jax.tree.map(lambda x: jnp.maximum(x, -18.0), new_dual_params)

        carry = carry._replace(dual_params=new_dual_params, dual_opt_state=new_dual_opt_state)
        return carry, actor_loss, critic_loss, temperature, mean_entropy

    # --- Train iteration ---
    def train_iteration(carry, actor, actor_opt, target_actor, critic, critic_opt, rngs):
        carry, trajectories = collect_trajectories(carry, target_actor, rngs)

        # Store last_obs/last_done for GAE bootstrap (these don't change across epochs)
        last_obs_for_gae = carry.last_obs
        last_done_for_gae = carry.last_done

        update_metric_names = ("actor_loss", "critic_loss", "temperature", "entropy")
        zero_update_metrics = {k: jnp.float32(0.0) for k in update_metric_names}

        def update_epoch(_, state):
            carry, act, act_opt, t_act, crit, crit_opt, rngs, metrics_sum = state

            # Recompute advantages with latest critic each epoch (matches reference)
            values_TB = jax.vmap(crit)(trajectories.obs)  # (T, B)
            last_val_B = crit(last_obs_for_gae)
            last_val_B = jnp.where(last_done_for_gae, 0, last_val_B)
            advantages, targets = calculate_gae(
                trajectories.reward,
                values_TB,
                trajectories.done,
                last_val_B,
                config.gamma,
                config.gae_lambda,
            )

            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = shuffle_and_split(batch, rngs(), iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def update_step(state, mb):
                carry, act, act_opt, t_act, crit, crit_opt, rngs = state
                carry, actor_loss, critic_loss, temperature, entropy = update_vmpo(
                    carry, mb, act, act_opt, t_act, crit, crit_opt, rngs
                )
                return (carry, act, act_opt, t_act, crit, crit_opt, rngs), {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "temperature": temperature,
                    "entropy": entropy,
                }

            (carry, act, act_opt, t_act, crit, crit_opt, rngs), mb_metrics = update_step(
                (carry, act, act_opt, t_act, crit, crit_opt, rngs), minibatches
            )
            epoch_metrics = jax.tree.map(jnp.mean, mb_metrics)
            metrics_sum = jax.tree.map(lambda s, e: s + e, metrics_sum, epoch_metrics)

            # Periodic target actor hard copy (per epoch)
            new_epoch_count = carry.epoch_count + 1

            def _do_copy(t_act):
                nnx.update(t_act, nnx.state(act, nnx.Param))
                return t_act

            def _no_copy(t_act):
                return t_act

            t_act = nnx.cond(
                new_epoch_count % config.actor_target_period == 0,
                _do_copy,
                _no_copy,
                t_act,
            )
            carry = carry._replace(epoch_count=new_epoch_count)
            return (carry, act, act_opt, t_act, crit, crit_opt, rngs, metrics_sum)

        carry, actor, actor_opt, target_actor, critic, critic_opt, rngs, epoch_metrics_sum = nnx.fori_loop(
            0,
            config.num_epochs,
            update_epoch,
            (carry, actor, actor_opt, target_actor, critic, critic_opt, rngs, zero_update_metrics),
        )

        # Average over epochs, add alpha metrics
        iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, epoch_metrics_sum)
        alpha_mean = jax.nn.softplus(carry.dual_params["log_alpha_mean"]) + 1e-8
        alpha_stddev = jax.nn.softplus(carry.dual_params["log_alpha_stddev"]) + 1e-8
        iter_metrics["alpha_mean"] = jnp.mean(alpha_mean)
        iter_metrics["alpha_stddev"] = jnp.mean(alpha_stddev)

        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, iter_metrics),
            metrics_count=carry.metrics_count + 1,
        )

        return carry, actor, actor_opt, target_actor, critic, critic_opt, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, actor, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, act, act_opt, t_act, crit, crit_opt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, act, act_opt, t_act, crit, crit_opt, rngs = nnx.fori_loop(
            0, num_iters_per_eval, train_body, (carry, act, act_opt, t_act, crit, crit_opt, rngs)
        )

        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)
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

        carry = carry._replace(
            train_metrics=jax.tree.map(lambda x: x * 0, carry.train_metrics),
            metrics_count=carry.metrics_count * 0,
            completed_returns_sum=carry.completed_returns_sum * 0,
            completed_returns_count=carry.completed_returns_count * 0,
        )

        return (carry, act, act_opt, t_act, crit, crit_opt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor, actor_opt, target_actor, critic, critic_opt, rngs), all_metrics = scan_eval(
        (carry, actor, actor_opt, target_actor, critic, critic_opt, rngs), jnp.zeros(num_evals)
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "actor": actor,
        "actor_optimizer": actor_opt,
        "target_actor": target_actor,
        "critic": critic,
        "critic_optimizer": critic_opt,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "dual_params": carry.dual_params,
        "global_step": carry.global_step,
    }
    return state, all_metrics
