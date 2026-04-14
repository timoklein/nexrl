"""Phasic Policy Gradient (PPG) — standalone training function.

"Phasic Policy Gradient" (Cobbe et al., 2021)

Separates policy and value training into distinct phases. During the policy
phase the critic receives stop_gradient(encoder_output) so value gradients
don't touch the shared encoder. During the auxiliary phase an auxiliary value
head trains the encoder while KL regularization prevents policy drift.
Uses improved value regularization from Moon et al. 2022.

Discrete action spaces only.

Memory note: the auxiliary buffer stores all observations/returns from
``n_policy_iters`` policy iterations on-device. For small obs (CartPole D=4)
this is negligible (~4 MB). For large obs (Craftax D≈12k) this can be
prohibitive (~12 GB at default settings). Reduce ``n_policy_iters`` or
``num_envs`` for large observation spaces.

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    T: trajectory length (num_steps)
    H: hidden layer size
    A: action dimension (num_actions)
    P: total aux buffer size (n_policy_iters * T * B flattened)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import distrax
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
from nexrl.configs import PPGConfig
from nexrl.evaluate import evaluate
from nexrl.networks import SharedMLPDiscreteActorCritic
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


class Trajectory(struct.PyTreeNode):
    """Container for trajectory data collected during rollout."""

    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    """Container for minibatch data with computed advantages."""

    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class AuxMinibatch(struct.PyTreeNode):
    """Minibatch for auxiliary phase: obs, returns, and frozen policy logits."""

    obs: chex.Array
    returns: chex.Array
    old_logits: chex.Array
    old_values: chex.Array


class PPGSystem(nnx.Module):
    """Container for PPG networks: shared AC + auxiliary critic head.

    A single ``nnx.Optimizer(system, tx, wrt=nnx.Param)`` manages all params.
    """

    def __init__(self, ac: SharedMLPDiscreteActorCritic, aux_critic: nnx.Linear):
        self.ac = ac
        self.aux_critic = aux_critic


class PPGCarry(NamedTuple):
    """Non-module carry state for PPG training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array
    # Auxiliary phase buffer
    aux_obs: chex.Array  # (n_policy_iters, T, B, D)
    aux_returns: chex.Array  # (n_policy_iters, T, B)
    aux_write_idx: int


def _create_networks(
    config: PPGConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[PPGSystem, nnx.Optimizer]:
    """Create PPG system (shared AC + aux critic) and optimizer."""
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    rngs = nnx.Rngs(rng)
    encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None

    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
    hidden_size = agent_kwargs["hidden_layer_sizes"][-1]

    ac = SharedMLPDiscreteActorCritic(
        in_features=in_features,
        action_dim=action_space.n,
        **agent_kwargs,
        rngs=rngs,
        cnn_encoder=encoder,
    )

    # Auxiliary critic head: maps hidden -> scalar value
    aux_critic = nnx.Linear(hidden_size, 1, rngs=rngs)

    system = PPGSystem(ac, aux_critic)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    opt = nnx.Optimizer(system, tx, wrt=nnx.Param)
    return system, opt


_PPG_METRIC_NAMES = (
    "actor_loss",
    "critic_loss",
    "entropy",
    "clip_fraction",
    "approx_kl",
    "explained_variance",
    "aux_value_loss",
    "aux_kl_loss",
    "aux_value_reg_loss",
)


def train_ppg(
    config: PPGConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a PPG agent.

    Phasic Policy Gradient with separate policy and auxiliary value phases.

    Args:
        config: PPGConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, PPGConfig

        config = PPGConfig.from_yaml("configs/ppg/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("ppg"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    num_steps = config.num_steps
    obs_space = env.observation_space(env_params)
    obs_shape = obs_space.shape
    obs_dim = int(np.prod(obs_shape))
    max_steps = env_params.max_steps_in_episode

    iteration_size = num_envs * num_steps
    assert iteration_size % config.num_minibatches == 0

    n_policy_iters = config.n_policy_iters

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    system, opt = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_shape)
    rew_rms = RewardRMSState.create(num_envs)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _PPG_METRIC_NAMES}

    # Auxiliary buffer: pre-allocated
    aux_obs = jnp.zeros((n_policy_iters, num_steps, num_envs, obs_dim))
    aux_returns = jnp.zeros((n_policy_iters, num_steps, num_envs))

    carry = PPGCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
        aux_obs=aux_obs,
        aux_returns=aux_returns,
        aux_write_idx=0,
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, system, rngs):
        """Roll out policy for num_steps. System is read-only."""
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_steps, rng_action = jax.random.split(step_key, 2)
            rng_steps = jax.random.split(rng_steps, num_envs)

            action_B, log_prob_B, value_B = system.ac.get_action_and_value(carry.last_obs, rng_action)

            next_obs_BD, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_B, env_params)

            # Track episode returns
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

            transition = Trajectory(carry.last_obs, action_B, log_prob_B, reward_B, value_B, done_B)
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

    # --- Policy phase update (PPO clipped) ---
    def update_ac(batch, system, opt):
        def loss_fn(system):
            log_prob_B, entropy_B, value_B = system.ac.get_pi_and_value(batch.trajectories.obs, batch.trajectories.action)
            entropy = entropy_B.mean()

            # Actor loss (PPO clipped)
            ratio_B = jnp.exp(log_prob_B - batch.trajectories.log_prob)
            adv_B = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            clipped_ratio_B = jnp.clip(ratio_B, 1 - config.clip_eps, 1 + config.clip_eps)
            pi_loss1_B = ratio_B * adv_B
            pi_loss2_B = clipped_ratio_B * adv_B
            actor_loss = -jnp.minimum(pi_loss1_B, pi_loss2_B).mean()

            # Critic loss (clipped)
            value_clipped_B = batch.trajectories.value + (value_B - batch.trajectories.value).clip(
                -config.clip_eps, config.clip_eps
            )
            loss_B = jnp.square(value_B - batch.targets)
            loss_clipped_B = jnp.square(value_clipped_B - batch.targets)
            critic_loss = 0.5 * jnp.maximum(loss_B, loss_clipped_B).mean()

            clip_fraction = jnp.mean(jnp.abs(ratio_B - 1) > config.clip_eps)
            approx_kl = jnp.mean((ratio_B - 1) - jnp.log(ratio_B))

            total_loss = actor_loss + config.vf_coef * critic_loss - config.ent_coef * entropy
            return total_loss, (actor_loss, critic_loss, entropy, clip_fraction, approx_kl)

        (_total, (actor_loss, critic_loss, entropy, clip_fraction, approx_kl)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(system)
        opt.update(system, grads)
        return actor_loss, critic_loss, entropy, clip_fraction, approx_kl

    # --- Single policy iteration (collect + PPO updates + store buffer) ---
    def policy_iteration(carry, system, opt, rngs):
        carry, trajectories = collect_trajectories(carry, system, rngs)

        last_val_B = system.ac.value(carry.last_obs)
        last_val_B = jnp.where(carry.last_done, 0, last_val_B)
        advantages, targets = calculate_gae(
            trajectories.reward,
            trajectories.value,
            trajectories.done,
            last_val_B,
            config.gamma,
            config.gae_lambda,
        )

        # Explained variance
        explained_var = 1 - jnp.var(targets - trajectories.value) / (jnp.var(targets) + 1e-8)

        update_metric_names = ("actor_loss", "critic_loss", "entropy", "clip_fraction", "approx_kl")
        zero_update_metrics = {k: jnp.float32(0.0) for k in update_metric_names}

        def update_epoch(_, state):
            system, opt, rngs, metrics_sum = state
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = shuffle_and_split(batch, rngs(), iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def update_step(state, mb):
                system, opt = state
                actor_loss, critic_loss, entropy, clip_fraction, approx_kl = update_ac(mb, system, opt)
                return (system, opt), {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "clip_fraction": clip_fraction,
                    "approx_kl": approx_kl,
                }

            (system, opt), mb_metrics = update_step((system, opt), minibatches)
            epoch_metrics = jax.tree.map(jnp.mean, mb_metrics)
            metrics_sum = jax.tree.map(lambda s, e: s + e, metrics_sum, epoch_metrics)
            return (system, opt, rngs, metrics_sum)

        system, opt, rngs, epoch_metrics_sum = nnx.fori_loop(
            0, config.num_epochs, update_epoch, (system, opt, rngs, zero_update_metrics)
        )

        iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, epoch_metrics_sum)
        iter_metrics["explained_variance"] = explained_var
        # Zero out aux keys so tree structure matches carry.train_metrics
        iter_metrics["aux_value_loss"] = jnp.float32(0.0)
        iter_metrics["aux_kl_loss"] = jnp.float32(0.0)
        iter_metrics["aux_value_reg_loss"] = jnp.float32(0.0)

        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, iter_metrics),
            metrics_count=carry.metrics_count + 1,
        )

        # Store obs and targets in aux buffer
        # trajectories.obs: (T, B, *obs_shape), targets: (T, B)
        obs_TBD = trajectories.obs.reshape(num_steps, num_envs, obs_dim)
        carry = carry._replace(
            aux_obs=carry.aux_obs.at[carry.aux_write_idx].set(obs_TBD),
            aux_returns=carry.aux_returns.at[carry.aux_write_idx].set(targets),
            aux_write_idx=carry.aux_write_idx + 1,
        )

        return carry, system, opt, rngs

    # --- Auxiliary phase ---
    def auxiliary_phase(carry, system, opt, rngs):
        # Flatten buffer: (n_policy_iters * T * B, D) and (n_policy_iters * T * B,)
        total_aux = n_policy_iters * num_steps * num_envs
        all_obs_PD = carry.aux_obs.reshape(total_aux, obs_dim)
        all_returns_P = carry.aux_returns.reshape(total_aux)

        # Compute frozen snapshots before aux updates
        old_logits_PA = jax.lax.stop_gradient(system.ac.actor_head(system.ac.get_hidden(all_obs_PD)))
        old_values_P = jax.lax.stop_gradient(system.ac.value(all_obs_PD))

        aux_batch = AuxMinibatch(
            obs=all_obs_PD,
            returns=all_returns_P,
            old_logits=old_logits_PA,
            old_values=old_values_P,
        )

        def aux_epoch(_, state):
            system, opt, rngs, aux_metrics_sum = state

            # Shuffle and split the flat buffer into minibatches
            perm = jax.random.permutation(rngs(), total_aux)

            def _shuffle_split(x):
                x = jnp.take(x, perm, axis=0)
                return x.reshape(config.aux_num_minibatches, -1, *x.shape[1:])

            minibatches = jax.tree.map(_shuffle_split, aux_batch)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def aux_update_step(state, mb):
                system, opt = state

                def aux_loss_fn(system):
                    obs_BD = mb.obs
                    returns_B = mb.returns
                    old_logits_BA = mb.old_logits
                    old_values_B = mb.old_values

                    # Auxiliary value loss: trains encoder via aux head
                    hidden_BH = system.ac.get_hidden(obs_BD)
                    aux_pred_B = system.aux_critic(hidden_BH).squeeze(-1)
                    aux_value_loss = 0.5 * jnp.square(aux_pred_B - returns_B).mean()

                    # KL loss: prevents policy drift
                    new_logits_BA = system.ac.actor_head(hidden_BH)
                    old_dist = distrax.Categorical(logits=old_logits_BA)
                    new_dist = distrax.Categorical(logits=new_logits_BA)
                    kl_loss = old_dist.kl_divergence(new_dist).mean()

                    # Value regularization (Moon et al. 2022)
                    new_values_B = system.ac.critic_head(jax.lax.stop_gradient(hidden_BH)).squeeze(-1)
                    value_reg_loss = 0.5 * jnp.square(new_values_B - old_values_B).mean()

                    total = aux_value_loss + config.beta_clone * kl_loss + value_reg_loss
                    return total, (aux_value_loss, kl_loss, value_reg_loss)

                (_total, (aux_value_loss, kl_loss, value_reg_loss)), grads = nnx.value_and_grad(aux_loss_fn, has_aux=True)(
                    system
                )
                opt.update(system, grads)
                return (system, opt), {
                    "aux_value_loss": aux_value_loss,
                    "aux_kl_loss": kl_loss,
                    "aux_value_reg_loss": value_reg_loss,
                }

            (system, opt), mb_aux_metrics = aux_update_step((system, opt), minibatches)
            epoch_aux = jax.tree.map(jnp.mean, mb_aux_metrics)
            aux_metrics_sum = jax.tree.map(lambda s, e: s + e, aux_metrics_sum, epoch_aux)
            return (system, opt, rngs, aux_metrics_sum)

        zero_aux_metrics = {
            "aux_value_loss": jnp.float32(0.0),
            "aux_kl_loss": jnp.float32(0.0),
            "aux_value_reg_loss": jnp.float32(0.0),
        }
        system, opt, rngs, aux_metrics_sum = nnx.fori_loop(
            0, config.num_aux_epochs, aux_epoch, (system, opt, rngs, zero_aux_metrics)
        )

        # Average aux metrics over epochs and accumulate
        aux_avg = jax.tree.map(lambda s: s / config.num_aux_epochs, aux_metrics_sum)
        # Pad with zero policy keys so tree structure matches carry.train_metrics
        for k in _PPG_METRIC_NAMES:
            if k not in aux_avg:
                aux_avg[k] = jnp.float32(0.0)
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, aux_avg),
            # Reset buffer write index with traced op
            aux_write_idx=carry.aux_write_idx * 0,
        )

        return carry, system, opt, rngs

    # --- PPG cycle: n_policy_iters policy iterations + auxiliary phase ---
    def ppg_cycle(carry, system, opt, rngs):
        def policy_body(_, s):
            return policy_iteration(*s)

        carry, system, opt, rngs = nnx.fori_loop(0, n_policy_iters, policy_body, (carry, system, opt, rngs))

        carry, system, opt, rngs = auxiliary_phase(carry, system, opt, rngs)
        return carry, system, opt, rngs

    # --- Eval callback ---
    def eval_callback(carry, system, rngs):
        act_fn = make_eval_act(lambda obs, rng: system.ac.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    timesteps_per_cycle = n_policy_iters * num_envs * num_steps
    num_cycles_per_eval = max(1, int(config.eval_freq // timesteps_per_cycle))
    total_cycles = int(np.ceil(config.total_timesteps / timesteps_per_cycle))
    num_evals = max(1, int(np.ceil(total_cycles / num_cycles_per_eval)))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, system, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _PPG_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, system, opt, rngs = state

        def cycle_body(_, s):
            return ppg_cycle(*s)

        carry, system, opt, rngs = nnx.fori_loop(0, num_cycles_per_eval, cycle_body, (carry, system, opt, rngs))

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, system, rngs)
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

        return (carry, system, opt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, system, opt, rngs), all_metrics = scan_eval((carry, system, opt, rngs), jnp.zeros(num_evals))

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "system": system,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
