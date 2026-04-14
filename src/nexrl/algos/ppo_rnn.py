"""PPO with Recurrent Neural Network (PPO-RNN) — standalone training function.

"Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

Ported from Craftax baselines (MichaelTMatthews/craftax_baselines).
Fixed architecture ActorCriticRNN with GRU hidden state propagation.
Discrete actions only.

Key differences from standard PPO:
- Hidden state carried through trajectory collection and reset on done
- Minibatch shuffling along env axis (axis=1), preserving temporal structure
- nnx.scan re-runs full sequence during update (model is grad target)
- Custom RNN-aware evaluation with hidden state propagation

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    T: trajectory length (num_steps)
    H: hidden state dimension (layer_size)
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
    create_env,
    create_vec_env_fns,
    normalize_obs,
    ppo_lr_schedule,
    update_and_normalize_obs,
    update_and_normalize_rew,
)
from nexrl.configs import PPORNNConfig
from nexrl.networks.rnn import ActorCriticRNN
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


class RNNTrajectory(struct.PyTreeNode):
    """Container for trajectory data collected during rollout."""

    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array


class RNNAdvantageMinibatch(struct.PyTreeNode):
    """Container for minibatch data with computed advantages and init hidden state."""

    trajectories: RNNTrajectory
    advantages: chex.Array
    targets: chex.Array
    init_hstate: chex.Array  # (B_mb, H)


class PPORNNCarry(NamedTuple):
    """Non-module carry state for PPO-RNN training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    hstate: chex.Array  # (B, H) — GRU hidden state
    global_step: int
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _create_networks(
    config: PPORNNConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[ActorCriticRNN, nnx.Optimizer]:
    """Create ActorCriticRNN and optimizer."""
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    rngs = nnx.Rngs(rng)

    ac = ActorCriticRNN(
        in_features=in_features,
        action_dim=action_space.n,
        hidden_dim=config.layer_size,
        rngs=rngs,
    )

    learning_rate = ppo_lr_schedule(
        config.learning_rate,
        config.anneal_lr,
        config.total_timesteps,
        config.num_steps,
        config.num_envs,
        config.num_minibatches,
        config.num_epochs,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=config.adam_eps),
    )
    opt = nnx.Optimizer(ac, tx, wrt=nnx.Param)
    return ac, opt


_PPO_RNN_METRIC_NAMES = ("actor_loss", "critic_loss", "entropy", "clip_fraction", "approx_kl", "explained_variance")


def _rnn_shuffle_and_split(
    trajectories: RNNTrajectory,
    advantages_TB: jax.Array,
    targets_TB: jax.Array,
    init_hstate_BH: jax.Array,
    rng: jax.Array,
    num_envs: int,
    num_minibatches: int,
) -> RNNAdvantageMinibatch:
    """Shuffle along env axis (axis=1) and split into minibatches, preserving temporal structure.

    Trajectory (T, B, ...) -> (num_mb, T, mb_size, ...)
    init_hstate (B, H) -> (num_mb, mb_size, H)
    advantages/targets (T, B) -> (num_mb, T, mb_size)
    """
    perm = jax.random.permutation(rng, num_envs)
    mb_size = num_envs // num_minibatches

    def _shuffle_split_tb(x: jax.Array) -> jax.Array:
        # x: (T, B, ...) -> permute axis=1 -> split B -> (T, num_mb, mb_size, ...) -> (num_mb, T, mb_size, ...)
        x = jnp.take(x, perm, axis=1)
        x = x.reshape(x.shape[0], num_minibatches, mb_size, *x.shape[2:])
        # Swap axes 0 and 1: (T, num_mb, mb_size, ...) -> (num_mb, T, mb_size, ...)
        return jnp.swapaxes(x, 0, 1)

    def _shuffle_split_b(x: jax.Array) -> jax.Array:
        # x: (B, ...) -> permute axis=0 -> reshape (num_mb, mb_size, ...)
        x = jnp.take(x, perm, axis=0)
        return x.reshape(num_minibatches, mb_size, *x.shape[1:])

    # Trajectory fields: (T, B, ...) -> (num_mb, T, mb_size, ...)
    shuffled_traj = jax.tree.map(_shuffle_split_tb, trajectories)
    # Advantages/targets: (T, B) -> (num_mb, T, mb_size)
    shuffled_adv = _shuffle_split_tb(advantages_TB)
    shuffled_tgt = _shuffle_split_tb(targets_TB)
    # init_hstate: (B, H) -> (num_mb, mb_size, H)
    shuffled_hstate = _shuffle_split_b(init_hstate_BH)

    return RNNAdvantageMinibatch(
        trajectories=shuffled_traj,
        advantages=shuffled_adv,
        targets=shuffled_tgt,
        init_hstate=shuffled_hstate,
    )


class _RNNEvalState(NamedTuple):
    """Carry state for RNN-aware evaluation."""

    rng: jax.Array
    env_state: Any
    last_obs: jax.Array
    done: jax.Array
    hstate: jax.Array
    return_: jax.Array
    length: jax.Array


def _evaluate_rnn(
    ac: ActorCriticRNN,
    rng: jax.Array,
    env: GymnaxEnv,
    env_params: EnvParams,
    num_seeds: int,
    max_steps: int,
    obs_rms: RMSState | None,
    normalize_observations: bool,
) -> tuple[jax.Array, jax.Array]:
    """Custom RNN-aware evaluation that carries hidden state through episodes.

    Each eval episode: init zero hidden -> step with ac.act(hstate, obs, done, rng)
    -> carry new_hstate. Vmapped over num_seeds.
    """

    def eval_single(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        rng_reset, rng_eval = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)
        obs = jnp.expand_dims(obs, 0)  # (1, D)
        if normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        hstate = jnp.zeros((1, ac.hidden_dim), dtype=jnp.float32)  # (1, H)
        done = jnp.zeros(1, dtype=bool)

        state = _RNNEvalState(
            rng=rng_eval,
            env_state=env_state,
            last_obs=obs,
            done=done,
            hstate=hstate,
            return_=jnp.float32(0.0),
            length=jnp.int32(0),
        )

        def cond_fn(state: _RNNEvalState) -> jax.Array:
            return jnp.logical_and(state.length < max_steps, jnp.logical_not(state.done.squeeze()))

        def body_fn(state: _RNNEvalState) -> _RNNEvalState:
            rng, rng_act, rng_step = jax.random.split(state.rng, 3)
            hstate, action = ac.act(state.hstate, state.last_obs, state.done, rng_act)
            obs, env_state, reward, done, _ = env.step(rng_step, state.env_state, action.squeeze(), env_params)
            obs = jnp.expand_dims(obs, 0)  # (1, D)
            if normalize_observations and obs_rms is not None:
                obs = normalize_obs(obs_rms, obs)
            return _RNNEvalState(
                rng=rng,
                env_state=env_state,
                last_obs=obs,
                done=jnp.expand_dims(done, 0),
                hstate=hstate,
                return_=state.return_ + reward.squeeze(),
                length=state.length + 1,
            )

        state = jax.lax.while_loop(cond_fn, body_fn, state)
        return state.length, state.return_

    seeds = jax.random.split(rng, num_seeds)
    return jax.vmap(eval_single)(seeds)


def train_ppo_rnn(
    config: PPORNNConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a PPO-RNN agent.

    PPO with GRU-based recurrent policy for partially observable environments.

    Args:
        config: PPORNNConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, PPORNNConfig

        config = PPORNNConfig.from_yaml("configs/ppo_rnn/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("ppo_rnn"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    num_steps = config.num_steps
    max_steps = env_params.max_steps_in_episode

    assert num_envs % config.num_minibatches == 0

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    ac, opt = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(env.observation_space(env_params).shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init hidden state
    hstate = ac.initialize_carry(num_envs)  # (B, H)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _PPO_RNN_METRIC_NAMES}

    carry = PPORNNCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        hstate=hstate,
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs, dtype=jnp.float32),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, ac, rngs):
        """Roll out policy for num_steps. AC is read-only. Saves init_hstate."""
        init_hstate = carry.hstate  # Save for minibatch construction
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_steps, rng_action = jax.random.split(step_key, 2)
            rng_steps = jax.random.split(rng_steps, num_envs)

            # RNN forward: pass hstate, obs, done
            new_hstate_BH, action_B, log_prob_B, value_B = ac.get_action_and_value(
                carry.hstate, carry.last_obs, carry.last_done, rng_action
            )

            next_obs_BD, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_B, env_params)

            # Track episode returns using raw rewards (before normalization)
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

            transition = RNNTrajectory(carry.last_obs, action_B, log_prob_B, reward_B, value_B, done_B)
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs_BD,
                last_done=done_B,
                hstate=new_hstate_BH,
                global_step=carry.global_step + num_envs,
                obs_rms=obs_rms,
                rew_rms=rew_rms,
                running_return=reset_running_B,
                completed_returns_sum=completed_sum,
                completed_returns_count=completed_count,
            )
            return carry, transition

        carry, trajectories = jax.lax.scan(env_step, carry, step_keys)
        return carry, trajectories, init_hstate

    # --- Combined update ---
    def update_ac(batch, ac, opt):
        def loss_fn(model):
            # Re-run full sequence via nnx.scan over timesteps
            # model is the gradient target inside nnx.value_and_grad, so we must use nnx.scan
            # Pack (model, hstate) as single carry — nnx.scan allows only one Carry arg
            @nnx.scan(in_axes=(nnx.Carry, 0, 0, 0), out_axes=(nnx.Carry, 0, 0, 0))
            def scan_step(carry, obs_B, done_B, action_B):
                model, hstate_BH = carry
                hstate_BH, log_prob_B, entropy_B, value_B = model.get_pi_and_value(hstate_BH, obs_B, done_B, action_B)
                return (model, hstate_BH), log_prob_B, entropy_B, value_B

            (model, _), log_prob_TB, entropy_TB, value_TB = scan_step(
                (model, batch.init_hstate),
                batch.trajectories.obs,
                batch.trajectories.done,
                batch.trajectories.action,
            )

            # Flatten (T, mb_size) -> (T*mb_size,) for PPO loss computation
            log_prob_B = log_prob_TB.reshape(-1)
            entropy_B = entropy_TB.reshape(-1)
            value_B = value_TB.reshape(-1)
            old_log_prob_B = batch.trajectories.log_prob.reshape(-1)
            old_value_B = batch.trajectories.value.reshape(-1)
            advantages_B = batch.advantages.reshape(-1)
            targets_B = batch.targets.reshape(-1)

            entropy = entropy_B.mean()

            # Actor loss (PPO clipped)
            ratio_B = jnp.exp(log_prob_B - old_log_prob_B)
            adv_B = (advantages_B - advantages_B.mean()) / (advantages_B.std() + 1e-8)
            clipped_ratio_B = jnp.clip(ratio_B, 1 - config.clip_eps, 1 + config.clip_eps)
            pi_loss1_B = ratio_B * adv_B
            pi_loss2_B = clipped_ratio_B * adv_B
            actor_loss = -jnp.minimum(pi_loss1_B, pi_loss2_B).mean()

            # Critic loss (clipped)
            value_clipped_B = old_value_B + (value_B - old_value_B).clip(-config.clip_eps, config.clip_eps)
            loss_B = jnp.square(value_B - targets_B)
            loss_clipped_B = jnp.square(value_clipped_B - targets_B)
            critic_loss = 0.5 * jnp.maximum(loss_B, loss_clipped_B).mean()

            clip_fraction = jnp.mean(jnp.abs(ratio_B - 1) > config.clip_eps)
            approx_kl = jnp.mean((ratio_B - 1) - jnp.log(ratio_B))

            total_loss = actor_loss + config.vf_coef * critic_loss - config.ent_coef * entropy
            return total_loss, (actor_loss, critic_loss, entropy, clip_fraction, approx_kl)

        (_total, (actor_loss, critic_loss, entropy, clip_fraction, approx_kl)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(ac)
        opt.update(ac, grads)
        return actor_loss, critic_loss, entropy, clip_fraction, approx_kl

    # --- Train iteration ---
    def train_iteration(carry, ac, opt, rngs):
        carry, trajectories, init_hstate = collect_trajectories(carry, ac, rngs)

        _, last_val_B = ac.value(carry.hstate, carry.last_obs, carry.last_done)
        last_val_B = jnp.where(carry.last_done, 0, last_val_B)
        advantages, targets = calculate_gae(
            trajectories.reward,
            trajectories.value,
            trajectories.done,
            last_val_B,
            config.gamma,
            config.gae_lambda,
        )

        # Explained variance (computed before updates modify model)
        explained_var = 1 - jnp.var(targets - trajectories.value) / (jnp.var(targets) + 1e-8)

        update_metric_names = ("actor_loss", "critic_loss", "entropy", "clip_fraction", "approx_kl")
        zero_update_metrics = {k: jnp.float32(0.0) for k in update_metric_names}

        def update_epoch(_, state):
            ac, opt, rngs, metrics_sum = state

            minibatches = _rnn_shuffle_and_split(
                trajectories, advantages, targets, init_hstate, rngs(), num_envs, config.num_minibatches
            )

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def update_step(state, mb):
                ac, opt = state
                actor_loss, critic_loss, entropy, clip_fraction, approx_kl = update_ac(mb, ac, opt)
                return (ac, opt), {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "clip_fraction": clip_fraction,
                    "approx_kl": approx_kl,
                }

            (ac, opt), mb_metrics = update_step((ac, opt), minibatches)
            epoch_metrics = jax.tree.map(jnp.mean, mb_metrics)
            metrics_sum = jax.tree.map(lambda s, e: s + e, metrics_sum, epoch_metrics)
            return (ac, opt, rngs, metrics_sum)

        ac, opt, rngs, epoch_metrics_sum = nnx.fori_loop(
            0, config.num_epochs, update_epoch, (ac, opt, rngs, zero_update_metrics)
        )

        # Average over epochs, add explained variance
        iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, epoch_metrics_sum)
        iter_metrics["explained_variance"] = explained_var

        # Accumulate in carry
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, iter_metrics),
            metrics_count=carry.metrics_count + 1,
        )

        return carry, ac, opt, rngs

    # --- Eval callback ---
    def eval_callback(carry, ac, rngs):
        return _evaluate_rnn(ac, rngs(), env, env_params, 128, max_steps, carry.obs_rms, config.normalize_observations)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, ac, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _PPO_RNN_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, ac, opt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, ac, opt, rngs = nnx.fori_loop(0, num_iters_per_eval, train_body, (carry, ac, opt, rngs))

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, ac, rngs)
        else:
            eval_lengths = jnp.zeros(128, dtype=jnp.float32)
            eval_returns = jnp.zeros(128, dtype=jnp.float32)

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

        return (carry, ac, opt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, ac, opt, rngs), all_metrics = scan_eval((carry, ac, opt, rngs), jnp.zeros(num_evals))

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "ac": ac,
        "opt": opt,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
