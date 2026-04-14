"""MRS.Q — Model-based Representations for Search and Q-learning.

"Model-based Representations for Search and Q-learning" (Fujimoto et al.)

Builds on MrQ with three targeted modifications:
1. MPPI planning for action selection instead of policy + noise
2. Ensemble of 10 Q-functions with aggressive minimization
3. Simplicial embeddings (SEM) for stable latent representations

Encoder updates happen in bursts every Ttarget steps; value and policy updates
happen every step. MPPI replaces exploration noise during data collection.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    D: observation dimension
    A: action dimension
    S: zs_dim (state embedding)
    X: zsa_dim (state-action embedding)
    N: num_bins (symexp two-hot)
    Q: num_critics (ensemble size, default 10)
    T: horizon (enc_horizon or q_horizon)
    H: mpc_horizon (MPPI planning)
    P: num proposals (num_samples + num_pi_trajs)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

from nexrl.algos.utils import (
    RewardRMSState,
    RMSState,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    normalize_obs,
    update_rew_rms,
    update_rms,
)
from nexrl.buffers import CircularBuffer
from nexrl.configs import MrsQConfig
from nexrl.evaluate import evaluate
from nexrl.networks.mrq import (
    MrQPolicy,
    make_symexp_support,
    symexp_twohot_cross_entropy,
    symexp_twohot_decode,
)
from nexrl.networks.mrsq import MrsQEncoder, MrsQValue
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_MRSQ_METRIC_NAMES = ("encoder_loss", "value_loss", "policy_loss", "mean_q_value", "reward_scale")


class MrsQTransition(NamedTuple):
    """Transition stored in CircularBuffer for MRS.Q."""

    obs: chex.Array  # (*obs_shape)
    action: chex.Array  # (A,) one-hot for discrete, raw for continuous
    reward: chex.Array  # ()
    done: chex.Array  # () bool


class MrsQCarry(NamedTuple):
    """Non-module carry state for MRS.Q training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    buffer: CircularBuffer
    priorities: chex.Array  # (buffer_size,) separate PER priorities
    max_priority: chex.Array  # scalar
    reward_scale: chex.Array  # mean |reward| across buffer
    target_reward_scale: chex.Array  # reward_scale at last target update
    training_steps: int  # counts env steps since training started (for burst timing)
    obs_rms: RMSState
    rew_rms: RewardRMSState
    prev_mean: chex.Array  # (num_envs, mpc_horizon, action_dim) MPPI warm-start
    # Metrics
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _hard_copy(src: nnx.Module, dst: nnx.Module) -> None:
    """Hard copy all Param variables from src to dst."""
    nnx.update(dst, nnx.state(src, nnx.Param))


def _create_networks(
    config: MrsQConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[
    MrsQEncoder,
    nnx.Optimizer,
    MrsQEncoder,
    MrsQValue,
    nnx.Optimizer,
    MrsQValue,
    MrQPolicy,
    nnx.Optimizer,
    MrQPolicy,
]:
    """Create encoder, value, policy (online + target) with separate optimizers."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    obs_dim = int(np.prod(obs_space.shape))
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]

    if discrete:
        action_dim = action_space.n
    else:
        action_dim = int(np.prod(action_space.shape))

    activation = getattr(nnx, config.activation)
    rngs = nnx.Rngs(rng)

    def _make_encoder():
        return MrsQEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            zs_dim=config.zs_dim,
            za_dim=config.za_dim,
            zsa_dim=config.zsa_dim,
            num_bins=config.reward_num_bins,
            discrete=discrete,
            activation=activation,
            sem_groups=config.sem_groups,
            rngs=rngs,
        )

    def _make_value():
        return MrsQValue(
            zsa_dim=config.zsa_dim,
            hidden_dim=config.hidden_dim,
            num_critics=config.num_critics,
            activation=activation,
            rngs=rngs,
        )

    policy_activation = getattr(nnx, config.policy_activation)

    def _make_policy():
        return MrQPolicy(
            zs_dim=config.zs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            discrete=discrete,
            activation=policy_activation,
            rngs=rngs,
        )

    # Encoder + target
    encoder = _make_encoder()
    enc_tx = optax.adamw(learning_rate=config.enc_learning_rate, weight_decay=config.enc_weight_decay)
    enc_opt = nnx.Optimizer(encoder, enc_tx, wrt=nnx.Param)
    encoder_target = _make_encoder()
    _hard_copy(encoder, encoder_target)

    # Value + target
    value = _make_value()
    val_tx = optax.chain(
        optax.clip_by_global_norm(config.value_grad_clip),
        optax.adamw(learning_rate=config.value_learning_rate, weight_decay=config.value_weight_decay),
    )
    val_opt = nnx.Optimizer(value, val_tx, wrt=nnx.Param)
    value_target = _make_value()
    _hard_copy(value, value_target)

    # Policy + target (reuses MrQPolicy unchanged)
    policy = _make_policy()
    pol_tx = optax.adamw(learning_rate=config.policy_learning_rate, weight_decay=config.policy_weight_decay)
    pol_opt = nnx.Optimizer(policy, pol_tx, wrt=nnx.Param)
    policy_target = _make_policy()
    _hard_copy(policy, policy_target)

    return encoder, enc_opt, encoder_target, value, val_opt, value_target, policy, pol_opt, policy_target


def train_mrsq(
    config: MrsQConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a MRS.Q agent.

    MRS.Q extending MrQ with MPPI planning and Q-ensemble.

    Args:
        config: MrsQConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, MrsQConfig

        config = MrsQConfig.from_yaml("configs/mrsq/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("mrsq"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]

    if discrete:
        action_dim = action_space.n
        max_action = 1.0
    else:
        action_dim = int(np.prod(action_space.shape))
        max_action = float(action_space.high)

    # Scale noise for discrete (reference: actions in [0,1] vs continuous [-1,1])
    target_noise = config.target_noise * (0.5 if discrete else 1.0)
    target_noise_clip = config.target_noise_clip * (0.5 if discrete else 1.0)

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    (
        encoder,
        enc_opt,
        encoder_target,
        value,
        val_opt,
        value_target,
        policy,
        pol_opt,
        policy_target,
    ) = _create_networks(config, env, env_params, rngs())

    # Reward support bins
    bins = make_symexp_support(config.reward_lower, config.reward_upper, config.reward_num_bins)

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init buffer — actions always stored as (A,) float (one-hot for discrete)
    dummy_transition = MrsQTransition(
        obs=jnp.empty((config.buffer_size, *obs_space.shape), dtype=obs_space.dtype),
        action=jnp.empty((config.buffer_size, action_dim), dtype=jnp.float32),
        reward=jnp.empty(config.buffer_size, dtype=jnp.float32),
        done=jnp.empty(config.buffer_size, dtype=bool),
    )
    buf = CircularBuffer.empty(config.buffer_size, dummy_transition)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _MRSQ_METRIC_NAMES}

    carry = MrsQCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        buffer=buf,
        priorities=jnp.ones(config.buffer_size, dtype=jnp.float32),
        max_priority=jnp.float32(1.0),
        reward_scale=jnp.float32(1.0),
        target_reward_scale=jnp.float32(1.0),
        training_steps=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        prev_mean=jnp.zeros((num_envs, config.mpc_horizon, action_dim), dtype=jnp.float32),
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- MPPI helpers ---
    def estimate_value(zs_S, actions_PHA, encoder_, policy_, value_, reward_scale):
        """Score action trajectories via model rollout + terminal Q-value.

        Args:
            zs_S: (S,) initial state embedding for one env
            actions_PHA: (P, H, A) action proposals
            encoder_: encoder for model predictions
            policy_: policy for terminal action
            value_: value ensemble for terminal Q
            reward_scale: current reward scale for normalization
        Returns:
            scores_P: (P,) estimated trajectory values
        """
        P = actions_PHA.shape[0]
        actions_HPA = jnp.transpose(actions_PHA, (1, 0, 2))  # (H, P, A)
        zs_PS = jnp.broadcast_to(zs_S[None], (P, zs_S.shape[0]))

        def step_fn(carry_inner, action_PA):
            zs_curr_PS, cum_reward_P, discount_P = carry_inner
            done_logit_P, next_zs_PS, reward_logits_PN = encoder_.model_all(zs_curr_PS, action_PA)
            reward_P = symexp_twohot_decode(reward_logits_PN, bins)
            done_P = jnp.round(jax.nn.sigmoid(done_logit_P))
            cum_reward_P = cum_reward_P + discount_P * reward_P
            discount_P = discount_P * config.gamma * (1.0 - done_P)
            return (next_zs_PS, cum_reward_P, discount_P), None

        init = (zs_PS, jnp.zeros(P), jnp.ones(P))
        (final_zs_PS, cum_reward_P, final_discount_P), _ = jax.lax.scan(step_fn, init, actions_HPA)

        # Terminal value from policy
        final_action_PA, _ = policy_.act(final_zs_PS)
        final_zsa_PX = encoder_.zsa(final_zs_PS, final_action_PA)
        final_q_PQ = value_(final_zsa_PX)  # (P, Q)
        min_q_P = jnp.min(final_q_PQ, axis=-1)  # (P,)

        # (G + discount * r' * min_Q) / r  (matches MrQ Eq 19 structure)
        scores_P = (cum_reward_P + final_discount_P * reward_scale * min_q_P) / jnp.maximum(reward_scale, 1e-8)
        return scores_P

    def mppi_plan_single(zs_S, prev_mean_HA, rng_mppi, encoder_, policy_, value_, reward_scale):
        """MPPI planning for a single environment (vmapped over envs).

        Args:
            zs_S: (S,) state embedding
            prev_mean_HA: (H, A) warm-start mean from previous step
            rng_mppi: PRNG key
            encoder_: encoder (read-only)
            policy_: policy (read-only)
            value_: value ensemble (read-only)
            reward_scale: current reward scale
        Returns:
            action_A: (A,) selected action
            new_prev_mean_HA: (H, A) warm-start for next step
        """
        H = config.mpc_horizon
        A = action_dim
        num_pi = config.num_pi_trajs
        num_samp = config.num_samples
        num_elite = config.num_elites

        rng_pi, rng_cem, rng_select = jax.random.split(rng_mppi, 3)

        # 1. Generate policy trajectories via model rollout
        init_zs_PS = jnp.broadcast_to(zs_S[None], (num_pi, zs_S.shape[0]))

        def pi_step(carry_pi, rng_t):
            zs_curr = carry_pi  # (num_pi, S)
            noise = config.pi_std * jax.random.normal(rng_t, (num_pi, A))
            action_PA, _ = policy_.act(zs_curr)
            if discrete:
                noisy = action_PA + noise
                action_PA = jax.nn.one_hot(jnp.argmax(noisy, axis=-1), A, dtype=jnp.float32)
            else:
                action_PA = jnp.clip(action_PA + noise, -1.0, 1.0)
            _, next_zs, _ = encoder_.model_all(zs_curr, action_PA)
            return next_zs, action_PA

        pi_rngs = jax.random.split(rng_pi, H)
        _, pi_actions_HPA = jax.lax.scan(pi_step, init_zs_PS, pi_rngs)
        pi_actions_PHA = jnp.transpose(pi_actions_HPA, (1, 0, 2))  # (num_pi, H, A)

        # 2. CEM loop
        init_std = jnp.full((H, A), (config.max_std + config.min_std) / 2)

        def cem_iter(_, state):
            mean_HA, std_HA, rng_it, _, _ = state
            rng_it, rng_samp = jax.random.split(rng_it)

            # Sample random trajectories from N(mean, std)
            noise_PHA = jax.random.normal(rng_samp, (num_samp, H, A))
            random_PHA = mean_HA[None] + std_HA[None] * noise_PHA
            if discrete:
                random_PHA = jax.nn.one_hot(jnp.argmax(random_PHA, axis=-1), A, dtype=jnp.float32)
            else:
                random_PHA = jnp.clip(random_PHA, -1.0, 1.0)

            # Concatenate with policy trajectories
            all_PHA = jnp.concatenate([random_PHA, pi_actions_PHA], axis=0)

            # Evaluate all proposals
            scores_P = estimate_value(zs_S, all_PHA, encoder_, policy_, value_, reward_scale)

            # Select top elites
            elite_idx = jnp.argsort(scores_P)[-(num_elite):]
            elite_actions = all_PHA[elite_idx]  # (num_elites, H, A)
            elite_scores = scores_P[elite_idx]  # (num_elites,)

            # Weighted mean/std from elites
            weights = jax.nn.softmax(elite_scores / config.mpc_temperature)  # (num_elites,)
            new_mean_HA = jnp.sum(weights[:, None, None] * elite_actions, axis=0)
            diff = elite_actions - new_mean_HA[None]
            new_std_HA = jnp.sqrt(jnp.sum(weights[:, None, None] * diff**2, axis=0) + 1e-8)
            new_std_HA = jnp.clip(new_std_HA, config.min_std, config.max_std)

            return new_mean_HA, new_std_HA, rng_it, elite_actions, elite_scores

        init_elites = jnp.zeros((num_elite, H, A))
        init_elite_scores = jnp.zeros(num_elite)

        _, _, _, final_elite_actions, final_elite_scores = jax.lax.fori_loop(
            0,
            config.mpc_iterations,
            cem_iter,
            (prev_mean_HA, init_std, rng_cem, init_elites, init_elite_scores),
        )

        # 3. Gumbel-softmax selection from last iteration's elites
        u = jax.random.uniform(rng_select, final_elite_scores.shape, minval=1e-8, maxval=1.0 - 1e-8)
        gumbel_noise = -jnp.log(-jnp.log(u))
        selected_idx = jnp.argmax(final_elite_scores / config.mpc_temperature + gumbel_noise)
        selected_traj_HA = final_elite_actions[selected_idx]  # (H, A)

        action_A = selected_traj_HA[0]

        # Warm-start: shift selected trajectory forward, zero-pad last step
        new_prev_mean_HA = jnp.concatenate([selected_traj_HA[1:], jnp.zeros((1, A))], axis=0)

        return action_A, new_prev_mean_HA

    # --- Collect transitions ---
    def collect_transition(carry, encoder_, policy_, value_, rngs_):
        start_training = carry.global_step > config.fill_buffer
        rng_action = rngs_()

        obs_for_net = carry.last_obs
        if config.normalize_observations:
            obs_for_net = normalize_obs(carry.obs_rms, obs_for_net)

        def sample_uniform(rng):
            if discrete:
                idx = jax.random.randint(rng, (num_envs,), 0, action_dim)
                return jax.nn.one_hot(idx, action_dim, dtype=jnp.float32), carry.prev_mean
            else:
                return jax.random.uniform(rng, (num_envs, action_dim), minval=-1.0, maxval=1.0), carry.prev_mean

        def sample_mppi(rng):
            zs_BS = jax.lax.stop_gradient(encoder_.zs(obs_for_net))
            rngs_B = jax.random.split(rng, num_envs)

            # vmap MPPI over envs — modules captured by closure (read-only)
            actions_BA, new_prev_mean = jax.vmap(
                lambda zs, pm, r: mppi_plan_single(zs, pm, r, encoder_, policy_, value_, carry.reward_scale)
            )(zs_BS, carry.prev_mean, rngs_B)

            return actions_BA, new_prev_mean

        actions_BA, new_prev_mean = jax.lax.cond(start_training, sample_mppi, sample_uniform, rng_action)

        # Reset prev_mean to zeros for envs that terminated last step
        new_prev_mean = jnp.where(carry.last_done[:, None, None], 0.0, new_prev_mean)

        # Step environment: scale continuous actions by max_action for env
        if discrete:
            action_env = jnp.argmax(actions_BA, axis=-1)
        else:
            action_env = actions_BA * max_action
        rng_steps = jax.random.split(rngs_(), num_envs)
        next_obs, new_env_state, rewards, dones, _ = vmap_step(rng_steps, carry.env_state, action_env, env_params)

        # Track episode returns
        new_running_B = carry.running_return + rewards
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * dones)
        completed_count = carry.completed_returns_count + jnp.sum(dones.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - dones)

        obs_rms = carry.obs_rms
        rew_rms = carry.rew_rms
        if config.normalize_observations:
            obs_rms = update_rms(obs_rms, next_obs)
        if config.normalize_rewards:
            rew_rms = update_rew_rms(rew_rms, rewards, dones, config.gamma)

        transition = MrsQTransition(
            obs=carry.last_obs,
            action=actions_BA,
            reward=rewards.astype(jnp.float32),
            done=dones,
        )

        # Store with max priority
        batch_flat, _ = jax.tree.flatten(transition)
        batch_size = batch_flat[0].shape[0]
        idx = (carry.buffer.index + jnp.arange(batch_size)) % carry.buffer.size
        new_priorities = carry.priorities.at[idx].set(carry.max_priority)

        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            last_done=dones,
            global_step=carry.global_step + num_envs,
            buffer=carry.buffer.extend(transition),
            priorities=new_priorities,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            prev_mean=new_prev_mean,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry

    # --- PER sampling ---
    def sample_per(priorities, num_entries, batch_size, rng):
        """Stratified PER sampling. Returns indices."""
        valid_mask = jnp.arange(config.buffer_size) < num_entries
        probs = (priorities**config.per_alpha) * valid_mask
        total = probs.sum()
        probs_norm = probs / jnp.maximum(total, 1e-8)
        cumsum = jnp.cumsum(probs_norm)
        segment_size = 1.0 / batch_size
        offsets = jnp.arange(batch_size, dtype=jnp.float32) * segment_size
        u = jax.random.uniform(rng, (batch_size,), minval=0.0, maxval=segment_size)
        sample_points = offsets + u
        indices = jnp.searchsorted(cumsum, sample_points)
        return jnp.clip(indices, 0, num_entries - 1)

    # --- Encoder update step ---
    def update_encoder_step(encoder_, enc_opt_, encoder_target_, buffer_, obs_rms_, rng_sample):
        """Single encoder update: sample sequences and compute encoder loss."""
        n = buffer_.num_entries
        num_timesteps = n // num_envs
        max_start_t = jnp.maximum(num_timesteps - config.enc_horizon - 1, 1)
        rng_t, rng_e = jax.random.split(rng_sample)
        start_timesteps = jax.random.randint(rng_t, (config.batch_size,), 0, max_start_t)
        env_ids = jax.random.randint(rng_e, (config.batch_size,), 0, num_envs)
        offsets = jnp.arange(config.enc_horizon + 1)
        seq_indices = env_ids[:, None] + (start_timesteps[:, None] + offsets[None, :]) * num_envs
        seq_indices = seq_indices % buffer_.size
        seqs = jax.tree.map(lambda arr: arr[seq_indices], buffer_.data)

        def enc_loss_fn(enc_model):
            obs_seq = seqs.obs  # (B, H+1, D)
            action_seq = seqs.action  # (B, H+1, A)
            reward_seq = seqs.reward  # (B, H+1)
            done_seq = seqs.done  # (B, H+1)
            B_enc = config.batch_size
            H_enc = config.enc_horizon

            obs_for_net = obs_seq
            if config.normalize_observations:
                obs_flat = obs_seq.reshape(B_enc * (H_enc + 1), -1)
                obs_flat = normalize_obs(obs_rms_, obs_flat)
                obs_for_net = obs_flat.reshape(B_enc, H_enc + 1, -1)

            # Target zs for each future step (from target encoder, stop gradient)
            target_zs_all = jax.lax.stop_gradient(
                jax.vmap(encoder_target_.zs)(obs_for_net[:, 1:].reshape(B_enc * H_enc, -1)).reshape(B_enc, H_enc, -1)
            )

            # Initial zs from online encoder
            zs_BS = enc_model.zs(obs_for_net[:, 0])

            total_loss = jnp.float32(0.0)
            prev_not_done = jnp.ones(B_enc, dtype=jnp.float32)

            def unroll_step(carry_inner, t):
                zs_curr, prev_nd, loss_acc = carry_inner
                action_t = action_seq[:, t]
                done_logit_B, zs_pred_BS, reward_logits_BN = enc_model.model_all(zs_curr, action_t)

                target_zs_t = target_zs_all[:, t]
                reward_t = reward_seq[:, t]
                done_t = done_seq[:, t].astype(jnp.float32)

                dyn_loss_B = jnp.mean((zs_pred_BS - target_zs_t) ** 2, axis=-1)
                rew_loss_B = symexp_twohot_cross_entropy(reward_logits_BN, reward_t, bins)
                done_loss_B = (jax.nn.sigmoid(done_logit_B) - done_t) ** 2

                step_loss = prev_nd * (
                    config.dyn_weight * dyn_loss_B + config.reward_weight * rew_loss_B + config.done_weight * done_loss_B
                )
                loss_acc = loss_acc + step_loss.mean()
                new_nd = prev_nd * (1.0 - done_t)
                return (zs_pred_BS, new_nd, loss_acc), None

            (_, _, total_loss), _ = jax.lax.scan(unroll_step, (zs_BS, prev_not_done, total_loss), jnp.arange(H_enc))
            return total_loss / H_enc

        loss, grads = nnx.value_and_grad(enc_loss_fn)(encoder_)
        enc_opt_.update(encoder_, grads)
        return loss

    # --- Single RL update step (value + policy) ---
    def update_rl_step(
        carry, encoder_, encoder_target_, value_, val_opt_, value_target_, policy_, pol_opt_, policy_target_, rng
    ):
        """Single RL update: sample via PER, compute n-step return, update value and policy."""
        rng_per, rng_noise, rng_policy = jax.random.split(rng, 3)
        indices = sample_per(carry.priorities, carry.buffer.num_entries, config.batch_size, rng_per)

        # Get sequences of length q_horizon for n-step returns
        env_ids = indices % num_envs
        timesteps = indices // num_envs
        q_offsets = jnp.arange(config.q_horizon)
        q_indices = env_ids[:, None] + (timesteps[:, None] + q_offsets[None, :]) * num_envs
        q_indices = q_indices % carry.buffer.size
        seqs = jax.tree.map(lambda arr: arr[q_indices], carry.buffer.data)

        obs_B = seqs.obs[:, 0]
        action_B = seqs.action[:, 0]

        obs_B_net = normalize_obs(carry.obs_rms, obs_B) if config.normalize_observations else obs_B

        # N-step return
        rewards_BT = seqs.reward
        dones_BT = seqs.done.astype(jnp.float32)

        def compute_nstep(carry_ns, t):
            G, discount = carry_ns
            G = G + discount * rewards_BT[:, t]
            discount = discount * config.gamma * (1.0 - dones_BT[:, t])
            return (G, discount), None

        (G_B, final_discount_B), _ = jax.lax.scan(
            compute_nstep,
            (jnp.zeros(config.batch_size, dtype=jnp.float32), jnp.ones(config.batch_size, dtype=jnp.float32)),
            jnp.arange(config.q_horizon),
        )

        # Next obs at t=q_horizon
        next_indices = indices + config.q_horizon * num_envs
        next_indices = next_indices % carry.buffer.size
        next_obs_B = jax.tree.map(lambda arr: arr[next_indices], carry.buffer.data).obs
        next_obs_B_net = normalize_obs(carry.obs_rms, next_obs_B) if config.normalize_observations else next_obs_B

        # Target Q-value — min over ensemble
        next_zs_BS = jax.lax.stop_gradient(encoder_target_.zs(next_obs_B_net))
        next_action_BA, _ = policy_target_.act(next_zs_BS)

        # Add clipped Gaussian noise to target action
        if discrete:
            noise = target_noise * jax.random.normal(rng_noise, next_action_BA.shape)
            noisy = next_action_BA + noise
            next_action_noisy_BA = jax.nn.one_hot(jnp.argmax(noisy, axis=-1), action_dim, dtype=jnp.float32)
        else:
            noise = jnp.clip(
                target_noise * jax.random.normal(rng_noise, next_action_BA.shape),
                -target_noise_clip,
                target_noise_clip,
            )
            next_action_noisy_BA = jnp.clip(next_action_BA + noise, -1.0, 1.0)

        next_zsa_BX = encoder_target_.zsa(next_zs_BS, next_action_noisy_BA)
        q_target_BQ = value_target_(next_zsa_BX)  # (B, Q)
        q_target_B = jnp.min(q_target_BQ, axis=-1)  # min over ensemble

        # Target: (1/r_bar) * (G + gamma^HQ * r_bar' * min Q'_j)
        target_B = (G_B + final_discount_B * q_target_B * carry.target_reward_scale) / jnp.maximum(carry.reward_scale, 1e-8)
        target_B = jax.lax.stop_gradient(target_B)

        # Encoder embeddings (stop_gradient — encoder not updated in RL phase)
        zs_BS = jax.lax.stop_gradient(encoder_.zs(obs_B_net))
        zsa_BX = jax.lax.stop_gradient(encoder_.zsa(zs_BS, action_B))

        # --- Value update: Huber loss across all Q heads ---
        def value_loss_fn(value_model):
            q_BQ = value_model(zsa_BX)  # (B, Q)
            target_BQ = jnp.broadcast_to(target_B[:, None], q_BQ.shape)
            return optax.huber_loss(q_BQ, target_BQ).mean()

        v_loss, v_grads = nnx.value_and_grad(value_loss_fn)(value_)
        val_opt_.update(value_, v_grads)

        mean_q = value_(zsa_BX).mean()

        # --- Policy update ---
        # Gradient flows: policy → action → encoder.zsa → value → loss
        def policy_loss_fn(policy_model):
            action_BA, pre_activ_BA = policy_model(zs_BS, rng_policy, tau=config.gumbel_tau)
            zsa_pol_BX = encoder_.zsa(zs_BS, action_BA)
            q_BQ = value_(zsa_pol_BX)  # (B, Q)
            q_loss = -q_BQ.mean()  # average over ensemble and batch
            reg_loss = config.pre_activ_weight * (pre_activ_BA**2).mean()
            return q_loss + reg_loss

        p_loss, p_grads = nnx.value_and_grad(policy_loss_fn)(policy_)
        pol_opt_.update(policy_, p_grads)

        # --- Priority update (LAP) — max TD error across ensemble ---
        q_current_BQ = value_(zsa_BX)  # (B, Q)
        td_error_BQ = jnp.abs(q_current_BQ - target_B[:, None])
        td_error_B = td_error_BQ.max(axis=-1)  # max across ensemble
        new_pri_B = jnp.maximum(td_error_B, config.min_priority) ** config.per_alpha
        new_priorities = carry.priorities.at[indices].set(new_pri_B)
        new_max_priority = jnp.maximum(carry.max_priority, new_pri_B.max())

        carry = carry._replace(priorities=new_priorities, max_priority=new_max_priority)
        return carry, v_loss, p_loss, mean_q

    # --- Encoder burst updates (encoder only) ---
    def encoder_burst(
        carry,
        encoder_,
        enc_opt_,
        encoder_target_,
        value_,
        value_target_,
        policy_,
        policy_target_,
        rngs_,
    ):
        """Hard copy targets, recompute reward_scale, then run Ttarget encoder updates."""
        _hard_copy(encoder_, encoder_target_)
        _hard_copy(value_, value_target_)
        _hard_copy(policy_, policy_target_)

        # Update reward scaling
        n = carry.buffer.num_entries
        abs_rewards = jnp.abs(carry.buffer.data.reward)
        valid_mask = jnp.arange(config.buffer_size) < n
        new_reward_scale = jnp.maximum(jnp.sum(abs_rewards * valid_mask) / jnp.maximum(n.astype(jnp.float32), 1.0), 1e-4)
        carry = carry._replace(target_reward_scale=new_reward_scale, reward_scale=new_reward_scale)

        # Burst: Ttarget iterations of encoder updates only
        def burst_body(_, state):
            carry_inner, enc_inner, enc_opt_inner, enc_tgt_inner, rngs_inner, enc_loss_sum = state
            rng_enc = rngs_inner()
            enc_loss = update_encoder_step(
                enc_inner, enc_opt_inner, enc_tgt_inner, carry_inner.buffer, carry_inner.obs_rms, rng_enc
            )
            return (carry_inner, enc_inner, enc_opt_inner, enc_tgt_inner, rngs_inner, enc_loss_sum + enc_loss)

        carry, encoder_, enc_opt_, encoder_target_, rngs_, total_enc_loss = nnx.fori_loop(
            0,
            config.target_update_freq,
            burst_body,
            (carry, encoder_, enc_opt_, encoder_target_, rngs_, jnp.float32(0.0)),
        )

        avg_enc_loss = total_enc_loss / config.target_update_freq

        return carry, encoder_, enc_opt_, encoder_target_, value_target_, policy_target_, rngs_, avg_enc_loss

    # --- Train iteration ---
    def train_iteration(
        carry, encoder_, enc_opt_, encoder_target_, value_, val_opt_, value_target_, policy_, pol_opt_, policy_target_, rngs_
    ):
        # Collect transition (MPPI replaces policy+noise)
        carry = collect_transition(carry, encoder_, policy_, value_, rngs_)

        start_training = carry.global_step > config.fill_buffer

        # Check if we should do encoder burst
        do_burst = jnp.logical_and(start_training, carry.training_steps % config.target_update_freq == 0)

        def do_burst_fn(carry_b, enc_b, enc_opt_b, enc_tgt_b, val_b, val_tgt_b, pol_b, pol_tgt_b, rngs_b):
            carry_b, enc_b, enc_opt_b, enc_tgt_b, val_tgt_b, pol_tgt_b, rngs_b, enc_loss = encoder_burst(
                carry_b, enc_b, enc_opt_b, enc_tgt_b, val_b, val_tgt_b, pol_b, pol_tgt_b, rngs_b
            )
            return carry_b, enc_b, enc_opt_b, enc_tgt_b, val_tgt_b, pol_tgt_b, rngs_b, enc_loss

        def no_burst_fn(carry_b, enc_b, enc_opt_b, enc_tgt_b, val_b, val_tgt_b, pol_b, pol_tgt_b, rngs_b):
            return carry_b, enc_b, enc_opt_b, enc_tgt_b, val_tgt_b, pol_tgt_b, rngs_b, jnp.float32(0.0)

        (
            carry,
            encoder_,
            enc_opt_,
            encoder_target_,
            value_target_,
            policy_target_,
            rngs_,
            enc_loss,
        ) = nnx.cond(
            do_burst,
            do_burst_fn,
            no_burst_fn,
            carry,
            encoder_,
            enc_opt_,
            encoder_target_,
            value_,
            value_target_,
            policy_,
            policy_target_,
            rngs_,
        )

        # RL update (value + policy) happens every step when training
        def do_rl_fn(carry_r, enc_r, enc_tgt_r, val_r, val_opt_r, val_tgt_r, pol_r, pol_opt_r, pol_tgt_r, rngs_r):
            rng_rl = rngs_r()
            carry_r, v_loss, p_loss, mean_q = update_rl_step(
                carry_r, enc_r, enc_tgt_r, val_r, val_opt_r, val_tgt_r, pol_r, pol_opt_r, pol_tgt_r, rng_rl
            )
            return carry_r, val_r, val_opt_r, pol_r, pol_opt_r, rngs_r, v_loss, p_loss, mean_q

        def no_rl_fn(carry_r, enc_r, enc_tgt_r, val_r, val_opt_r, val_tgt_r, pol_r, pol_opt_r, pol_tgt_r, rngs_r):
            return carry_r, val_r, val_opt_r, pol_r, pol_opt_r, rngs_r, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)

        (
            carry,
            value_,
            val_opt_,
            policy_,
            pol_opt_,
            rngs_,
            v_loss,
            p_loss,
            mean_q,
        ) = nnx.cond(
            start_training,
            do_rl_fn,
            no_rl_fn,
            carry,
            encoder_,
            encoder_target_,
            value_,
            val_opt_,
            value_target_,
            policy_,
            pol_opt_,
            policy_target_,
            rngs_,
        )

        # Increment training steps (only when start_training)
        carry = carry._replace(
            training_steps=jnp.where(start_training, carry.training_steps + 1, carry.training_steps),
        )

        # Accumulate metrics (only when training)
        iter_metrics = {
            "encoder_loss": enc_loss,
            "value_loss": v_loss,
            "policy_loss": p_loss,
            "mean_q_value": mean_q,
            "reward_scale": carry.reward_scale,
        }
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, iter_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return (
            carry,
            encoder_,
            enc_opt_,
            encoder_target_,
            value_,
            val_opt_,
            value_target_,
            policy_,
            pol_opt_,
            policy_target_,
            rngs_,
        )

    # --- Eval callback (policy-only, no MPPI) ---
    def eval_callback(carry, encoder_, policy_, rngs_):
        def act_fn(obs, rng):
            zs_BS = encoder_.zs(obs)
            action_BA, _ = policy_.act(zs_BS)
            if discrete:
                return jnp.argmax(action_BA, axis=-1)
            return action_BA * max_action

        act = make_eval_act(act_fn, config, carry.obs_rms)
        return evaluate(act, rngs_(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = num_envs
    num_train_its_per_eval = int(np.ceil(config.eval_freq / steps_per_iter))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, encoder, policy, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _MRSQ_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, enc, enc_opt, enc_tgt, val, val_opt, val_tgt, pol, pol_opt, pol_tgt, rngs_ = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, enc, enc_opt, enc_tgt, val, val_opt, val_tgt, pol, pol_opt, pol_tgt, rngs_ = nnx.fori_loop(
            0,
            num_train_its_per_eval,
            train_body,
            (carry, enc, enc_opt, enc_tgt, val, val_opt, val_tgt, pol, pol_opt, pol_tgt, rngs_),
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, enc, pol, rngs_)
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

        return (carry, enc, enc_opt, enc_tgt, val, val_opt, val_tgt, pol, pol_opt, pol_tgt, rngs_), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (
        (carry, encoder, enc_opt, encoder_target, value, val_opt, value_target, policy, pol_opt, policy_target, rngs),
        all_metrics,
    ) = scan_eval(
        (carry, encoder, enc_opt, encoder_target, value, val_opt, value_target, policy, pol_opt, policy_target, rngs),
        jnp.zeros(num_evals),
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "encoder": encoder,
        "encoder_optimizer": enc_opt,
        "encoder_target": encoder_target,
        "value": value,
        "value_optimizer": val_opt,
        "value_target": value_target,
        "policy": policy,
        "policy_optimizer": pol_opt,
        "policy_target": policy_target,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
