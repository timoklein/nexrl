"""DrQ — Data-regularized Q, discrete Atari-100k variant — standalone training function.

"Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning
from Pixels" (Kostrikov et al., 2021)

Follows the authors' official discrete-control implementation (google/dopamine
labs/atari_100k, DrQ.gin) rather than the PyTorch continuous-control release:
Double + Dueling DQN, n-step-10 targets, and ONE random-shift + intensity
augmentation applied independently to the sampled obs and next_obs inside the
train step (no K/M target averaging — that is the DMC/SAC variant). The buffer
stores raw uint8 frames; action selection and evaluation see unaugmented frames.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (num_actions)
    C/H/W: image channels (stacked frames) / height / width
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
from nexrl.buffers import Minibatch, NStepState, ReplayBuffer, create_nstep_state
from nexrl.configs import DrQConfig
from nexrl.evaluate import evaluate
from nexrl.networks import DiscreteQNetwork, DuelingQNetwork
from nexrl.networks.sgf import random_intensity, random_shift
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_DRQ_METRIC_NAMES = ("td_loss", "mean_q_value", "epsilon")


class DrQCarry(NamedTuple):
    """Non-module carry state for DrQ training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    n_step_state: NStepState
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _anneal_epsilon(
    global_step: jax.Array,
    fill_buffer: int,
    decay_period: int,
    epsilon_train: float,
) -> jax.Array:
    """Linear epsilon decay that starts AFTER buffer fill (dopamine ``linearly_decaying_epsilon``).

    Holds epsilon = 1 for the first ``min_replay_history`` (== ``fill_buffer``) env
    steps, then decays 1 -> ``epsilon_train`` linearly over the next ``decay_period``
    env steps (DrQ.gin: min_replay_history 1600, epsilon_decay_period 5000,
    epsilon_train 0.1). During fill the action is uniform regardless, so the
    meaningful effect is the annealed exploration window after training begins.
    """
    decay_frac = jnp.clip((global_step - fill_buffer).astype(jnp.float32) / decay_period, 0.0, 1.0)
    return 1.0 - (1.0 - epsilon_train) * decay_frac


def _compute_nstep_transition(
    nss: NStepState,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    next_obs: jax.Array,
    n_step: int,
    gamma: float,
) -> tuple[NStepState, Minibatch, jax.Array]:
    """Write transition to n-step ring buffer and compute n-step returns when ready.

    Returns (updated_nss, n_step_minibatch, ready_mask_B).
    Non-ready entries get a dummy minibatch (reward is masked to 0 by the caller).
    """
    num_envs = obs.shape[0]
    w = nss.write_idx % n_step

    # Write new transition into ring buffer
    new_obs = nss.obs.at[w].set(obs)
    new_action = nss.action.at[w].set(action)
    new_reward = nss.reward.at[w].set(reward)
    new_done = nss.done.at[w].set(done)
    new_next_obs = nss.next_obs.at[w].set(next_obs)
    new_count = nss.count + 1

    # Reset count for envs that terminated in this step
    new_count = jnp.where(done, jnp.int32(1), new_count)

    nss = NStepState(
        obs=new_obs,
        action=new_action,
        reward=new_reward,
        done=new_done,
        next_obs=new_next_obs,
        write_idx=nss.write_idx + 1,
        count=new_count,  # pyright: ignore[reportArgumentType]
    )

    # Which envs have enough data for an n-step transition
    ready_B = new_count >= n_step  # pyright: ignore[reportOperatorIssue]

    # Compute n-step discounted return, truncating at episode boundaries
    # Read oldest entry first (write_idx points to next write, so oldest is write_idx - n_step + 1)
    # After increment, oldest is at (write_idx - n_step) % n_step = (nss.write_idx - n_step) % n_step
    oldest_idx = (nss.write_idx - n_step) % n_step

    # Gather n-step returns in chronological order
    indices = (oldest_idx + jnp.arange(n_step)) % n_step

    # n_step_obs is the obs from oldest step
    n_step_obs = nss.obs[indices[0]]  # (num_envs, *obs_shape)
    n_step_action = nss.action[indices[0]]  # (num_envs, *action_shape)

    # Compute discounted return and find effective next_obs
    # We iterate from oldest to newest, accumulating reward with truncation at done
    def _accumulate_return(carry, idx):
        """Scan over n-step window accumulating discounted rewards."""
        running_return, effective_next_obs, effective_done = carry
        r = nss.reward[idx]  # (num_envs,)
        d = nss.done[idx]  # (num_envs,)
        no = nss.next_obs[idx]  # (num_envs, *obs_shape)

        # If this step is terminal, truncate: return = running + r, done = True
        running_return = running_return * gamma * (1.0 - effective_done) + r
        # Update effective next_obs and done to latest non-truncated step
        effective_next_obs = jnp.where(effective_done.reshape((num_envs,) + (1,) * (no.ndim - 1)), effective_next_obs, no)
        effective_done = jnp.logical_or(effective_done, d)
        return (running_return, effective_next_obs, effective_done), None

    init_carry = (
        jnp.zeros(num_envs, dtype=jnp.float32),
        nss.next_obs[indices[-1]],  # default: newest next_obs
        jnp.zeros(num_envs, dtype=bool),
    )
    (n_step_return, n_step_next_obs, n_step_done), _ = jax.lax.scan(_accumulate_return, init_carry, indices)

    n_step_mb = Minibatch(
        obs=n_step_obs,
        action=n_step_action,
        reward=n_step_return,
        done=n_step_done,
        next_obs=n_step_next_obs,  # pyright: ignore[reportArgumentType]
    )

    return nss, n_step_mb, ready_B


def _create_networks(
    config: DrQConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[DiscreteQNetwork | DuelingQNetwork, nnx.Optimizer, DiscreteQNetwork | DuelingQNetwork]:
    """Create Q-network, optimizer, and Q-target."""
    agent_name = getattr(config, "agent", "DuelingQNetwork")
    agent_cls = {"QNetwork": DiscreteQNetwork, "DuelingQNetwork": DuelingQNetwork}[agent_name]

    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    q_kwargs = {"in_features": in_features, "action_dim": action_dim, **agent_kwargs}

    rngs = nnx.Rngs(rng)
    q_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    qt_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    q_network = agent_cls(**q_kwargs, encoder=q_encoder, rngs=rngs)
    q_target = agent_cls(**q_kwargs, encoder=qt_encoder, rngs=rngs)
    # Target starts as an exact copy of the online net (Dopamine syncs at construction).
    nnx.update(q_target, nnx.state(q_network, nnx.Param))

    # clip_by_global_norm, not optax.clip: the paper's Table 4 specifies "Max gradient
    # norm 10", which is a global-norm clip. optax.clip is a per-element VALUE clip and
    # would apply a different (weaker, elementwise) constraint at the same threshold.
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate, eps=config.adam_eps),
    )
    q_optimizer = nnx.Optimizer(q_network, tx, wrt=nnx.Param)

    return q_network, q_optimizer, q_target


def train_drq(
    config: DrQConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a DrQ agent (discrete, Dopamine labs/atari_100k variant).

    Double + Dueling DQN with n-step targets and random-shift + intensity image
    augmentation applied to sampled minibatches.

    Args:
        config: DrQConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, DrQConfig

        config = DrQConfig.from_yaml("configs/drq/atari/breakout.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("drq"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    # --- Setup (traced once) ---
    env, env_params = create_env(config, env, env_params)
    # Dedicated eval instances — sharing stateful Atari/DMC envs corrupts training (see make_eval_env).
    eval_env = make_eval_env(env, config.run_evaluation)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]
    assert discrete, "DrQ requires discrete action space"

    # n-step bootstrap discount, applied once at the window end (episode-internal
    # truncation is handled inside _compute_nstep_transition via the done mask).
    gamma_n = config.gamma**config.n_step

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    q_network, q_optimizer, q_target = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer + n-step ring (obs_dtype from the space keeps Atari frames uint8)
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)
    nss = create_nstep_state(
        n_step=config.n_step,
        num_envs=num_envs,
        obs_shape=obs_space.shape,
        action_shape=action_space.shape,
        obs_dtype=obs_space.dtype,
        action_dtype=jnp.int32,
    )

    zero_train_metrics = {k: jnp.float32(0.0) for k in _DRQ_METRIC_NAMES}

    carry = DrQCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        n_step_state=nss,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect transitions ---
    def collect_transitions(carry, q_net, epsilon, uniform, rngs):
        rng_action = rngs()

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs))

        def sample_policy(rng):
            last_obs = carry.last_obs
            if config.normalize_observations:
                last_obs = normalize_obs(carry.obs_rms, last_obs)
            return q_net.act(last_obs, rng, epsilon=epsilon)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, rewards, dones, info = vmap_step(rng_steps, carry.env_state, actions, env_params)

        # With terminal_on_life_loss, `dones` is the true GAME boundary (game over) — a
        # life loss is carried out-of-band in `info['life_terminal']` so it does NOT
        # trigger an env autoreset. `real_dones` (== game over) drives the reported
        # full-game train_return; the buffer/n-step terminal is `dones OR life_terminal`
        # (bootstrap cut at each life). Without the wrapper both info keys are absent, so
        # real_dones == dones and life_terminal is all-False — a no-op for other configs.
        real_dones = info.get("real_done", dones)
        life_terminals = info.get("life_terminal", jnp.zeros_like(dones))
        agent_dones = jnp.logical_or(dones, life_terminals)

        # Reward clipping for the TRAINING signal only (Dopamine clips the agent/buffer
        # reward to [-1, 1]); running_return uses the RAW reward so train_return stays
        # the true game score.
        train_rewards = jnp.clip(rewards, -1.0, 1.0) if config.clip_rewards else rewards

        # Track episode returns using raw rewards and full-game boundaries
        new_running_B = carry.running_return + rewards
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * real_dones)
        completed_count = carry.completed_returns_count + jnp.sum(real_dones.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - real_dones)

        obs_rms = carry.obs_rms
        rew_rms = carry.rew_rms
        if config.normalize_observations:
            obs_rms = update_rms(obs_rms, next_obs)
        if config.normalize_rewards:
            rew_rms = update_rew_rms(rew_rms, rewards, dones, config.gamma)

        # Accumulate into the n-step ring; the emitted transition is the one n steps back.
        # Non-ready rows (first n-1 steps of each life/episode) are still inserted with
        # reward masked to 0 — a negligible fraction of a uniform buffer (drqv2/rainbow
        # accept the same trade-off; Dopamine instead gates serving, which our fixed-shape
        # extend path cannot express).
        new_nss, nstep_mb, ready_B = _compute_nstep_transition(
            carry.n_step_state, carry.last_obs, actions, train_rewards, agent_dones, next_obs, config.n_step, config.gamma
        )
        masked_mb = Minibatch(
            obs=nstep_mb.obs,
            action=nstep_mb.action,
            reward=jnp.where(ready_B, nstep_mb.reward, 0.0),
            done=nstep_mb.done,
            next_obs=nstep_mb.next_obs,
        )

        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            n_step_state=new_nss,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry, masked_mb

    # --- Update Q-network ---
    def update_q(mb, q_net, q_opt, q_target, rngs):
        obs_B = mb.obs
        next_obs_B = mb.next_obs
        if config.data_augmentation:
            # Dopamine drq_image_augmentation: random shift then intensity, with
            # INDEPENDENT keys for obs and next_obs. Offsets are drawn from
            # [0, 2*pad) exclusive (== _per_image_random_crop). random_shift is
            # dtype-agnostic, so the crop runs precision-free on uint8; intensity
            # promotes uint8 * float32 -> float32, and the encoder's
            # `.astype(compute_dtype) / 255.0` afterwards is mathematically
            # identical to Dopamine's float-first order (scalar multiply commutes).
            max_offset = 2 * config.shift_pad
            rng_s1, rng_i1 = jax.random.split(rngs(), 2)
            rng_s2, rng_i2 = jax.random.split(rngs(), 2)
            obs_B = random_shift(rng_s1, obs_B, pad=config.shift_pad, max_offset=max_offset)
            obs_B = random_intensity(rng_i1, obs_B, scale=config.intensity_scale)
            next_obs_B = random_shift(rng_s2, next_obs_B, pad=config.shift_pad, max_offset=max_offset)
            next_obs_B = random_intensity(rng_i2, next_obs_B, scale=config.intensity_scale)

        # The single augmented next_obs feeds BOTH the DDQN argmax and the target
        # evaluation, matching Dopamine's train step.
        next_q_target_BA = q_target(next_obs_B)

        def vanilla_targets():
            return jnp.max(next_q_target_BA, axis=1)

        def ddqn_targets():
            next_q_BA = q_net(next_obs_B)
            next_action_B1 = jnp.argmax(next_q_BA, axis=1, keepdims=True)
            return jnp.take_along_axis(next_q_target_BA, next_action_B1, axis=1).squeeze(axis=1)

        next_q_B = jax.lax.cond(config.ddqn, ddqn_targets, vanilla_targets)
        # mb.reward is the discounted n-step sum (truncated at within-window terminals);
        # gamma_n applies the remaining bootstrap discount at the window end.
        target_B = mb.reward + jnp.logical_not(mb.done) * gamma_n * next_q_B

        def loss_fn(model):
            q_B = model.take(obs_B, mb.action)
            # Huber, not MSE: full_rainbow's non-distributional loss defaults to
            # losses.huber_loss (DrQ.gin does not set mse_loss).
            return optax.huber_loss(q_B, target_B, delta=1.0).mean(), q_B.mean()

        (loss, mean_q), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_net)
        q_opt.update(q_net, grads)
        return loss, mean_q

    # --- Train iteration ---
    def train_iteration(carry, q_network, q_optimizer, q_target, rngs):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        epsilon = _anneal_epsilon(carry.global_step, config.fill_buffer, config.epsilon_decay_period, config.epsilon_train)
        uniform = jnp.logical_not(start_training)
        carry, batch = collect_transitions(carry, q_network, epsilon, uniform, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        zero_update_metrics = {"td_loss": jnp.float32(0.0), "mean_q_value": jnp.float32(0.0)}

        def update_iteration(_, state):
            carry, q_net, q_opt, q_tgt, rngs, metrics_sum = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)
            td_loss, mean_q = update_q(minibatch, q_net, q_opt, q_tgt, rngs)
            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, {"td_loss": td_loss, "mean_q_value": mean_q})
            return (carry, q_net, q_opt, q_tgt, rngs, metrics_sum)

        def do_updates(carry, q_net, q_opt, q_tgt, rngs):
            carry, q_net, q_opt, q_tgt, rngs, metrics_sum = nnx.fori_loop(
                0, config.num_epochs, update_iteration, (carry, q_net, q_opt, q_tgt, rngs, zero_update_metrics)
            )
            iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry, q_net, q_opt, q_tgt, rngs, iter_metrics

        def no_updates(carry, q_net, q_opt, q_tgt, rngs):
            return carry, q_net, q_opt, q_tgt, rngs, zero_update_metrics

        carry, q_network, q_optimizer, q_target, rngs, update_metrics = nnx.cond(
            start_training, do_updates, no_updates, carry, q_network, q_optimizer, q_target, rngs
        )

        # Update target network. target_update_freq=1 + polyak=0.0 == hard copy every
        # env step (Dopamine target_update_period=1 at replay ratio 1, synced after the
        # gradient step). See DrQConfig for why polyak must stay 0.0.
        q_target = maybe_update_targets(
            q_network, q_target, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )

        # Accumulate metrics (only when training)
        iter_metrics = {**update_metrics, "epsilon": epsilon}
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, iter_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, q_network, q_optimizer, q_target, rngs

    # --- Eval callback ---
    def eval_callback(carry, q_net, rngs):
        act_fn = make_eval_act(lambda obs, rng: q_net.act(obs, rng, epsilon=config.epsilon_eval), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), eval_env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, q_network, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _DRQ_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, q_net, q_opt, q_tgt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, q_net, q_opt, q_tgt, rngs = nnx.fori_loop(0, steps_per_iter, train_body, (carry, q_net, q_opt, q_tgt, rngs))

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, q_net, rngs)
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

        return (carry, q_net, q_opt, q_tgt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, q_network, q_optimizer, q_target, rngs), all_metrics = scan_eval(
        (carry, q_network, q_optimizer, q_target, rngs), jnp.zeros(num_evals)
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "q_target": q_target,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
