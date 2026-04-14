"""Rainbow DQN — C51 + n-step returns + prioritized experience replay.

"Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2018)

Standalone training function mirroring dqn.py structure.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (num_actions)
    N: number of atoms (n_atoms)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import gymnax
import jax
import numpy as np
import optax
import rlax
from flax import nnx
from jax import numpy as jnp
from optax import linear_schedule

from nexrl.algos.utils import (
    RewardRMSState,
    RMSState,
    create_encoder,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    maybe_update_targets,
    normalize_minibatch,
    normalize_obs,
    resolve_network_kwargs,
    update_rew_rms,
    update_rms,
)
from nexrl.buffers import Minibatch, NStepState, PrioritizedReplayBuffer, create_nstep_state
from nexrl.configs import RainbowConfig
from nexrl.evaluate import evaluate
from nexrl.networks import CategoricalQNetwork
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_RAINBOW_METRIC_NAMES = ("cross_entropy_loss", "mean_q_value", "epsilon", "mean_priority")


class RainbowCarry(NamedTuple):
    """Non-module carry state for Rainbow training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: PrioritizedReplayBuffer
    n_step_state: NStepState
    max_priority: chex.Array
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


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
    Non-ready entries get a dummy minibatch (never sampled due to priority=0).
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
        effective_next_obs = jnp.where(effective_done[:, None], effective_next_obs, no)
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
    config: RainbowConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[CategoricalQNetwork, nnx.Optimizer, CategoricalQNetwork]:
    """Create C51 Q-network, optimizer, and target network."""
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    q_kwargs = {
        "in_features": in_features,
        "action_dim": action_dim,
        "n_atoms": config.n_atoms,
        "v_min": config.v_min,
        "v_max": config.v_max,
        **agent_kwargs,
    }

    rngs = nnx.Rngs(rng)
    q_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    qt_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    q_network = CategoricalQNetwork(**q_kwargs, encoder=q_encoder, rngs=rngs)
    q_target = CategoricalQNetwork(**q_kwargs, encoder=qt_encoder, rngs=rngs)

    tx = optax.chain(
        optax.clip(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    q_optimizer = nnx.Optimizer(q_network, tx, wrt=nnx.Param)

    return q_network, q_optimizer, q_target


def train_rainbow(
    config: RainbowConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a Rainbow DQN agent.

    Rainbow DQN combining C51, n-step returns, and prioritized replay.

    Args:
        config: RainbowConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, RainbowConfig

        config = RainbowConfig.from_yaml("configs/rainbow/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("rainbow"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    # --- Setup (traced once) ---
    env, env_params = create_env(config, env, env_params)

    eps_schedule = linear_schedule(
        config.eps_start,
        config.eps_end,
        int(config.exploration_fraction * config.total_timesteps),
    )

    beta_schedule = linear_schedule(
        config.per_beta_start,
        config.per_beta_end,
        config.total_timesteps,
    )

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]
    assert discrete, "Rainbow requires discrete action space"

    n_step = config.n_step
    gamma_n = config.gamma**n_step

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    q_network, q_optimizer, q_target = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = PrioritizedReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    # Init n-step state
    nss = create_nstep_state(n_step, num_envs, obs_space.shape, action_space.shape)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _RAINBOW_METRIC_NAMES}

    carry = RainbowCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        n_step_state=nss,
        max_priority=jnp.float32(1.0),
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
        next_obs, new_env_state, rewards, dones, _ = vmap_step(rng_steps, carry.env_state, actions, env_params)

        # Track episode returns using raw rewards
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

        # Route through n-step buffer
        nss, n_step_mb, ready_B = _compute_nstep_transition(
            carry.n_step_state, carry.last_obs, actions, rewards, dones, next_obs, n_step, config.gamma
        )

        # Non-ready entries get priority=0 (never sampled)
        priorities_B = jnp.where(ready_B, carry.max_priority**config.per_alpha, jnp.float32(0.0))

        # Extend buffer
        new_buf = carry.replay_buffer.extend_with_priorities(n_step_mb, priorities_B)

        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            replay_buffer=new_buf,
            n_step_state=nss,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry

    # --- Update Q-network (C51 distributional) ---
    def update_q(mb, q_net, q_opt, q_tgt, weights_B):
        support_N = q_net.support

        # DDQN: online net picks best action, target net provides distribution
        next_q_BA = q_net.q_values(mb.next_obs)
        next_action_B = jnp.argmax(next_q_BA, axis=1)

        target_probs_BN = q_tgt.take_probs(mb.next_obs, next_action_B)

        # Target support: r + gamma^n * (1 - done) * z_i
        target_support_BN = mb.reward[:, None] + gamma_n * (1.0 - mb.done[:, None].astype(jnp.float32)) * support_N[None, :]

        # Project target distribution (L2 CDF projection onto fixed support)
        projected_BN = jnp.asarray(
            jax.vmap(rlax.categorical_l2_project, in_axes=(0, 0, None))(target_support_BN, target_probs_BN, support_N)
        )
        projected_BN = jax.lax.stop_gradient(projected_BN)

        def loss_fn(model):
            log_probs_BN = model.take_logprobs(mb.obs, mb.action)
            # Cross-entropy: -sum(target * log_pred), per sample
            # Use jnp.where to avoid 0 * -inf = NaN when projected prob is zero
            safe_log_BN = jnp.where(projected_BN > 0, log_probs_BN, 0.0)
            ce_B = -jnp.sum(projected_BN * safe_log_BN, axis=-1)
            # Weighted by PER importance sampling weights
            loss = jnp.mean(ce_B * weights_B)
            q_BA = model.q_values(mb.obs)
            mean_q = q_BA.mean()
            return loss, (mean_q, ce_B)

        (loss, (mean_q, ce_B)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_net)
        q_opt.update(q_net, grads)

        # New priorities: CE loss + epsilon
        new_priorities_B = ce_B + config.per_eps
        return loss, mean_q, new_priorities_B

    # --- Train iteration ---
    def train_iteration(carry, q_network, q_optimizer, q_target, rngs):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        epsilon = eps_schedule(carry.global_step)
        uniform = jnp.logical_not(start_training)
        carry = collect_transitions(carry, q_network, epsilon, uniform, rngs)

        zero_update_metrics = {
            "cross_entropy_loss": jnp.float32(0.0),
            "mean_q_value": jnp.float32(0.0),
            "mean_priority": jnp.float32(0.0),
        }

        def update_iteration(_, state):
            carry, q_net, q_opt, q_tgt, rngs, metrics_sum = state
            beta = beta_schedule(carry.global_step)
            mb, indices, weights = carry.replay_buffer.sample(config.batch_size, rngs(), beta)
            mb = normalize_minibatch(mb, config, carry.obs_rms, carry.rew_rms)
            ce_loss, mean_q, new_priorities_B = update_q(mb, q_net, q_opt, q_tgt, weights)
            # Update priorities
            new_buf = carry.replay_buffer.update_priorities(indices, new_priorities_B**config.per_alpha)
            new_max_priority = jnp.maximum(carry.max_priority, new_priorities_B.max())
            carry = carry._replace(replay_buffer=new_buf, max_priority=new_max_priority)
            iter_m = {"cross_entropy_loss": ce_loss, "mean_q_value": mean_q, "mean_priority": new_priorities_B.mean()}
            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, iter_m)
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

        # Update target network
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
        act_fn = make_eval_act(lambda obs, rng: q_net.act(obs, rng, epsilon=0.005), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

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
            **{k: jnp.float32(0.0) for k in _RAINBOW_METRIC_NAMES},
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
