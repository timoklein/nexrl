"""Twin Delayed Deep Deterministic Policy Gradient (TD3) — standalone training function.

"Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (continuous action_dim)
    C: num critics (2)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
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
    make_eval_env,
    maybe_update_targets,
    normalize_minibatch,
    normalize_obs,
    resolve_network_kwargs,
    update_rew_rms,
    update_rms,
)
from nexrl.buffers import Minibatch, ReplayBuffer
from nexrl.configs import TD3Config
from nexrl.evaluate import evaluate
from nexrl.networks import DeterministicPolicy, QNetwork
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


NUM_CRITICS = 2

_TD3_METRIC_NAMES = ("actor_loss", "critic_loss", "mean_q_value")


class TD3Carry(NamedTuple):
    """Non-module carry state for TD3 training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _create_networks(
    config: TD3Config, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[DeterministicPolicy, nnx.Optimizer, DeterministicPolicy, list[QNetwork], list[nnx.Optimizer], list[QNetwork]]:
    """Create actor, actor_optimizer, actor_target, critics, critic_optimizers, and critic_targets."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    actor_kwargs = resolve_network_kwargs(getattr(config, "actor_kwargs", None))
    actor_kwargs = {
        "in_features": in_features,
        "action_dim": action_dim,
        "action_range": action_range,
        **actor_kwargs,
    }

    critic_kwargs = resolve_network_kwargs(getattr(config, "critic_kwargs", None))
    critic_kwargs = {
        "obs_dim": in_features,
        "action_dim": action_dim,
        **critic_kwargs,
    }

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )

    rngs = nnx.Rngs(rng)

    # Actor + actor target
    actor = DeterministicPolicy(**actor_kwargs, rngs=rngs)
    actor_optimizer = nnx.Optimizer(actor, tx, wrt=nnx.Param)

    actor_target = DeterministicPolicy(**actor_kwargs, rngs=rngs)
    nnx.update(actor_target, nnx.state(actor, nnx.Param))

    # Critics + critic targets
    critics = []
    critic_opts = []
    critic_targets = []
    for _ in range(NUM_CRITICS):
        critic = QNetwork(**critic_kwargs, rngs=rngs)
        critics.append(critic)
        critic_opts.append(nnx.Optimizer(critic, tx, wrt=nnx.Param))

        target = QNetwork(**critic_kwargs, rngs=rngs)
        nnx.update(target, nnx.state(critic, nnx.Param))
        critic_targets.append(target)

    return actor, actor_optimizer, actor_target, critics, critic_opts, critic_targets


def train_td3(
    config: TD3Config,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a TD3 agent.

    Twin Delayed DDPG with target policy smoothing and delayed policy updates.

    Args:
        config: TD3Config with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, TD3Config

        config = TD3Config.from_yaml("configs/td3/gymnax/pendulum.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("td3"), in_axes=(None, 0)),
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
    max_steps = env_params.max_steps_in_episode
    action_low = action_space.low
    action_high = action_space.high

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    actor, actor_optimizer, actor_target, critics, critic_opts, critic_targets = _create_networks(
        config, env, env_params, rngs()
    )

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _TD3_METRIC_NAMES}

    carry = TD3Carry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect transitions ---
    def collect_transitions(carry, actor, uniform, rngs):
        rng_action = rngs()

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs))

        def sample_policy(rng):
            last_obs = carry.last_obs
            if config.normalize_observations:
                last_obs = normalize_obs(carry.obs_rms, last_obs)
            action_BA = actor(last_obs)
            noise_BA = config.exploration_noise * jax.random.normal(rng, action_BA.shape)
            return jnp.clip(action_BA + noise_BA, action_low, action_high)

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

        mb = Minibatch(obs=carry.last_obs, action=actions, reward=rewards, next_obs=next_obs, done=dones)
        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry, mb

    # --- Update critics ---
    def update_critics(mb, actor_target, critics, critic_opts, critic_targets, rngs):
        next_action_BA = actor_target(mb.next_obs)
        noise_BA = jnp.clip(
            config.target_noise * jax.random.normal(rngs(), next_action_BA.shape),
            -config.target_noise_clip,
            config.target_noise_clip,
        )
        next_action_BA = jnp.clip(next_action_BA + noise_BA, action_low, action_high)

        qs_target_CB = jnp.stack([critic(mb.next_obs, next_action_BA) for critic in critic_targets])  # (C, B)
        q_target_B = jnp.min(qs_target_CB, axis=0)
        target_B = mb.reward + (1 - mb.done) * config.gamma * q_target_B

        # Mean Q-value for logging (before updates)
        mean_q = critics[0](mb.obs, mb.action).mean()

        total_critic_loss = jnp.float32(0.0)
        for critic_model, critic_opt in zip(critics, critic_opts):

            def critic_loss_fn(critic_model):
                q_B = critic_model(mb.obs, mb.action)
                return optax.l2_loss(q_B, target_B).mean()

            loss, grads = nnx.value_and_grad(critic_loss_fn)(critic_model)
            critic_opt.update(critic_model, grads)
            total_critic_loss = total_critic_loss + loss
        return total_critic_loss / NUM_CRITICS, mean_q

    # --- Update actor ---
    def update_actor(mb, actor, actor_optimizer, critics):

        def actor_loss_fn(actor_model):
            action_BA = actor_model(mb.obs)
            q_B = critics[0](mb.obs, action_BA)
            return -q_B.mean()

        loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_optimizer.update(actor, grads)
        return loss

    # --- Train critic phase (called policy_delay times) ---
    def train_critic_phase(carry, actor_model, actor_target, critics, critic_opts, critic_targets, placeholder_mb, rngs):
        """Collect transitions + update critics. Returns stored minibatches for later actor update."""
        start_training = carry.global_step > config.fill_buffer
        uniform = jnp.logical_not(start_training)

        carry, transitions = collect_transitions(carry, actor_model, uniform, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(transitions))

        zero_critic_metrics = {"critic_loss": jnp.float32(0.0), "mean_q_value": jnp.float32(0.0)}

        # actor_target is read-only in update_critics
        def update_body(i, state):
            carry, crs, c_opts, c_tgts, minibatches, rngs, metrics_sum = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)
            critic_loss, mean_q = update_critics(minibatch, actor_target, crs, c_opts, c_tgts, rngs)
            minibatches = jax.tree.map(lambda mb_all, m: mb_all.at[i].set(m), minibatches, minibatch)
            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, {"critic_loss": critic_loss, "mean_q_value": mean_q})
            return (carry, crs, c_opts, c_tgts, minibatches, rngs, metrics_sum)

        def do_updates(carry, crs, c_opts, c_tgts, rngs):
            carry, crs, c_opts, c_tgts, mbs, rngs, metrics_sum = nnx.fori_loop(
                0, config.num_epochs, update_body, (carry, crs, c_opts, c_tgts, placeholder_mb, rngs, zero_critic_metrics)
            )
            critic_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry, crs, c_opts, c_tgts, mbs, rngs, critic_metrics

        def no_updates(carry, crs, c_opts, c_tgts, rngs):
            return carry, crs, c_opts, c_tgts, placeholder_mb, rngs, zero_critic_metrics

        carry, critics, critic_opts, critic_targets, minibatches, rngs, critic_metrics = nnx.cond(
            start_training, do_updates, no_updates, carry, critics, critic_opts, critic_targets, rngs
        )
        return carry, critics, critic_opts, critic_targets, minibatches, rngs, critic_metrics

    # --- Train iteration ---
    def train_iteration(carry, actor, actor_optimizer, actor_target, critics, critic_opts, critic_targets, rngs):
        old_global_step = carry.global_step

        # Create placeholder minibatch for index-based stacking
        placeholder_mb = jax.tree.map(
            lambda sdstr: jnp.empty((config.num_epochs, *sdstr.shape), sdstr.dtype),
            carry.replay_buffer.sample(config.batch_size, jax.random.PRNGKey(0)),
        )

        zero_critic_metrics = {"critic_loss": jnp.float32(0.0), "mean_q_value": jnp.float32(0.0)}

        # actor and actor_target are read-only inside critic phase
        def policy_delay_body(_, state):
            carry, minibatches, crs, c_opts, c_tgts, rngs, critic_metrics_sum = state
            carry, crs, c_opts, c_tgts, minibatches, rngs, phase_critic_metrics = train_critic_phase(
                carry, actor, actor_target, crs, c_opts, c_tgts, placeholder_mb, rngs
            )
            critic_metrics_sum = jax.tree.map(lambda s, m: s + m, critic_metrics_sum, phase_critic_metrics)
            return (carry, minibatches, crs, c_opts, c_tgts, rngs, critic_metrics_sum)

        carry, minibatches, critics, critic_opts, critic_targets, rngs, critic_metrics_sum = nnx.fori_loop(
            0,
            config.policy_delay,
            policy_delay_body,
            (carry, placeholder_mb, critics, critic_opts, critic_targets, rngs, zero_critic_metrics),
        )
        critic_metrics = jax.tree.map(lambda s: s / config.policy_delay, critic_metrics_sum)

        # Update actor using stored minibatches
        start_training = carry.global_step > config.fill_buffer

        def do_actor_updates(actor_model, actor_opt):
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def scan_update(actor_state, mb):
                actor_model, actor_opt = actor_state
                actor_loss = update_actor(mb, actor_model, actor_opt, critics)
                return (actor_model, actor_opt), actor_loss

            (actor_model, actor_opt), actor_losses = scan_update((actor_model, actor_opt), minibatches)
            return actor_model, actor_opt, actor_losses.mean()

        def no_actor_updates(actor_model, actor_opt):
            return actor_model, actor_opt, jnp.float32(0.0)

        actor, actor_optimizer, actor_loss = nnx.cond(
            start_training, do_actor_updates, no_actor_updates, actor, actor_optimizer
        )

        # Update target networks
        online_models = [actor, *critics]
        target_models = [actor_target, *critic_targets]
        updated_targets: list[nnx.Module] = maybe_update_targets(  # type: ignore[assignment]
            online_models, target_models, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )
        actor_target = updated_targets[0]  # type: ignore[assignment]
        critic_targets = updated_targets[1:]  # type: ignore[assignment]

        # Accumulate metrics (only when training)
        iter_metrics = {"actor_loss": actor_loss, **critic_metrics}
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, iter_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, actor, actor_optimizer, actor_target, critics, critic_opts, critic_targets, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor(obs), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), eval_env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_train_it = num_envs * config.policy_delay
    num_train_its_per_eval = int(np.ceil(config.eval_freq / steps_per_train_it))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, actor, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _TD3_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, act, actor_opt, actor_tgt, crs, c_opts, c_tgts, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, act, actor_opt, actor_tgt, crs, c_opts, c_tgts, rngs = nnx.fori_loop(
            0, num_train_its_per_eval, train_body, (carry, act, actor_opt, actor_tgt, crs, c_opts, c_tgts, rngs)
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

        return (carry, act, actor_opt, actor_tgt, crs, c_opts, c_tgts, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor, actor_optimizer, actor_target, critics, critic_opts, critic_targets, rngs), all_metrics = scan_eval(
        (carry, actor, actor_optimizer, actor_target, critics, critic_opts, critic_targets, rngs), jnp.zeros(num_evals)
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
        "actor_target": actor_target,
        "critics": critics,
        "critic_optimizers": critic_opts,
        "critic_targets": critic_targets,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
