"""DrQ-v2 — Data-Regularized Q-v2 for visual continuous control.

"Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning"
(Yarats et al., 2022)

DDPG-style pixel-based RL with random shift augmentation, n-step returns,
scheduled exploration noise, and a learned CNN encoder.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    C: input channels (frame_stack * 3 for RGB)
    H: image height (84)
    W: image width (84)
    R: repr_dim (39200 for 84x84 input)
    F: feature_dim (50)
    A: action dimension
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
    _polyak_update,
    make_eval_env,
)
from nexrl.buffers import Minibatch, NStepState, ReplayBuffer, create_nstep_state
from nexrl.compat import create
from nexrl.configs import DrQV2Config
from nexrl.evaluate import evaluate
from nexrl.networks.drqv2 import DrQV2Actor, DrQV2Critic, DrQV2Encoder
from nexrl.networks.sgf import random_shift
from nexrl.types import Metrics, TrainState


_DRQV2_METRIC_NAMES = ("actor_loss", "critic_loss", "mean_q_value")

NUM_EVAL_SEEDS = 10  # DMC callback is sequential; 128 would be too slow


class DrQV2Carry(NamedTuple):
    """Non-module carry state for DrQ-v2 training."""

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


def _noise_schedule(step: jax.Array, init: float, final: float, schedule_steps: int) -> jax.Array:
    """Linear noise decay from init to final over schedule_steps."""
    t = jnp.minimum(step / schedule_steps, 1.0)
    return final + (1.0 - t) * (init - final)


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
    Non-ready entries get a dummy minibatch (never sampled due to ready_mask=False).
    """
    num_envs = obs.shape[0]
    obs_shape = obs.shape[1:]
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

    oldest_idx = (nss.write_idx - n_step) % n_step
    indices = (oldest_idx + jnp.arange(n_step)) % n_step

    n_step_obs = nss.obs[indices[0]]
    n_step_action = nss.action[indices[0]]

    def _accumulate_return(carry, idx):
        running_return, effective_next_obs, effective_done = carry
        r = nss.reward[idx]
        d = nss.done[idx]
        no = nss.next_obs[idx]

        running_return = running_return * gamma * (1.0 - effective_done) + r
        # Broadcast done for image obs
        effective_next_obs = jnp.where(
            effective_done.reshape(num_envs, *((1,) * len(obs_shape))),
            effective_next_obs,
            no,
        )
        effective_done = jnp.logical_or(effective_done, d)
        return (running_return, effective_next_obs, effective_done), None

    init_carry = (
        jnp.zeros(num_envs, dtype=jnp.float32),
        nss.next_obs[indices[-1]],
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
    config: DrQV2Config, in_channels: int, action_dim: int, rng: jax.Array
) -> tuple[
    DrQV2Encoder,
    nnx.Optimizer,
    DrQV2Actor,
    nnx.Optimizer,
    DrQV2Critic,
    nnx.Optimizer,
    DrQV2Critic,
]:
    """Create encoder, actor, critic, their optimizers, and critic target."""
    rngs = nnx.Rngs(rng)
    # repr_dim for 84x84 with strides [2,1,1,1]: 32*35*35 = 39200
    repr_dim = 32 * 35 * 35

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )

    encoder = DrQV2Encoder(in_channels=in_channels, rngs=rngs)
    encoder_opt = nnx.Optimizer(encoder, tx, wrt=nnx.Param)

    actor = DrQV2Actor(
        repr_dim=repr_dim,
        feature_dim=config.feature_dim,
        action_dim=action_dim,
        hidden_layer_sizes=config.hidden_layer_sizes,
        rngs=rngs,
    )
    actor_opt = nnx.Optimizer(actor, tx, wrt=nnx.Param)

    critic = DrQV2Critic(
        repr_dim=repr_dim,
        feature_dim=config.feature_dim,
        action_dim=action_dim,
        hidden_layer_sizes=config.hidden_layer_sizes,
        rngs=rngs,
    )
    critic_opt = nnx.Optimizer(critic, tx, wrt=nnx.Param)

    critic_target = DrQV2Critic(
        repr_dim=repr_dim,
        feature_dim=config.feature_dim,
        action_dim=action_dim,
        hidden_layer_sizes=config.hidden_layer_sizes,
        rngs=rngs,
    )
    nnx.update(critic_target, nnx.state(critic, nnx.Param))

    return encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target


def train_drqv2(
    config: DrQV2Config,
    rng: jax.Array,
    *,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a DrQ-v2 agent.

    DrQ-v2 for visual continuous control with random shift augmentation.

    Args:
        config: DrQV2Config with hyperparameters and environment specification.
        rng: PRNG key for initialization and stochasticity. Use
            ``jax.vmap`` over different keys for independent seeds.
        log_fn: Optional callback invoked at each evaluation point via
            ``jax.debug.callback``. Receives ``(step, metrics_dict)``.

    Returns:
        A ``(state, metrics)`` tuple where ``state`` is a dict of trained
        parameters and ``metrics`` is a dict of JAX arrays with keys
        including ``eval_returns``, ``eval_lengths``, and ``global_step``.

    Example::

        from nexrl import get_train_fn, DrQV2Config

        config = DrQV2Config.from_yaml("configs/drqv2/dm_control/cartpole_balance.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("drqv2"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()

    # --- Setup ---
    env, env_params = create(
        config.env,
        num_envs=config.num_envs,
        pixels=True,
        rgb=True,
        action_repeat=config.action_repeat,
        frame_stack=config.frame_stack,
    )
    # Dedicated eval instances — DrQ-v2 always runs on DMC, whose multi-instance adapter is
    # stateful, so sharing would destroy in-progress training episodes at every eval point.
    # Keep passing the *training* `env_params`: create_dmc returns action-repeat-adjusted
    # params that the env object itself does not carry (its `_max_steps` stays None), so
    # re-deriving them from the clone would silently drop the action-repeat cap.
    eval_env = make_eval_env(env, config.run_evaluation)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    action_dim = int(np.prod(action_space.shape))
    action_low = action_space.low
    action_high = action_space.high
    in_channels = obs_space.shape[0]
    n_step = config.n_step
    gamma_n = config.gamma**n_step

    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))

    # Create networks
    rngs = nnx.Rngs(rng)
    encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target = _create_networks(
        config, in_channels, action_dim, rngs()
    )

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization (unused but kept for carry compatibility)
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    # Init n-step state
    nss = create_nstep_state(
        n_step=n_step,
        num_envs=num_envs,
        obs_shape=obs_space.shape,
        action_shape=action_space.shape,
        obs_dtype=obs_space.dtype,
        action_dtype=jnp.float32,
    )

    zero_train_metrics = {k: jnp.float32(0.0) for k in _DRQV2_METRIC_NAMES}

    carry = DrQV2Carry(
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
        running_return=jnp.zeros(num_envs, dtype=jnp.float32),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect one env step ---
    def collect_step(carry, encoder, actor, rngs):
        rng_action, rng_step = rngs(), rngs()

        seed_phase = carry.global_step < config.seed_frames

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs)).astype(jnp.float32)

        def sample_policy(rng):
            features_BR = encoder(carry.last_obs)
            action_BA = actor(features_BR)
            stddev = _noise_schedule(carry.global_step, config.stddev_init, config.stddev_final, config.stddev_schedule_steps)
            noise_BA = jnp.clip(
                stddev * jax.random.normal(rng, action_BA.shape),
                -config.noise_clip,
                config.noise_clip,
            )
            return jnp.clip(action_BA + noise_BA, action_low, action_high)

        actions_BA = jax.lax.cond(seed_phase, sample_uniform, sample_policy, rng_action)

        rng_steps = jax.random.split(rng_step, num_envs)
        next_obs, new_env_state, rewards, dones, _ = vmap_step(rng_steps, carry.env_state, actions_BA, env_params)

        # Track episode returns
        new_running_B = carry.running_return + rewards
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * dones)
        completed_count = carry.completed_returns_count + jnp.sum(dones.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - dones)

        # N-step return computation
        new_nss, nstep_mb, ready_B = _compute_nstep_transition(
            carry.n_step_state, carry.last_obs, actions_BA, rewards, dones, next_obs, n_step, config.gamma
        )

        # Add ready transitions to replay buffer
        buf = carry.replay_buffer
        # For num_envs=1, extend with the single transition (masked by ready)
        # Mask reward to 0 for non-ready entries so they have minimal impact if sampled
        masked_mb = Minibatch(
            obs=nstep_mb.obs,
            action=nstep_mb.action,
            reward=jnp.where(ready_B, nstep_mb.reward, 0.0),
            done=nstep_mb.done,
            next_obs=nstep_mb.next_obs,
        )
        buf = buf.extend(masked_mb)

        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            replay_buffer=buf,
            n_step_state=new_nss,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry

    # --- Update networks ---
    def update_networks(carry, encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target, rngs):
        rng_aug, rng_sample = rngs(), rngs()
        rng_aug1, rng_aug2 = jax.random.split(rng_aug)

        mb = carry.replay_buffer.sample(config.batch_size, rng_sample)

        # Augment observations
        aug_obs = random_shift(rng_aug1, mb.obs, config.shift_pad)
        aug_next_obs = random_shift(rng_aug2, mb.next_obs, config.shift_pad)

        # --- Critic loss (encoder gets gradients through obs encoding) ---
        # Compute target Q outside of grad to avoid differentiating through target
        next_feat_BR = jax.lax.stop_gradient(encoder(aug_next_obs))
        next_action_BA = jax.lax.stop_gradient(actor(next_feat_BR))
        tq1_B, tq2_B = critic_target(next_feat_BR, next_action_BA)
        target_B = mb.reward + (1.0 - mb.done) * gamma_n * jnp.minimum(tq1_B, tq2_B)
        target_B = jax.lax.stop_gradient(target_B)

        def critic_loss_fn(encoder, critic):
            feat_BR = encoder(aug_obs)
            q1_B, q2_B = critic(feat_BR, mb.action)
            loss = optax.l2_loss(q1_B, target_B).mean() + optax.l2_loss(q2_B, target_B).mean()
            mean_q = q1_B.mean()
            return loss, mean_q

        (critic_loss, mean_q), (enc_grads, crit_grads) = nnx.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
            encoder, critic
        )
        encoder_opt.update(encoder, enc_grads)
        critic_opt.update(critic, crit_grads)

        # --- Actor loss (encoder is stop_gradient'd) ---
        # Reuse aug_obs features with stop_gradient on encoder
        sg_feat_BR = jax.lax.stop_gradient(encoder(aug_obs))

        def actor_loss_fn(actor):
            action_BA = actor(sg_feat_BR)
            q1_B, q2_B = critic(sg_feat_BR, action_BA)
            return -jnp.minimum(q1_B, q2_B).mean()

        actor_loss, actor_grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_opt.update(actor, actor_grads)

        # --- Target update ---
        _polyak_update(critic, critic_target, config.polyak)

        metrics = {"actor_loss": actor_loss, "critic_loss": critic_loss, "mean_q_value": mean_q}
        return encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target, metrics

    # --- Train iteration (1 env step + conditional update) ---
    def train_iteration(carry, encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target, rngs):
        carry = collect_step(carry, encoder, actor, rngs)

        start_training = carry.global_step >= config.seed_frames
        do_update = jnp.logical_and(start_training, carry.global_step % config.update_every_steps == 0)

        zero_metrics = {k: jnp.float32(0.0) for k in _DRQV2_METRIC_NAMES}

        def do_updates(enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs):
            def update_body(_, state):
                enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs, metrics_sum = state
                enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, step_metrics = update_networks(
                    carry, enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs
                )
                metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, step_metrics)
                return (enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs, metrics_sum)

            enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs, metrics_sum = nnx.fori_loop(
                0,
                config.num_epochs,
                update_body,
                (enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs, zero_metrics),
            )
            avg_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs, avg_metrics

        def no_updates(enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs):
            return enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs, zero_metrics

        encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target, rngs, iter_metrics = nnx.cond(
            do_update,
            do_updates,
            no_updates,
            encoder,
            encoder_opt,
            actor,
            actor_opt,
            critic,
            critic_opt,
            critic_target,
            rngs,
        )

        # Accumulate metrics only when updating
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(do_update, s + m, s), carry.train_metrics, iter_metrics),
            metrics_count=jnp.where(do_update, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target, rngs

    # --- Eval callback ---
    def eval_callback(encoder, actor, rngs):
        def act_fn(obs, rng):
            obs = jnp.expand_dims(obs, 0)
            features = encoder(obs)
            return jnp.squeeze(actor(features))

        return evaluate(act_fn, rngs(), eval_env, env_params, NUM_EVAL_SEEDS, max_steps)

    # --- Outer training loop ---
    num_train_its_per_eval = config.eval_freq
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(encoder, actor, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _DRQV2_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs = nnx.fori_loop(
            0,
            num_train_its_per_eval,
            train_body,
            (carry, enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs),
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(enc, act, rngs)
        else:
            eval_lengths = jnp.zeros(NUM_EVAL_SEEDS, dtype=jnp.float32)
            eval_returns = jnp.zeros(NUM_EVAL_SEEDS, dtype=jnp.float32)

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

        return (carry, enc, enc_opt, act, act_opt, crt, crt_opt, crt_tgt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target, rngs), all_metrics = scan_eval(
        (carry, encoder, encoder_opt, actor, actor_opt, critic, critic_opt, critic_target, rngs),
        jnp.zeros(num_evals, dtype=jnp.float32),
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "encoder": encoder,
        "encoder_optimizer": encoder_opt,
        "actor": actor,
        "actor_optimizer": actor_opt,
        "critic": critic,
        "critic_optimizer": critic_opt,
        "critic_target": critic_target,
        "global_step": carry.global_step,
    }
    return state, all_metrics
