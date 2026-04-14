"""PPO with Intrinsic Curiosity Module (ICM) exploration.

"Curiosity-driven Exploration by Self-Supervised Prediction" (Pathak et al., 2017)

Extends unified PPO with dual value heads (extrinsic + intrinsic), ICM-based
intrinsic reward (forward prediction error), and separate GAE for
episodic extrinsic / non-episodic intrinsic.

Discrete actions only (one-hot in forward model, cross-entropy in inverse loss).

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    T: trajectory length (num_steps)
    L: ICM latent dimension
    A: number of discrete actions
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import gymnax
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
    ppo_lr_schedule,
    resolve_network_kwargs,
    shuffle_and_split,
    update_and_normalize_obs,
    update_and_normalize_rew,
    update_rms,
)
from nexrl.configs import PPOICMConfig
from nexrl.evaluate import evaluate
from nexrl.networks import (
    ActorCritic,
    DiscretePolicy,
    ICMModule,
    SharedDiscreteActorCritic,
    VNetwork,
)
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


class ICMTrajectory(struct.PyTreeNode):
    """Trajectory data with extrinsic and intrinsic reward/value fields."""

    obs: chex.Array
    next_obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array  # extrinsic
    value: chex.Array  # extrinsic
    intrinsic_reward: chex.Array
    intrinsic_value: chex.Array
    done: chex.Array


class ICMAdvantageMinibatch(struct.PyTreeNode):
    """Minibatch with dual advantages for extrinsic/intrinsic streams."""

    trajectories: ICMTrajectory
    advantages_ext: chex.Array
    targets_ext: chex.Array
    advantages_int: chex.Array
    targets_int: chex.Array


class PPOICMSystem(nnx.Module):
    """Container for all PPO-ICM networks.

    A single ``nnx.Optimizer(system, tx, wrt=nnx.Param)`` updates
    ac + intrinsic_critic + icm (all params are trainable).
    """

    def __init__(self, ac: nnx.Module, intrinsic_critic: VNetwork, icm: ICMModule):
        self.ac = ac
        self.intrinsic_critic = intrinsic_critic
        self.icm = icm


class PPOICMCarry(NamedTuple):
    """Non-module carry state for PPO-ICM training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    obs_rms: RMSState
    rew_rms: RewardRMSState
    intrinsic_rew_rms: RMSState
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _create_networks(
    config: PPOICMConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[PPOICMSystem, nnx.Optimizer]:
    """Create PPO-ICM system (ac + intrinsic critic + ICM) and optimizer."""
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    rngs = nnx.Rngs(rng)

    # ICM requires discrete actions
    assert isinstance(action_space, gymnax.environments.spaces.Discrete), "PPO-ICM only supports discrete action spaces"  # pyright: ignore[reportAttributeAccessIssue]
    num_actions = action_space.n

    # Actor-critic (same as PPO)
    if config.shared_backbone:
        assert config.map_obs_shape is not None, "map_obs_shape required for shared_backbone=True"
        ac: nnx.Module = SharedDiscreteActorCritic(
            in_features=in_features,
            action_dim=num_actions,
            map_obs_shape=config.map_obs_shape,
            layer_width=config.layer_width,
            rngs=rngs,
        )
        int_critic_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    else:
        actor_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
        critic_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
        int_critic_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
        agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
        actor = DiscretePolicy(in_features=in_features, action_dim=num_actions, **agent_kwargs, rngs=rngs, encoder=actor_enc)
        critic = VNetwork(in_features=in_features, **agent_kwargs, rngs=rngs, encoder=critic_enc)
        ac = ActorCritic(actor, critic)

    # Intrinsic value critic
    intrinsic_critic = VNetwork(
        in_features=in_features,
        **resolve_network_kwargs(getattr(config, "agent_kwargs", None)),
        rngs=rngs,
        encoder=int_critic_enc,
    )

    # ICM module
    icm = ICMModule(
        in_features=in_features,
        num_actions=num_actions,
        latent_size=config.icm_latent_size,
        hidden_layer_sizes=config.icm_hidden_layer_sizes,
        rngs=rngs,
    )

    system = PPOICMSystem(ac, intrinsic_critic, icm)

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
    opt = nnx.Optimizer(system, tx, wrt=nnx.Param)
    return system, opt


_PPO_ICM_METRIC_NAMES = (
    "actor_loss",
    "ext_critic_loss",
    "int_critic_loss",
    "icm_forward_loss",
    "icm_inverse_loss",
    "entropy",
    "clip_fraction",
    "approx_kl",
    "explained_variance",
    "mean_intrinsic_reward",
)


def train_ppo_icm(
    config: PPOICMConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a PPO-ICM agent.

    PPO with Intrinsic Curiosity Module for exploration via prediction error.

    Args:
        config: PPOICMConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, PPOICMConfig

        config = PPOICMConfig.from_yaml("configs/ppo_icm/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("ppo_icm"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    num_steps = config.num_steps
    max_steps = env_params.max_steps_in_episode

    iteration_size = num_envs * num_steps
    assert iteration_size % config.num_minibatches == 0

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    system, opt = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(env.observation_space(env_params).shape)
    rew_rms = RewardRMSState.create(num_envs)
    intrinsic_rew_rms = RMSState.create(())

    zero_train_metrics = {k: jnp.float32(0.0) for k in _PPO_ICM_METRIC_NAMES}

    carry = PPOICMCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        intrinsic_rew_rms=intrinsic_rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, system, rngs):
        """Roll out policy for num_steps. System is read-only."""
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_steps, rng_action = jax.random.split(step_key, 2)
            rng_steps = jax.random.split(rng_steps, num_envs)

            # Extrinsic action + value
            action_B, log_prob_B, value_B = system.ac.get_action_and_value(carry.last_obs, rng_action)

            # Intrinsic value
            intrinsic_value_B = system.intrinsic_critic(carry.last_obs)

            next_obs_BD, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_B, env_params)

            # Intrinsic reward from ICM (forward prediction error, masked at terminal)
            raw_intrinsic_B = system.icm.intrinsic_reward(carry.last_obs, next_obs_BD, action_B)
            intrinsic_reward_B = raw_intrinsic_B * (1 - done_B)

            # Normalize intrinsic rewards
            intrinsic_rew_rms = carry.intrinsic_rew_rms
            if config.intrinsic_reward_norm:
                intrinsic_rew_rms = update_rms(intrinsic_rew_rms, intrinsic_reward_B)
                intrinsic_reward_B = intrinsic_reward_B / jnp.sqrt(intrinsic_rew_rms.var + 1e-8)

            # Track episode returns using raw extrinsic rewards
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

            transition = ICMTrajectory(
                obs=carry.last_obs,
                next_obs=next_obs_BD,
                action=action_B,
                log_prob=log_prob_B,
                reward=reward_B,
                value=value_B,
                intrinsic_reward=intrinsic_reward_B,
                intrinsic_value=intrinsic_value_B,
                done=done_B,
            )
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs_BD,
                last_done=done_B,
                global_step=carry.global_step + num_envs,
                obs_rms=obs_rms,
                rew_rms=rew_rms,
                intrinsic_rew_rms=intrinsic_rew_rms,
                running_return=reset_running_B,
                completed_returns_sum=completed_sum,
                completed_returns_count=completed_count,
            )
            return carry, transition

        carry, trajectories = jax.lax.scan(env_step, carry, step_keys)
        return carry, trajectories

    # --- Combined update ---
    def update_system(batch, system, opt):
        def loss_fn(system):
            obs_B = batch.trajectories.obs
            action_B = batch.trajectories.action
            next_obs_B = batch.trajectories.next_obs
            done_B = batch.trajectories.done
            mask_B = 1 - done_B

            # Actor + extrinsic critic
            log_prob_B, entropy_B, ext_value_B = system.ac.get_pi_and_value(obs_B, action_B)
            entropy = entropy_B.mean()

            # Intrinsic critic
            int_value_B = system.intrinsic_critic(obs_B)

            # --- ICM losses ---
            latent_BL = system.icm.encode(obs_B)
            next_latent_BL = system.icm.encode(next_obs_B)

            # Inverse loss: cross-entropy on action prediction (trains encoder + inverse model)
            action_logits_BA = system.icm.inverse_prediction(latent_BL, next_latent_BL)
            true_action_oh = jax.nn.one_hot(action_B, system.icm.num_actions)
            inverse_loss = -(action_logits_BA * true_action_oh * mask_B[:, None]).sum(axis=-1).mean()

            # Forward loss: MSE (stop-gradient on encoder so forward model doesn't affect encoder)
            pred_next_BL = system.icm.forward_prediction(jax.lax.stop_gradient(latent_BL), action_B)
            forward_error_BL = (pred_next_BL - jax.lax.stop_gradient(next_latent_BL)) * mask_B[:, None]
            forward_loss = jnp.square(forward_error_BL).mean()

            icm_loss = config.icm_inverse_loss_coef * inverse_loss + config.icm_forward_loss_coef * forward_loss

            # --- Actor loss (PPO clipped, combined advantages) ---
            combined_adv_B = batch.advantages_ext + config.intrinsic_reward_coef * batch.advantages_int
            ratio_B = jnp.exp(log_prob_B - batch.trajectories.log_prob)
            adv_B = (combined_adv_B - combined_adv_B.mean()) / (combined_adv_B.std() + 1e-8)
            clipped_ratio_B = jnp.clip(ratio_B, 1 - config.clip_eps, 1 + config.clip_eps)
            pi_loss1_B = ratio_B * adv_B
            pi_loss2_B = clipped_ratio_B * adv_B
            actor_loss = -jnp.minimum(pi_loss1_B, pi_loss2_B).mean()

            # --- Extrinsic critic loss (clipped) ---
            value_clipped_B = batch.trajectories.value + (ext_value_B - batch.trajectories.value).clip(
                -config.clip_eps, config.clip_eps
            )
            loss_B = jnp.square(ext_value_B - batch.targets_ext)
            loss_clipped_B = jnp.square(value_clipped_B - batch.targets_ext)
            ext_critic_loss = 0.5 * jnp.maximum(loss_B, loss_clipped_B).mean()

            # --- Intrinsic critic loss (clipped) ---
            int_value_clipped_B = batch.trajectories.intrinsic_value + (int_value_B - batch.trajectories.intrinsic_value).clip(
                -config.clip_eps, config.clip_eps
            )
            int_loss_B = jnp.square(int_value_B - batch.targets_int)
            int_loss_clipped_B = jnp.square(int_value_clipped_B - batch.targets_int)
            int_critic_loss = 0.5 * jnp.maximum(int_loss_B, int_loss_clipped_B).mean()

            clip_fraction = jnp.mean(jnp.abs(ratio_B - 1) > config.clip_eps)
            approx_kl = jnp.mean((ratio_B - 1) - jnp.log(ratio_B))

            total_loss = (
                actor_loss
                + config.vf_coef * (ext_critic_loss + int_critic_loss)
                + config.icm_loss_coef * icm_loss
                - config.ent_coef * entropy
            )
            return total_loss, (
                actor_loss,
                ext_critic_loss,
                int_critic_loss,
                forward_loss,
                inverse_loss,
                entropy,
                clip_fraction,
                approx_kl,
            )

        (
            (
                _total,
                (actor_loss, ext_critic_loss, int_critic_loss, forward_loss, inverse_loss, entropy, clip_fraction, approx_kl),
            ),
            grads,
        ) = nnx.value_and_grad(loss_fn, has_aux=True)(system)
        opt.update(system, grads)
        return actor_loss, ext_critic_loss, int_critic_loss, forward_loss, inverse_loss, entropy, clip_fraction, approx_kl

    # --- Train iteration ---
    def train_iteration(carry, system, opt, rngs):
        carry, trajectories = collect_trajectories(carry, system, rngs)

        # Extrinsic GAE (episodic)
        last_ext_val_B = system.ac.value(carry.last_obs)
        last_ext_val_B = jnp.where(carry.last_done, 0, last_ext_val_B)
        advantages_ext, targets_ext = calculate_gae(
            trajectories.reward,
            trajectories.value,
            trajectories.done,
            last_ext_val_B,
            config.gamma,
            config.gae_lambda,
        )

        # Intrinsic GAE (non-episodic: done is ignored)
        last_int_val_B = system.intrinsic_critic(carry.last_obs)
        advantages_int, targets_int = calculate_gae(
            trajectories.intrinsic_reward,
            trajectories.intrinsic_value,
            trajectories.done,
            last_int_val_B,
            config.intrinsic_gamma,
            config.gae_lambda,
            episodic=False,
        )

        # Explained variance (extrinsic)
        explained_var = 1 - jnp.var(targets_ext - trajectories.value) / (jnp.var(targets_ext) + 1e-8)

        # Mean intrinsic reward (for logging)
        mean_intrinsic_reward = trajectories.intrinsic_reward.mean()

        update_metric_names = (
            "actor_loss",
            "ext_critic_loss",
            "int_critic_loss",
            "icm_forward_loss",
            "icm_inverse_loss",
            "entropy",
            "clip_fraction",
            "approx_kl",
        )
        zero_update_metrics = {k: jnp.float32(0.0) for k in update_metric_names}

        def update_epoch(_, state):
            system, opt, rngs, metrics_sum = state
            batch = ICMAdvantageMinibatch(trajectories, advantages_ext, targets_ext, advantages_int, targets_int)
            minibatches = shuffle_and_split(batch, rngs(), iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def update_step(state, mb):
                system, opt = state
                actor_loss, ext_critic_loss, int_critic_loss, forward_loss, inverse_loss, entropy, clip_fraction, approx_kl = (
                    update_system(mb, system, opt)
                )
                return (system, opt), {
                    "actor_loss": actor_loss,
                    "ext_critic_loss": ext_critic_loss,
                    "int_critic_loss": int_critic_loss,
                    "icm_forward_loss": forward_loss,
                    "icm_inverse_loss": inverse_loss,
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

        # Average over epochs, add non-update metrics
        iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, epoch_metrics_sum)
        iter_metrics["explained_variance"] = explained_var
        iter_metrics["mean_intrinsic_reward"] = mean_intrinsic_reward

        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, iter_metrics),
            metrics_count=carry.metrics_count + 1,
        )

        return carry, system, opt, rngs

    # --- Eval callback ---
    def eval_callback(carry, system, rngs):
        act_fn = make_eval_act(lambda obs, rng: system.ac.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, system, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _PPO_ICM_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, system, opt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, system, opt, rngs = nnx.fori_loop(0, num_iters_per_eval, train_body, (carry, system, opt, rngs))

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

        # Reset accumulator (traced ops to preserve nnx.scan carry references)
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
        "intrinsic_rew_rms_state": carry.intrinsic_rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
