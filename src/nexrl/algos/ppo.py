"""Proximal Policy Optimization (PPO) — standalone training function.

"Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

Unified implementation supporting separate-network and shared-backbone actor-critics
via the ActorCritic protocol (get_action_and_value, get_pi_and_value, value, act).

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    T: trajectory length (num_steps)
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
)
from nexrl.configs import PPOConfig
from nexrl.evaluate import evaluate
from nexrl.networks import (
    ActorCritic,
    DiscretePolicy,
    GaussianPolicy,
    SharedDiscreteActorCritic,
    VNetwork,
)
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


class PPOCarry(NamedTuple):
    """Non-module carry state for PPO training."""

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


def _create_networks(
    config: PPOConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[nnx.Module, nnx.Optimizer]:
    """Create actor-critic and optimizer for all PPO variants."""
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    rngs = nnx.Rngs(rng)

    if config.shared_backbone:
        assert config.map_obs_shape is not None, "map_obs_shape required for shared_backbone=True"
        ac: nnx.Module = SharedDiscreteActorCritic(
            in_features=in_features,
            action_dim=action_space.n,
            map_obs_shape=config.map_obs_shape,
            layer_width=config.layer_width,
            rngs=rngs,
        )
    else:
        agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]
        actor_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
        critic_encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None

        if discrete:
            actor = DiscretePolicy(
                in_features=in_features,
                action_dim=action_space.n,
                **agent_kwargs,
                rngs=rngs,
                encoder=actor_encoder,
            )
            critic = VNetwork(in_features=in_features, **agent_kwargs, rngs=rngs, encoder=critic_encoder)
            ac = ActorCritic(actor, critic)
        else:
            actor = GaussianPolicy(
                in_features=in_features,
                action_dim=int(np.prod(action_space.shape)),
                action_range=(float(action_space.low), float(action_space.high)),
                **agent_kwargs,
                rngs=rngs,
                encoder=actor_encoder,
            )
            critic = VNetwork(in_features=in_features, **agent_kwargs, rngs=rngs, encoder=critic_encoder)
            ac = ActorCritic(actor, critic)

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


_PPO_METRIC_NAMES = ("actor_loss", "critic_loss", "entropy", "clip_fraction", "approx_kl", "explained_variance")


def train_ppo(
    config: PPOConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a PPO agent.

    Proximal Policy Optimization with clipped surrogate objective and Generalized Advantage Estimation.

    Args:
        config: PPOConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, PPOConfig

        config = PPOConfig.from_yaml("configs/ppo/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("ppo"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    num_steps = config.num_steps
    action_space = env.action_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]

    iteration_size = num_envs * num_steps
    assert iteration_size % config.num_minibatches == 0

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Action clipping bounds for continuous envs
    if not discrete:
        action_low = action_space.low
        action_high = action_space.high

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    ac, opt = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(env.observation_space(env_params).shape)
    rew_rms = RewardRMSState.create(num_envs)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _PPO_METRIC_NAMES}

    carry = PPOCarry(
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
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, ac, rngs):
        """Roll out policy for num_steps. AC is read-only."""
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_steps, rng_action = jax.random.split(step_key, 2)
            rng_steps = jax.random.split(rng_steps, num_envs)

            # Single forward pass through actor-critic
            action_B, log_prob_B, value_B = ac.get_action_and_value(carry.last_obs, rng_action)

            # Clip for env step (continuous only)
            env_action = action_B if discrete else jnp.clip(action_B, action_low, action_high)

            next_obs_BD, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, env_action, env_params)

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

            # Store raw (unclipped) action for correct log_prob in update
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

    # --- Combined update ---
    def update_ac(batch, ac, opt):
        def loss_fn(model):
            log_prob_B, entropy_B, value_B = model.get_pi_and_value(batch.trajectories.obs, batch.trajectories.action)
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
        )(ac)
        opt.update(ac, grads)
        return actor_loss, critic_loss, entropy, clip_fraction, approx_kl

    # --- Train iteration ---
    def train_iteration(carry, ac, opt, rngs):
        carry, trajectories = collect_trajectories(carry, ac, rngs)

        last_val_B = ac.value(carry.last_obs)
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
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = shuffle_and_split(batch, rngs(), iteration_size, config.num_minibatches)

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
        act_fn = make_eval_act(lambda obs, rng: ac.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

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
            **{k: jnp.float32(0.0) for k in _PPO_METRIC_NAMES},
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
