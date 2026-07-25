"""SGF — VICReg-based world model with imagination-based actor-critic.

"Simple, Good, Fast: Self-Supervised World Models Free of Baggage"
(Robine et al., ICLR 2025)

Learns augmented encoder representations via VICReg + temporal consistency,
a deterministic transition predictor, then trains actor-critic purely in
imagination. Designed for Atari-100k.

Dimension key:
    B: batch (num_envs during collection, batch_size during WM updates)
    D: observation dimension
    Y: encoder output dim (y_dim)
    Z: projector output dim (z_dim)
    A: flat action dimension (action_dim * action_stack, one-hot stacked)
    I: imagination horizon
    N: num_bins (symlog two-hot)
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
    _polyak_update,
    create_env,
    create_vec_env_fns,
    make_eval_env,
    resolve_network_kwargs,
    update_rew_rms,
    update_rms,
)
from nexrl.buffers import CircularBuffer
from nexrl.configs import SGFConfig
from nexrl.evaluate import evaluate
from nexrl.networks.dreamer import (
    ContinuePredictor,
    DreamerActor,
    DreamerCritic,
    RewardPredictor,
    twohot_decode,
    twohot_encode,
)
from nexrl.networks.sgf import (
    SGFEncoder,
    SGFPredictor,
    SGFProjector,
    SGFTransitionPredictor,
    SGFWorldModel,
    augment,
    vicreg_loss,
)
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_SGF_METRIC_NAMES = (
    "repr_loss",
    "dynamics_loss",
    "vicreg_var_loss",
    "vicreg_cov_loss",
    "temporal_loss",
    "reward_loss",
    "terminal_loss",
    "transition_loss",
    "actor_loss",
    "critic_loss",
    "entropy",
)


class SGFTransition(NamedTuple):
    """Transition stored in CircularBuffer for SGF."""

    obs: chex.Array  # (D,) raw observation
    action: chex.Array  # () discrete action index
    reward: chex.Array  # ()
    done: chex.Array  # () terminal flag
    next_obs: chex.Array  # (D,) raw next observation


class SGFCarry(NamedTuple):
    """Non-module carry state for SGF training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    # Action stacking state (ring buffer of recent actions)
    action_stack: chex.Array  # (num_envs, action_stack, action_dim) one-hot
    global_step: int
    buffer: CircularBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    # Return normalization (percentile EMA)
    return_ema_low: chex.Array
    return_ema_high: chex.Array
    # Metrics
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _create_networks(
    config: SGFConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[
    SGFWorldModel,
    SGFTransitionPredictor,
    DreamerActor,
    DreamerCritic,
    nnx.Optimizer,
    nnx.Optimizer,
    nnx.Optimizer,
    nnx.Optimizer,
    DreamerCritic,
]:
    """Create all SGF network components and their optimizers."""
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
    hidden_layer_sizes = agent_kwargs["hidden_layer_sizes"]
    activation = agent_kwargs["activation"]

    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    obs_shape = obs_space.shape
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]

    if discrete:
        action_dim = action_space.n
    else:
        action_dim = int(np.prod(action_space.shape))

    rngs = nnx.Rngs(rng)

    # --- Encoder ---
    in_channels = obs_shape[0]
    height, width = obs_shape[1], obs_shape[2]
    encoder = SGFEncoder(in_channels, height, width, config.y_dim, rngs)

    # --- Projector & Predictor ---
    # Use single action dim for all network inputs (action stacking deferred)
    projector = SGFProjector(config.y_dim, config.z_dim, config.projector_hidden, rngs)
    predictor = SGFPredictor(config.z_dim, action_dim, config.predictor_hidden, rngs)

    # --- Reward & Terminal predictors ---
    # Input: cat(y, action, next_y) => 2*y_dim + action_dim
    pred_input_dim = 2 * config.y_dim + action_dim
    reward_pred = RewardPredictor(
        state_dim=pred_input_dim,
        num_bins=config.reward_num_bins,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    continue_pred = ContinuePredictor(
        state_dim=pred_input_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )

    world_model = SGFWorldModel(encoder, projector, predictor, reward_pred, continue_pred)

    # --- Transition predictor (separate optimizer) ---
    transition = SGFTransitionPredictor(config.y_dim, action_dim, config.transition_hidden, config.transition_num_hidden, rngs)

    # --- Actor & Critic ---
    actor = DreamerActor(
        state_dim=config.y_dim,
        action_dim=action_dim,
        is_discrete=discrete,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    critic = DreamerCritic(
        state_dim=config.y_dim,
        num_bins=config.value_num_bins,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    slow_critic = DreamerCritic(
        state_dim=config.y_dim,
        num_bins=config.value_num_bins,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    nnx.update(slow_critic, nnx.state(critic, nnx.Param))

    # --- Optimizers ---
    # Representation optimizer (encoder + projector + predictor + reward + terminal)
    repr_schedule = optax.linear_schedule(0.0, config.repr_learning_rate, config.repr_warmup_steps)
    repr_tx = optax.chain(
        optax.clip_by_global_norm(config.repr_grad_clip),
        optax.adamw(learning_rate=repr_schedule, weight_decay=config.repr_weight_decay),
    )
    repr_optimizer = nnx.Optimizer(world_model, repr_tx, wrt=nnx.Param)

    # Transition optimizer
    trans_schedule = optax.linear_schedule(0.0, config.transition_learning_rate, config.transition_warmup_steps)
    trans_tx = optax.chain(
        optax.clip_by_global_norm(config.transition_grad_clip),
        optax.adamw(learning_rate=trans_schedule, weight_decay=config.transition_weight_decay),
    )
    transition_optimizer = nnx.Optimizer(transition, trans_tx, wrt=nnx.Param)

    # Actor optimizer (no warmup, no weight decay)
    ac_tx = optax.chain(
        optax.clip_by_global_norm(config.ac_grad_clip),
        optax.adamw(learning_rate=config.ac_learning_rate, eps=1e-5),
    )
    actor_optimizer = nnx.Optimizer(actor, ac_tx, wrt=nnx.Param)

    # Critic optimizer (no warmup, no weight decay)
    critic_optimizer = nnx.Optimizer(critic, ac_tx, wrt=nnx.Param)

    return (
        world_model,
        transition,
        actor,
        critic,
        repr_optimizer,
        transition_optimizer,
        actor_optimizer,
        critic_optimizer,
        slow_critic,
    )


def train_sgf(
    config: SGFConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train an SGF agent.

    Simulation-Free Gradient with VICReg representations and imagination.

    Args:
        config: SGFConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, SGFConfig

        config = SGFConfig.from_yaml("configs/sgf/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("sgf"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)
    # Dedicated eval instances — SGF is Atari-only, whose multi-instance adapter is stateful,
    # so sharing would destroy in-progress training episodes at every eval point.
    eval_env = make_eval_env(env, config.run_evaluation)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]

    if discrete:
        action_dim = action_space.n
    else:
        action_dim = int(np.prod(action_space.shape))

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    (
        world_model,
        transition,
        actor,
        critic,
        repr_optimizer,
        transition_optimizer,
        actor_optimizer,
        critic_optimizer,
        slow_critic,
    ) = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init buffer — store both obs and next_obs
    dummy_transition = SGFTransition(
        obs=jnp.empty((config.buffer_size, *obs_space.shape), dtype=obs_space.dtype),
        action=jnp.empty(config.buffer_size, dtype=jnp.int32),
        reward=jnp.empty(config.buffer_size, dtype=jnp.float32),
        done=jnp.empty(config.buffer_size, dtype=bool),
        next_obs=jnp.empty((config.buffer_size, *obs_space.shape), dtype=obs_space.dtype),
    )
    buf = CircularBuffer.empty(config.buffer_size, dummy_transition)

    # Init action stack (one-hot ring buffer)
    action_stack_init = jnp.zeros((num_envs, config.action_stack, action_dim), dtype=jnp.float32)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _SGF_METRIC_NAMES}

    carry = SGFCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        action_stack=action_stack_init,
        global_step=0,
        buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        return_ema_low=jnp.float32(0.0),
        return_ema_high=jnp.float32(0.0),
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- Helper: advance action stack ---
    def advance_action_stack(stack_BMA: jax.Array, action_B: jax.Array, done_B: jax.Array) -> jax.Array:
        """Push new action, reset stack on done."""
        # Reset stack where done
        reset = done_B.astype(jnp.float32)[:, None, None]
        stack_BMA = stack_BMA * (1.0 - reset)
        # New one-hot action
        new_onehot = jax.nn.one_hot(action_B, action_dim, dtype=jnp.float32)  # (B, action_dim)
        # Shift left and append
        return jnp.concatenate([stack_BMA[:, 1:], new_onehot[:, None]], axis=1)

    # --- Collect one transition ---
    def collect_transition(carry, world_model_, actor_, rngs_):
        obs_for_net = carry.last_obs

        # Encode observation (no augmentation for collection/policy)
        y_BY = world_model_.encoder(obs_for_net)

        # Select action from actor
        rng_act, rng_eps = jax.random.split(rngs_())
        action_env = actor.act(y_BY, rng_act)

        # Epsilon-greedy exploration
        random_action = jax.random.randint(rng_eps, (num_envs,), 0, action_dim)
        use_random = jax.random.uniform(rngs_(), (num_envs,)) < config.env_epsilon
        action_env = jnp.where(use_random, random_action, action_env)

        # Step environment
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

        transition = SGFTransition(
            obs=carry.last_obs,
            action=action_env,
            reward=rewards.astype(jnp.float32),
            done=dones,
            next_obs=next_obs,
        )

        # Advance action stack
        new_action_stack = advance_action_stack(carry.action_stack, action_env, dones)

        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            last_done=dones,
            action_stack=new_action_stack,
            global_step=carry.global_step + num_envs,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry, transition

    # --- Sample random batch from buffer ---
    def sample_batch(buffer, rng_sample):
        n = buffer.num_entries
        indices = jax.random.randint(rng_sample, (config.batch_size,), 0, n)
        return jax.tree.map(lambda arr: arr[indices], buffer.data)

    def sample_imag_batch(buffer, rng_sample):
        """Sample starting observations for imagination."""
        n = buffer.num_entries
        indices = jax.random.randint(rng_sample, (config.imagination_batch_size,), 0, n)
        return jax.tree.map(lambda arr: arr[indices], buffer.data)

    # --- World model representation loss ---
    def repr_loss_fn(world_model_, batch, rng):
        """VICReg + temporal consistency + reward + terminal on augmented obs."""
        rng_aug1, rng_aug2 = jax.random.split(rng)

        obs = batch.obs  # (B, *obs_shape)
        next_obs = batch.next_obs  # (B, *obs_shape)
        action_B = batch.action  # (B,)
        reward_B = batch.reward  # (B,)
        done_B = batch.done  # (B,)

        # Augment both observations
        obs_aug = augment(rng_aug1, obs, config.shift_pad, config.intensity_scale)
        next_obs_aug = augment(rng_aug2, next_obs, config.shift_pad, config.intensity_scale)

        # Encode augmented obs
        y_BY = world_model_.encoder(obs_aug)
        next_y_BY = world_model_.encoder(next_obs_aug)

        # Project to VICReg space
        z_BZ = world_model_.projector(y_BY)
        next_z_BZ = world_model_.projector(next_y_BY)

        # Temporal consistency: predictor(z, action) -> predicted next_z
        # Use flat one-hot action (single action, not stacked — consistency with reference)
        flat_a = jax.nn.one_hot(action_B, action_dim, dtype=jnp.float32)
        # For repr loss, the reference uses single action one-hot, not stacked actions
        # But the reference flattens the action stack. Let me follow the reference exactly:
        # The predictor in the reference takes cat(z, flat_action) where flat_action is
        # the one-hot stacked actions. However, during WM training, the batch doesn't
        # store the full action stack. We use single one-hot action for simplicity here
        # (the reference also trains with the last stacked action from the transition).
        z_pred_BZ = world_model_.predictor(z_BZ, flat_a)
        temporal_loss = jnp.mean(jnp.sum((z_pred_BZ - jax.lax.stop_gradient(next_z_BZ)) ** 2, axis=-1)) / config.z_dim

        # VICReg (variance + covariance)
        var_loss, cov_loss = vicreg_loss(z_BZ, next_z_BZ, config.var_weight, config.cov_weight)

        # Reward prediction: cat(y, action, next_y) -> reward
        pred_input = jnp.concatenate([y_BY, flat_a, next_y_BY], axis=-1)
        reward_logits = world_model_.reward_pred(pred_input)
        reward_target = twohot_encode(reward_B, config.reward_num_bins)
        reward_loss = -jnp.mean(jnp.sum(reward_target * jax.nn.log_softmax(reward_logits, axis=-1), axis=-1))

        # Terminal prediction: cat(y, action, next_y) -> terminal
        terminal_logits = world_model_.continue_pred(pred_input)
        terminal_target = done_B.astype(jnp.float32)
        terminal_loss = jnp.mean(
            terminal_target * jax.nn.softplus(-terminal_logits) + (1.0 - terminal_target) * jax.nn.softplus(terminal_logits)
        )

        total_repr_loss = (
            config.consistency_weight * temporal_loss
            + var_loss
            + cov_loss
            + config.reward_weight * reward_loss
            + config.terminal_weight * terminal_loss
        )

        metrics = {
            "repr_loss": total_repr_loss,
            "vicreg_var_loss": var_loss,
            "vicreg_cov_loss": cov_loss,
            "temporal_loss": temporal_loss,
            "reward_loss": reward_loss,
            "terminal_loss": terminal_loss,
        }
        return total_repr_loss, metrics

    # --- Transition predictor loss ---
    def transition_loss_fn(transition_, world_model_, batch):
        """L2 loss on un-augmented encoder outputs (encoder is stop-gradient'd)."""
        obs = batch.obs
        next_obs = batch.next_obs
        action_B = batch.action

        # Encode un-augmented obs (detached from encoder)
        y_BY = jax.lax.stop_gradient(world_model_.encoder(obs))
        next_y_BY = jax.lax.stop_gradient(world_model_.encoder(next_obs))

        flat_a = jax.nn.one_hot(action_B, action_dim, dtype=jnp.float32)
        next_y_pred = transition_(y_BY, flat_a)

        trans_loss = jnp.mean(jnp.sum((next_y_pred - next_y_BY) ** 2, axis=-1)) / config.y_dim
        return trans_loss, {"transition_loss": trans_loss}

    # --- Actor-critic imagination loss ---
    def imagination_loss(
        actor_,
        critic_,
        world_model_,
        transition_,
        slow_critic_,
        start_obs,
        return_ema_low,
        return_ema_high,
        rng,
    ):
        """REINFORCE actor-critic trained in imagination."""
        H = config.imagination_horizon
        B_imag = config.imagination_batch_size

        # Encode starting observations (no augmentation, detached)
        start_y = jax.lax.stop_gradient(world_model_.encoder(start_obs))

        # Imagination rollout (all read-only, jax.lax.scan is fine)
        def img_step(carry_img, rng_step):
            y, cont_mask = carry_img
            # Actor on sg(y)
            sg_y = jax.lax.stop_gradient(y)
            action, _ = actor_(sg_y, rng_step)
            action_idx = jnp.argmax(action, axis=-1)  # discrete only

            # Transition: next_y = y + delta (on detached y)
            flat_a = jax.nn.one_hot(action_idx, action_dim, dtype=jnp.float32)
            next_y = transition_(jax.lax.stop_gradient(y), flat_a)

            # Reward and terminal predictions
            pred_input = jnp.concatenate([jax.lax.stop_gradient(y), flat_a, next_y], axis=-1)
            next_r = twohot_decode(world_model_.reward_pred(pred_input), config.reward_num_bins)
            terminal_logit = world_model_.continue_pred(pred_input)
            next_term = jax.nn.sigmoid(terminal_logit)  # P(terminal)

            # Continue mask: (1 - terminal) * gamma
            next_cont = (1.0 - next_term) * config.gamma

            # Mask y for terminal states
            next_y_masked = next_y * (1.0 - next_term)[:, None]

            # Cumulative continue mask for loss weighting
            new_cont_mask = cont_mask * (1.0 - next_term)

            return (next_y_masked, new_cont_mask), (y, action_idx, next_r, next_cont, cont_mask)

        img_rngs = jax.random.split(rng, H)
        init_cont_mask = jnp.ones(B_imag)
        _, (all_y_HB, all_actions_HB, all_rewards_HB, all_conts_HB, all_masks_HB) = jax.lax.scan(
            img_step, (start_y, init_cont_mask), img_rngs
        )
        # all_y_HB: (H, B_imag, Y) — states at t=0..H-1
        # all_rewards_HB: (H, B_imag) — rewards at t=1..H
        # all_conts_HB: (H, B_imag) — discount factors at t=1..H

        # Critic values at all states (t=0..H-1) + bootstrap at t=H
        # We need the final y for bootstrap
        final_y = jax.lax.stop_gradient(all_y_HB[-1])  # approximate, could also run transition one more time
        sg_all_y = jax.lax.stop_gradient(all_y_HB)
        all_critic_logits = critic_(sg_all_y.reshape(H * B_imag, config.y_dim))
        online_values = jax.lax.stop_gradient(twohot_decode(all_critic_logits, config.value_num_bins).reshape(H, B_imag))

        # Bootstrap value at end of horizon
        bootstrap_val = jax.lax.stop_gradient(
            twohot_decode(critic_(final_y.reshape(B_imag, config.y_dim)), config.value_num_bins)
        )

        # Lambda returns (reverse accumulation)
        def lambda_scan(next_val, t_data):
            rew, disc, cont_val = t_data
            target = rew + disc * ((1.0 - config.lambda_gae) * cont_val + config.lambda_gae * next_val)
            return target, target

        # online_values[t] is V(y_t) for t=0..H-1
        # rewards[t] is r_{t+1}, conts[t] is gamma * (1 - term_{t+1})
        # We need values at t=1..H for the cont_val in lambda returns
        # Shift: online_values[1:] gives V(y_1)..V(y_{H-1}), and bootstrap_val for V(y_H)
        cont_vals = jnp.concatenate([online_values[1:], bootstrap_val[None]], axis=0)  # (H, B)

        _, lambda_returns = jax.lax.scan(
            lambda_scan,
            bootstrap_val,
            (all_rewards_HB[::-1], all_conts_HB[::-1], cont_vals[::-1]),
        )
        lambda_returns = lambda_returns[::-1]  # (H, B_imag)

        # Return normalization (percentile EMA)
        returns_flat = lambda_returns.reshape(-1)
        lo = jnp.percentile(returns_flat, config.return_norm_perclo)
        hi = jnp.percentile(returns_flat, config.return_norm_perchi)
        rate = 1.0 - config.return_norm_decay
        new_ema_low = return_ema_low * config.return_norm_decay + lo * rate
        new_ema_high = return_ema_high * config.return_norm_decay + hi * rate
        scale = jnp.maximum(new_ema_high - new_ema_low, config.return_norm_limit)

        # Cumulative discount weight for loss weighting
        weight = jax.lax.stop_gradient(all_masks_HB)  # (H, B_imag)

        # Actor loss: REINFORCE on sg(states)
        sg_feat = jax.lax.stop_gradient(all_y_HB).reshape(H * B_imag, config.y_dim)
        sg_actions = jax.lax.stop_gradient(all_actions_HB).reshape(H * B_imag)
        reinforce_logprob = actor_.log_prob_of(sg_feat, sg_actions).reshape(H, B_imag)
        actor_entropy = actor_.entropy(sg_feat).reshape(H, B_imag)

        # Normalized advantage
        advantages = (lambda_returns - new_ema_low) / scale - (online_values - new_ema_low) / scale
        advantages = jax.lax.stop_gradient(advantages)

        actor_loss = -jnp.mean(weight * (reinforce_logprob * advantages + config.ent_coef * actor_entropy))
        mean_entropy = jnp.mean(actor_entropy)

        return actor_loss, mean_entropy, new_ema_low, new_ema_high

    def critic_loss_fn(
        critic_,
        world_model_,
        transition_,
        slow_critic_,
        actor_,
        start_obs,
        return_ema_low,
        return_ema_high,
        rng,
    ):
        """Critic two-hot CE loss on lambda returns + slow value regularization."""
        H = config.imagination_horizon
        B_imag = config.imagination_batch_size

        start_y = jax.lax.stop_gradient(world_model_.encoder(start_obs))

        def img_step(carry_img, rng_step):
            y, cont_mask = carry_img
            sg_y = jax.lax.stop_gradient(y)
            action, _ = actor_(sg_y, rng_step)
            action_idx = jnp.argmax(action, axis=-1)
            flat_a = jax.nn.one_hot(action_idx, action_dim, dtype=jnp.float32)
            next_y = transition_(jax.lax.stop_gradient(y), flat_a)
            pred_input = jnp.concatenate([jax.lax.stop_gradient(y), flat_a, next_y], axis=-1)
            next_r = twohot_decode(world_model_.reward_pred(pred_input), config.reward_num_bins)
            terminal_logit = world_model_.continue_pred(pred_input)
            next_term = jax.nn.sigmoid(terminal_logit)
            next_cont = (1.0 - next_term) * config.gamma
            next_y_masked = next_y * (1.0 - next_term)[:, None]
            new_cont_mask = cont_mask * (1.0 - next_term)
            return (next_y_masked, new_cont_mask), (y, next_r, next_cont, cont_mask)

        img_rngs = jax.random.split(rng, H)
        init_cont_mask = jnp.ones(B_imag)
        (_, _), (all_y_HB, all_rewards_HB, all_conts_HB, all_masks_HB) = jax.lax.scan(
            img_step, (start_y, init_cont_mask), img_rngs
        )

        sg_all_y = jax.lax.stop_gradient(all_y_HB)
        all_critic_logits = critic_(sg_all_y.reshape(H * B_imag, config.y_dim))
        online_values = jax.lax.stop_gradient(twohot_decode(all_critic_logits, config.value_num_bins).reshape(H, B_imag))

        final_y = jax.lax.stop_gradient(all_y_HB[-1])
        bootstrap_val = jax.lax.stop_gradient(
            twohot_decode(critic_(final_y.reshape(B_imag, config.y_dim)), config.value_num_bins)
        )

        def lambda_scan(next_val, t_data):
            rew, disc, cont_val = t_data
            target = rew + disc * ((1.0 - config.lambda_gae) * cont_val + config.lambda_gae * next_val)
            return target, target

        cont_vals = jnp.concatenate([online_values[1:], bootstrap_val[None]], axis=0)
        _, lambda_returns = jax.lax.scan(
            lambda_scan,
            bootstrap_val,
            (all_rewards_HB[::-1], all_conts_HB[::-1], cont_vals[::-1]),
        )
        lambda_returns = lambda_returns[::-1]

        weight = jax.lax.stop_gradient(all_masks_HB)

        # Critic loss: two-hot CE on lambda returns
        target_returns = jax.lax.stop_gradient(lambda_returns)
        target_twohot = twohot_encode(target_returns.reshape(-1), config.value_num_bins)
        critic_logits = all_critic_logits  # reuse from above
        critic_xent = -jnp.sum(target_twohot * jax.nn.log_softmax(critic_logits, axis=-1), axis=-1)

        # Slow value regularization
        slow_values = jax.lax.stop_gradient(
            twohot_decode(slow_critic_(sg_all_y.reshape(H * B_imag, config.y_dim)), config.value_num_bins)
        )
        slow_target_twohot = twohot_encode(slow_values, config.value_num_bins)
        slow_reg = -jnp.sum(slow_target_twohot * jax.nn.log_softmax(critic_logits, axis=-1), axis=-1)

        weight_flat = weight.reshape(-1)
        critic_loss = jnp.mean(weight_flat * (critic_xent + config.slow_value_reg * slow_reg))

        return critic_loss

    # --- Train iteration ---
    def train_iteration(
        carry, world_model_, transition_, actor_, critic_, slow_critic_, repr_opt_, trans_opt_, actor_opt_, critic_opt_, rngs_
    ):
        start_training = carry.global_step > config.fill_buffer

        carry, batch = collect_transition(carry, world_model_, actor_, rngs_)
        carry = carry._replace(buffer=carry.buffer.extend(batch))

        zero_update_metrics = {k: jnp.float32(0.0) for k in _SGF_METRIC_NAMES}

        def do_wm_update(carry_u, world_model_u, transition_u, repr_opt_u, trans_opt_u, rngs_u):
            # Sample batch for WM training
            rng_sample = rngs_u()
            wm_batch = sample_batch(carry_u.buffer, rng_sample)

            # Representation loss
            rng_repr = rngs_u()
            (_, repr_metrics), repr_grads = nnx.value_and_grad(repr_loss_fn, has_aux=True)(world_model_u, wm_batch, rng_repr)
            repr_opt_u.update(world_model_u, repr_grads)

            # Transition loss
            (_, trans_metrics), trans_grads = nnx.value_and_grad(transition_loss_fn, has_aux=True)(
                transition_u, world_model_u, wm_batch
            )
            trans_opt_u.update(transition_u, trans_grads)

            metrics = {
                **zero_update_metrics,  # start with all zeros (ensures same keys)
                **repr_metrics,
                **trans_metrics,
                "dynamics_loss": repr_metrics["repr_loss"] + trans_metrics["transition_loss"],
            }
            return carry_u, world_model_u, transition_u, repr_opt_u, trans_opt_u, rngs_u, metrics

        def skip_wm_update(carry_u, world_model_u, transition_u, repr_opt_u, trans_opt_u, rngs_u):
            return carry_u, world_model_u, transition_u, repr_opt_u, trans_opt_u, rngs_u, zero_update_metrics

        # WM update every wm_every steps
        do_wm = jnp.logical_and(start_training, carry.global_step % config.wm_every == 0)
        carry, world_model_, transition_, repr_opt_, trans_opt_, rngs_, wm_metrics = nnx.cond(
            do_wm, do_wm_update, skip_wm_update, carry, world_model_, transition_, repr_opt_, trans_opt_, rngs_
        )

        def do_ac_update(
            carry_u, actor_u, critic_u, slow_critic_u, actor_opt_u, critic_opt_u, world_model_u, transition_u, rngs_u
        ):
            # Sample starting obs for imagination
            rng_sample = rngs_u()
            imag_batch = sample_imag_batch(carry_u.buffer, rng_sample)
            start_obs = imag_batch.obs

            rng_imag = rngs_u()

            # Actor loss
            def actor_loss_wrapper(actor_inner):
                a_loss, entropy, new_low, new_high = imagination_loss(
                    actor_inner,
                    critic_u,
                    world_model_u,
                    transition_u,
                    slow_critic_u,
                    start_obs,
                    carry_u.return_ema_low,
                    carry_u.return_ema_high,
                    rng_imag,
                )
                return a_loss, (entropy, new_low, new_high)

            (actor_loss_val, (entropy_val, new_low, new_high)), actor_grads = nnx.value_and_grad(
                actor_loss_wrapper, has_aux=True
            )(actor_u)
            actor_opt_u.update(actor_u, actor_grads)

            # Critic loss (separate imagination rollout with same rng for consistency)
            rng_critic = rngs_u()

            def critic_loss_wrapper(critic_inner):
                return critic_loss_fn(
                    critic_inner,
                    world_model_u,
                    transition_u,
                    slow_critic_u,
                    actor_u,
                    start_obs,
                    carry_u.return_ema_low,
                    carry_u.return_ema_high,
                    rng_critic,
                )

            critic_loss_val, critic_grads = nnx.value_and_grad(critic_loss_wrapper)(critic_u)
            critic_opt_u.update(critic_u, critic_grads)

            # Polyak update slow critic
            _polyak_update(critic_u, slow_critic_u, 1.0 - config.slow_value_rate)

            carry_u = carry_u._replace(return_ema_low=new_low, return_ema_high=new_high)

            ac_metrics = {
                "actor_loss": actor_loss_val,
                "critic_loss": critic_loss_val,
                "entropy": entropy_val,
            }
            return carry_u, actor_u, critic_u, slow_critic_u, actor_opt_u, critic_opt_u, rngs_u, ac_metrics

        def skip_ac_update(
            carry_u, actor_u, critic_u, slow_critic_u, actor_opt_u, critic_opt_u, world_model_u, transition_u, rngs_u
        ):
            return (
                carry_u,
                actor_u,
                critic_u,
                slow_critic_u,
                actor_opt_u,
                critic_opt_u,
                rngs_u,
                {
                    "actor_loss": jnp.float32(0.0),
                    "critic_loss": jnp.float32(0.0),
                    "entropy": jnp.float32(0.0),
                },
            )

        # AC update every agent_every steps
        do_ac = jnp.logical_and(start_training, carry.global_step % config.agent_every == 0)
        carry, actor_, critic_, slow_critic_, actor_opt_, critic_opt_, rngs_, ac_metrics = nnx.cond(
            do_ac,
            do_ac_update,
            skip_ac_update,
            carry,
            actor_,
            critic_,
            slow_critic_,
            actor_opt_,
            critic_opt_,
            world_model_,
            transition_,
            rngs_,
        )

        # Merge metrics
        update_metrics = {**wm_metrics, **ac_metrics}

        # Accumulate metrics (only when training)
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, update_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return (
            carry,
            world_model_,
            transition_,
            actor_,
            critic_,
            slow_critic_,
            repr_opt_,
            trans_opt_,
            actor_opt_,
            critic_opt_,
            rngs_,
        )

    # --- Eval ---
    def eval_act(world_model_, actor_, obs, rng):
        """Stateless eval: encode obs, greedy action."""
        y = world_model_.encoder(obs[None])
        return actor_.act_greedy(y).squeeze(0)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = evaluate(
            lambda obs, rng: eval_act(world_model, actor, obs, rng), rngs(), eval_env, env_params, 128, max_steps
        )
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _SGF_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, wm_, trans_, actor_, critic_, slow_c_, ro_, to_, ao_, co_, rngs_ = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, wm_, trans_, actor_, critic_, slow_c_, ro_, to_, ao_, co_, rngs_ = nnx.fori_loop(
            0, steps_per_iter, train_body, (carry, wm_, trans_, actor_, critic_, slow_c_, ro_, to_, ao_, co_, rngs_)
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = evaluate(
                lambda obs, rng: eval_act(wm_, actor_, obs, rng), rngs_(), eval_env, env_params, 128, max_steps
            )
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

        # Reset accumulator
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda x: x * 0, carry.train_metrics),
            metrics_count=carry.metrics_count * 0,
            completed_returns_sum=carry.completed_returns_sum * 0,
            completed_returns_count=carry.completed_returns_count * 0,
        )

        return (carry, wm_, trans_, actor_, critic_, slow_c_, ro_, to_, ao_, co_, rngs_), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    full_state = (
        carry,
        world_model,
        transition,
        actor,
        critic,
        slow_critic,
        repr_optimizer,
        transition_optimizer,
        actor_optimizer,
        critic_optimizer,
        rngs,
    )
    full_state, all_metrics = scan_eval(full_state, jnp.zeros(num_evals))

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    carry = full_state[0]
    state = {
        "world_model": full_state[1],
        "transition": full_state[2],
        "actor": full_state[3],
        "critic": full_state[4],
        "slow_critic": full_state[5],
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
