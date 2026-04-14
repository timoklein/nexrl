"""Dreamer-v3 — world model learning with imagination-based actor-critic.

"Mastering Diverse Domains through World Models" (Hafner et al., 2024)

Learns an RSSM world model from experience, then trains actor-critic purely
in imagination (latent rollouts). Supports both discrete and continuous actions.

Dimension key:
    B: batch (num_envs during collection, batch_size * sequence_length during updates)
    D: observation dimension (flattened obs_space.shape)
    G: deter_dim (GRU hidden state)
    Z: stoch_dim (C * L flattened)
    A: action dimension
    T: sequence length
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
    normalize_obs,
    resolve_network_kwargs,
    update_rew_rms,
    update_rms,
)
from nexrl.buffers import CircularBuffer
from nexrl.configs import DreamerConfig
from nexrl.networks.dreamer import (
    RSSM,
    ContinuePredictor,
    DreamerActor,
    DreamerCritic,
    DreamerDecoder,
    DreamerEncoder,
    DreamerWorldModel,
    RewardPredictor,
    symlog,
    twohot_decode,
    twohot_encode,
)
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_DREAMER_METRIC_NAMES = (
    "world_model_loss",
    "recon_loss",
    "reward_loss",
    "continue_loss",
    "dyn_kl",
    "rep_kl",
    "actor_loss",
    "critic_loss",
    "entropy",
)


class DreamerTransition(NamedTuple):
    """Transition stored in CircularBuffer for Dreamer."""

    obs: chex.Array  # (D,)
    action: chex.Array  # () discrete or (A,) continuous
    reward: chex.Array  # ()
    done: chex.Array  # () terminal flag
    is_first: chex.Array  # () episode start flag


class EvalCarry(NamedTuple):
    """Carry state for stateful RSSM evaluation."""

    rng: jax.Array
    env_st: Any
    obs: jax.Array
    h: jax.Array
    z: jax.Array
    last_action: jax.Array
    done: jax.Array
    ret: jax.Array
    length: jax.Array


class DreamerCarry(NamedTuple):
    """Non-module carry state for Dreamer training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    last_is_first: chex.Array
    # Persistent RSSM state for collection
    rssm_h: chex.Array
    rssm_z: chex.Array
    last_action: chex.Array
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


def categorical_kl(
    post_logits_BCL: jax.Array,
    prior_logits_BCL: jax.Array,
    unimix: float = 0.01,
) -> jax.Array:
    """KL divergence KL[post || prior] for product of categoricals.

    Applies unimix before computing KL, matching the reference CatDist which
    parameterizes distributions as 99% predicted + 1% uniform.

    Args:
        post_logits_BCL: posterior logits (B, C, L)
        prior_logits_BCL: prior logits (B, C, L)
        unimix: uniform mixing coefficient

    Returns:
        KL divergence per sample (B,), summed over categories
    """
    L = post_logits_BCL.shape[-1]
    uniform = jnp.ones_like(post_logits_BCL) / L
    # Apply unimix to both distributions (matching reference CatDist)
    post_probs = (1.0 - unimix) * jax.nn.softmax(post_logits_BCL, axis=-1) + unimix * uniform
    prior_probs = (1.0 - unimix) * jax.nn.softmax(prior_logits_BCL, axis=-1) + unimix * uniform
    # Recompute log-probs from mixed distributions
    post_logprobs = jnp.log(post_probs + 1e-8)
    prior_logprobs = jnp.log(prior_probs + 1e-8)
    # KL per category, sum over classes then categories
    kl_BCL = post_probs * (post_logprobs - prior_logprobs)
    return kl_BCL.sum(axis=(-2, -1))  # (B,)


def _create_networks(
    config: DreamerConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[DreamerWorldModel, nnx.Optimizer, DreamerCritic]:
    """Create DreamerWorldModel, optimizer, and slow critic."""
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
    hidden_layer_sizes = agent_kwargs["hidden_layer_sizes"]
    activation = agent_kwargs["activation"]

    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]

    if discrete:
        action_dim = action_space.n
        action_range = None
    else:
        action_dim = int(np.prod(action_space.shape))
        action_range = (float(action_space.low), float(action_space.high))

    rssm_cfg = config.rssm_kwargs
    stoch_dim = rssm_cfg.stoch_categories * rssm_cfg.stoch_classes
    state_dim = rssm_cfg.deter_dim + stoch_dim
    embed_dim = hidden_layer_sizes[-1]

    rngs = nnx.Rngs(rng)

    encoder = DreamerEncoder(
        in_features=in_features,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    rssm = RSSM(
        stoch_categories=rssm_cfg.stoch_categories,
        stoch_classes=rssm_cfg.stoch_classes,
        deter_dim=rssm_cfg.deter_dim,
        action_dim=action_dim,
        embed_dim=embed_dim,
        hidden_dim=rssm_cfg.hidden_dim,
        activation=activation,
        unimix=rssm_cfg.unimix,
        rngs=rngs,
    )
    decoder = DreamerDecoder(
        state_dim=state_dim,
        obs_dim=in_features,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    reward_pred = RewardPredictor(
        state_dim=state_dim,
        num_bins=config.value_num_bins,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    continue_pred = ContinuePredictor(
        state_dim=state_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    actor = DreamerActor(
        state_dim=state_dim,
        action_dim=action_dim,
        is_discrete=discrete,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
        action_range=action_range,
    )
    critic = DreamerCritic(
        state_dim=state_dim,
        num_bins=config.value_num_bins,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )

    model = DreamerWorldModel(encoder, rssm, decoder, reward_pred, continue_pred, actor, critic)

    lr_schedule = optax.linear_schedule(init_value=0.0, end_value=config.learning_rate, transition_steps=1000)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=lr_schedule),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Slow critic for value regularization
    slow_critic = DreamerCritic(
        state_dim=state_dim,
        num_bins=config.value_num_bins,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        rngs=rngs,
    )
    nnx.update(slow_critic, nnx.state(critic, nnx.Param))

    return model, optimizer, slow_critic


def train_dreamer(
    config: DreamerConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a Dreamer-v3 agent.

    Dreamer-v3 with RSSM world model and imagination-based actor-critic.

    Args:
        config: DreamerConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, DreamerConfig

        config = DreamerConfig.from_yaml("configs/dreamer/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("dreamer"), in_axes=(None, 0)),
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
    else:
        action_dim = int(np.prod(action_space.shape))

    rssm_cfg = config.rssm_kwargs
    stoch_dim = rssm_cfg.stoch_categories * rssm_cfg.stoch_classes
    state_dim = rssm_cfg.deter_dim + stoch_dim

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    model, optimizer, slow_critic = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init buffer
    if discrete:
        action_shape = ()
        action_dtype = jnp.int32
    else:
        action_shape = (action_dim,)
        action_dtype = jnp.float32

    dummy_transition = DreamerTransition(
        obs=jnp.empty((config.buffer_size, *obs_space.shape), dtype=obs_space.dtype),
        action=jnp.empty((config.buffer_size, *action_shape), dtype=action_dtype),
        reward=jnp.empty(config.buffer_size, dtype=jnp.float32),
        done=jnp.empty(config.buffer_size, dtype=bool),
        is_first=jnp.empty(config.buffer_size, dtype=bool),
    )
    buf = CircularBuffer.empty(config.buffer_size, dummy_transition)

    # Init RSSM state
    h_init, z_init = model.rssm.initial_state(num_envs)
    if discrete:
        last_action = jnp.zeros(num_envs, dtype=jnp.int32)
    else:
        last_action = jnp.zeros((num_envs, action_dim), dtype=jnp.float32)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _DREAMER_METRIC_NAMES}

    carry = DreamerCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        last_is_first=jnp.ones(num_envs, dtype=bool),
        rssm_h=h_init,
        rssm_z=z_init,
        last_action=last_action,
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

    # --- Collect transitions ---
    def collect_transition(carry, model_, rngs_):
        """Collect one transition from all envs, update RSSM state."""
        obs_for_net = carry.last_obs
        if config.normalize_observations:
            obs_for_net = normalize_obs(carry.obs_rms, obs_for_net)

        # Encode observation
        embed_BF = model_.encoder(obs_for_net)

        # Prepare action for RSSM (one-hot for discrete)
        if discrete:
            action_for_rssm = jax.nn.one_hot(carry.last_action, action_dim, dtype=jnp.float32)
        else:
            action_for_rssm = carry.last_action

        # RSSM obs_step: update state with new observation
        rng_rssm = rngs_()
        # Reset RSSM state on episode boundaries
        reset_mask = carry.last_is_first.astype(jnp.float32)
        h_reset = carry.rssm_h * (1.0 - reset_mask[:, None])
        z_reset = carry.rssm_z * (1.0 - reset_mask[:, None])
        action_reset = action_for_rssm * (1.0 - reset_mask[:, None] if action_for_rssm.ndim > 1 else 1.0 - reset_mask)

        h_new, z_new, _, _ = model_.rssm.obs_step(h_reset, z_reset, action_reset, embed_BF, rng_rssm)

        # Select action from actor
        state_feat = jnp.concatenate([h_new, z_new], axis=-1)
        rng_act = rngs_()
        action_env = model_.actor.act(state_feat, rng_act)

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

        transition = DreamerTransition(
            obs=carry.last_obs,
            action=action_env,
            reward=rewards.astype(jnp.float32),
            done=dones,
            is_first=carry.last_is_first,
        )

        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            last_done=dones,
            last_is_first=dones,  # Next step is first if current done
            rssm_h=h_new,
            rssm_z=z_new,
            last_action=action_env,
            global_step=carry.global_step + num_envs,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry, transition

    # --- Sequence sampling (strided for multi-env interleaving) ---
    def sample_sequences(buffer, rng_sample):
        """Sample temporal sequences accounting for interleaved multi-env storage."""
        n = buffer.num_entries
        num_timesteps = n // num_envs
        seq_len = config.sequence_length
        max_start_t = jnp.maximum(num_timesteps - seq_len, 1)
        rng_t, rng_e = jax.random.split(rng_sample)
        start_timesteps = jax.random.randint(rng_t, (config.batch_size,), 0, max_start_t)
        env_ids = jax.random.randint(rng_e, (config.batch_size,), 0, num_envs)
        offsets = jnp.arange(seq_len)
        # indices[i, j] = env_ids[i] + (start_timesteps[i] + j) * num_envs
        indices = env_ids[:, None] + (start_timesteps[:, None] + offsets[None, :]) * num_envs
        indices = indices % buffer.size
        return jax.tree.map(lambda arr: arr[indices], buffer.data)

    # --- Joint loss function ---
    def dreamer_loss(model_, slow_critic_, sequences, obs_rms, return_ema_low, return_ema_high, rng):
        """Joint world model + imagination loss.

        sequences: pytree with shape (batch_size, seq_len, ...)
        """
        B_seq = config.batch_size
        T = config.sequence_length
        rng_wm, rng_imag = jax.random.split(rng)

        # === World Model Loss ===
        obs_BT = sequences.obs  # (B, T, D)
        action_BT = sequences.action  # (B, T) or (B, T, A)
        reward_BT = sequences.reward  # (B, T)
        done_BT = sequences.done  # (B, T)
        is_first_BT = sequences.is_first  # (B, T)

        # Normalize obs for network
        obs_flat = obs_BT.reshape(B_seq * T, *obs_BT.shape[2:])
        if config.normalize_observations:
            obs_flat = normalize_obs(obs_rms, obs_flat)

        # Encode all obs
        embed_flat = model_.encoder(obs_flat)  # (B*T, F)
        embed_BT = embed_flat.reshape(B_seq, T, -1)

        # Prepare actions for RSSM
        if discrete:
            action_rssm_BT = jax.nn.one_hot(action_BT, action_dim, dtype=jnp.float32)  # (B,T,A)
        else:
            action_rssm_BT = action_BT  # (B, T, A)

        # Scan RSSM obs_step over sequence
        h_init_B, z_init_B = model_.rssm.initial_state(B_seq)

        def rssm_scan_step(carry_inner, t_data):
            h, z = carry_inner
            embed_t, action_t, is_first_t, rng_t = t_data
            # Reset RSSM state where is_first=True
            reset = is_first_t.astype(jnp.float32)
            h = h * (1.0 - reset[:, None])
            z = z * (1.0 - reset[:, None])
            action_t = action_t * (1.0 - reset[:, None])
            h_new, z_new, prior_logits, post_logits = model_.rssm.obs_step(h, z, action_t, embed_t, rng_t)
            return (h_new, z_new), (h_new, z_new, prior_logits, post_logits)

        rssm_rngs = jax.random.split(rng_wm, T)
        # Transpose to (T, B, ...) for scan
        if action_rssm_BT.ndim == 3:
            scan_actions = jnp.transpose(action_rssm_BT, (1, 0, 2))  # (B,T,A) -> (T,B,A)
        else:
            scan_actions = action_rssm_BT.T  # (B,T) -> (T,B)
        scan_inputs = (
            jnp.transpose(embed_BT, (1, 0, 2)),
            scan_actions,
            is_first_BT.T,
            rssm_rngs,
        )

        _, (h_all_TB, z_all_TB, prior_all_TB, post_all_TB) = jax.lax.scan(rssm_scan_step, (h_init_B, z_init_B), scan_inputs)
        # Transpose back to (B, T, ...)
        h_all_BT = jnp.transpose(h_all_TB, (1, 0, 2))  # (B, T, G)
        z_all_BT = jnp.transpose(z_all_TB, (1, 0, 2))  # (B, T, Z)
        prior_all_BT = jnp.transpose(prior_all_TB, (1, 0, 2, 3))  # (B, T, C, L)
        post_all_BT = jnp.transpose(post_all_TB, (1, 0, 2, 3))  # (B, T, C, L)

        # Latent features: concat(h, z)
        feat_BT = jnp.concatenate([h_all_BT, z_all_BT], axis=-1)  # (B, T, G+Z)
        feat_flat = feat_BT.reshape(B_seq * T, state_dim)

        # Decoder loss: MSE in symlog space
        obs_for_recon = obs_BT.reshape(B_seq * T, -1)
        pred_symlog_obs = model_.decoder(feat_flat)  # (B*T, D)
        target_symlog_obs = symlog(obs_for_recon)
        recon_loss = jnp.mean((pred_symlog_obs - target_symlog_obs) ** 2)

        # Reward loss: cross-entropy on symlog two-hot
        reward_logits = model_.reward_pred(feat_flat)  # (B*T, N)
        reward_target = twohot_encode(reward_BT.reshape(-1), config.value_num_bins)
        reward_loss = -jnp.mean(jnp.sum(reward_target * jax.nn.log_softmax(reward_logits, axis=-1), axis=-1))

        # Continue loss: binary cross-entropy
        # Continue target: (1 - done) * gamma per contdisc pattern
        continue_target = (1.0 - done_BT.astype(jnp.float32)).reshape(-1) * config.gamma
        continue_logits = model_.continue_pred(feat_flat)  # (B*T,)
        continue_loss = -jnp.mean(
            continue_target * jax.nn.log_sigmoid(continue_logits)
            + (1.0 - continue_target) * jax.nn.log_sigmoid(-continue_logits)
        )

        # KL losses with free bits
        # dyn_kl: KL[sg(post) || prior] — trains prior to match posterior
        dyn_kl_BT = categorical_kl(
            jax.lax.stop_gradient(post_all_BT).reshape(B_seq * T, rssm_cfg.stoch_categories, rssm_cfg.stoch_classes),
            prior_all_BT.reshape(B_seq * T, rssm_cfg.stoch_categories, rssm_cfg.stoch_classes),
            unimix=rssm_cfg.unimix,
        )
        dyn_kl = jnp.maximum(dyn_kl_BT, config.kl_free_bits).mean()

        # rep_kl: KL[post || sg(prior)] — trains posterior to be predictable
        rep_kl_BT = categorical_kl(
            post_all_BT.reshape(B_seq * T, rssm_cfg.stoch_categories, rssm_cfg.stoch_classes),
            jax.lax.stop_gradient(prior_all_BT).reshape(B_seq * T, rssm_cfg.stoch_categories, rssm_cfg.stoch_classes),
            unimix=rssm_cfg.unimix,
        )
        rep_kl = jnp.maximum(rep_kl_BT, config.kl_free_bits).mean()

        wm_loss = (
            config.recon_scale * recon_loss
            + config.reward_scale * reward_loss
            + config.continue_scale * continue_loss
            + config.dyn_scale * dyn_kl
            + config.rep_scale * rep_kl
        )

        # === Imagination ===
        # Start from all posterior states (detached)
        start_h = jax.lax.stop_gradient(h_all_BT.reshape(B_seq * T, -1))  # (BT, G)
        start_z = jax.lax.stop_gradient(z_all_BT.reshape(B_seq * T, -1))  # (BT, Z)

        H = config.imagination_horizon
        BT = B_seq * T

        def img_step(carry_img, rng_step):
            h, z = carry_img
            # Stop-gradient state for actor (pure REINFORCE, no indirect gradients)
            feat_sg = jnp.concatenate([jax.lax.stop_gradient(h), jax.lax.stop_gradient(z)], axis=-1)
            action, _ = model_.actor(feat_sg, rng_step)
            # Store action index for log_prob_of (discrete: argmax of ST one-hot, continuous: raw)
            action_idx = jnp.argmax(action, axis=-1) if discrete else action
            rng_img_step = jax.random.fold_in(rng_step, 1)
            h_new, z_new, _ = model_.rssm.img_step(h, z, action, rng_img_step)
            return (h_new, z_new), (h_new, z_new, action_idx)

        img_rngs = jax.random.split(rng_imag, H)
        _, (img_h_HBT, img_z_HBT, img_actions_HBT) = jax.lax.scan(img_step, (start_h, start_z), img_rngs)
        # img_h_HBT: (H, BT, G), img_actions_HBT: (H, BT) or (H, BT, A)

        # Prepend start states: total horizon = H+1
        all_h = jnp.concatenate([start_h[None], img_h_HBT], axis=0)  # (H+1, BT, G)
        all_z = jnp.concatenate([start_z[None], img_z_HBT], axis=0)  # (H+1, BT, Z)
        all_feat = jnp.concatenate([all_h, all_z], axis=-1)  # (H+1, BT, G+Z)

        # Predict rewards and continues from imagined states (t=1..H)
        img_feat_flat = all_feat[1:].reshape(H * BT, state_dim)
        img_reward_logits = model_.reward_pred(img_feat_flat)
        img_rewards = twohot_decode(img_reward_logits, config.value_num_bins).reshape(H, BT)
        img_continue_logits = model_.continue_pred(img_feat_flat)
        img_continues = jax.nn.sigmoid(img_continue_logits).reshape(H, BT)

        # Critic values at all timesteps (t=0..H)
        # Compute critic logits once, reuse for values (sg'd) and loss (gradient-bearing)
        sg_all_feat = jax.lax.stop_gradient(all_feat.reshape((H + 1) * BT, state_dim))
        all_critic_logits = model_.critic(sg_all_feat)  # ((H+1)*BT, N) — single forward pass
        online_values = jax.lax.stop_gradient(twohot_decode(all_critic_logits, config.value_num_bins).reshape(H + 1, BT))

        # Slow critic values for regularization only
        slow_values = jax.lax.stop_gradient(twohot_decode(slow_critic_(sg_all_feat), config.value_num_bins).reshape(H + 1, BT))

        # Lambda returns (reverse accumulation) using online critic
        # Continue target = (1-done) * gamma, so predictions already encode discount
        disc_H = img_continues  # (H, BT)

        def lambda_scan(next_val, t_data):
            rew, disc, cont_val = t_data
            target = rew + disc * ((1.0 - config.lambda_gae) * cont_val + config.lambda_gae * next_val)
            return target, target

        # Reverse scan: from t=H-1 to t=0
        bootstrap = online_values[H]
        _, lambda_returns = jax.lax.scan(
            lambda_scan,
            bootstrap,
            (img_rewards[::-1], disc_H[::-1], online_values[1:][::-1]),
        )
        lambda_returns = lambda_returns[::-1]  # (H, BT), returns for t=0..H-1

        # Return normalization: percentile EMA
        returns_flat = lambda_returns.reshape(-1)
        lo = jnp.percentile(returns_flat, config.return_norm_perclo)
        hi = jnp.percentile(returns_flat, config.return_norm_perchi)
        rate = config.return_norm_rate
        new_ema_low = return_ema_low * (1.0 - rate) + lo * rate
        new_ema_high = return_ema_high * (1.0 - rate) + hi * rate
        scale = jnp.maximum(new_ema_high - new_ema_low, config.return_norm_limit)

        # Weight: cumprod of discount (for loss weighting)
        log_disc = jnp.log(jnp.maximum(disc_H, 1e-8))
        log_cumdisc = jnp.cumsum(log_disc, axis=0)  # (H, BT)
        weight = jnp.concatenate([jnp.ones((1, BT)), jnp.exp(log_cumdisc)], axis=0)  # (H+1, BT)
        weight = jax.lax.stop_gradient(weight[:H])  # (H, BT), for t=0..H-1

        # Actor loss: recompute log_probs on sg(states) for pure REINFORCE
        # (reference: pol = self.actor(sg(traj)); logpi = pol.log_prob(sg(action)))
        sg_feat = jax.lax.stop_gradient(all_feat[:H]).reshape(H * BT, state_dim)
        sg_actions = jax.lax.stop_gradient(img_actions_HBT).reshape(H * BT, *img_actions_HBT.shape[2:])
        reinforce_logprob = model_.actor.log_prob_of(sg_feat, sg_actions).reshape(H, BT)
        actor_entropy = model_.actor.entropy(sg_feat).reshape(H, BT)

        # Normalized advantage using online critic values
        advantages = (lambda_returns - online_values[:H]) / scale
        advantages = jax.lax.stop_gradient(advantages)

        actor_loss = -jnp.mean(weight * (reinforce_logprob * advantages + config.ent_coef * actor_entropy))

        # Critic loss: reuse logits from the single forward pass above (first H of H+1 states)
        target_returns = jax.lax.stop_gradient(lambda_returns)  # (H, BT)
        target_twohot = twohot_encode(target_returns.reshape(-1), config.value_num_bins)  # (H*BT, N)
        critic_logits = all_critic_logits[: H * BT]  # (H*BT, N) — reuse, no second forward pass
        critic_xent = -jnp.sum(target_twohot * jax.nn.log_softmax(critic_logits, axis=-1), axis=-1)

        # Slow value regularization
        slow_target_twohot = twohot_encode(slow_values[:H].reshape(-1), config.value_num_bins)
        slow_reg = -jnp.sum(slow_target_twohot * jax.nn.log_softmax(critic_logits, axis=-1), axis=-1)

        weight_flat = weight.reshape(-1)
        critic_loss = jnp.mean(weight_flat * (critic_xent + config.slow_value_reg * slow_reg))

        # Mean entropy for metrics
        mean_entropy = jnp.mean(actor_entropy)

        # Total loss
        total_loss = wm_loss + config.policy_scale * actor_loss + config.value_scale * critic_loss

        metrics = {
            "world_model_loss": wm_loss,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "continue_loss": continue_loss,
            "dyn_kl": dyn_kl,
            "rep_kl": rep_kl,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": mean_entropy,
        }
        return total_loss, (metrics, new_ema_low, new_ema_high)

    # --- Train iteration ---
    def train_iteration(carry, model_, optimizer_, slow_critic_, rngs_):
        start_training = carry.global_step > config.fill_buffer

        carry, batch = collect_transition(carry, model_, rngs_)
        carry = carry._replace(buffer=carry.buffer.extend(batch))

        zero_update_metrics = {k: jnp.float32(0.0) for k in _DREAMER_METRIC_NAMES}

        def update_iteration(_, state):
            carry_inner, model_inner, optimizer_inner, slow_critic_inner, rngs_inner, metrics_sum = state
            rng_sample = rngs_inner()
            seqs = sample_sequences(carry_inner.buffer, rng_sample)

            rng_loss = rngs_inner()
            (_, (step_metrics, new_ema_low, new_ema_high)), grads = nnx.value_and_grad(dreamer_loss, has_aux=True)(
                model_inner,
                slow_critic_inner,
                seqs,
                carry_inner.obs_rms,
                carry_inner.return_ema_low,
                carry_inner.return_ema_high,
                rng_loss,
            )
            optimizer_inner.update(model_inner, grads)

            # Polyak update slow critic
            _polyak_update(model_inner.critic, slow_critic_inner, 1.0 - config.slow_value_rate)

            carry_inner = carry_inner._replace(
                return_ema_low=new_ema_low,
                return_ema_high=new_ema_high,
            )

            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, step_metrics)
            return (carry_inner, model_inner, optimizer_inner, slow_critic_inner, rngs_inner, metrics_sum)

        def do_updates(carry_u, model_u, optimizer_u, slow_critic_u, rngs_u):
            carry_u, model_u, optimizer_u, slow_critic_u, rngs_u, metrics_sum = nnx.fori_loop(
                0,
                config.num_epochs,
                update_iteration,
                (carry_u, model_u, optimizer_u, slow_critic_u, rngs_u, zero_update_metrics),
            )
            iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry_u, model_u, optimizer_u, slow_critic_u, rngs_u, iter_metrics

        def no_updates(carry_u, model_u, optimizer_u, slow_critic_u, rngs_u):
            return carry_u, model_u, optimizer_u, slow_critic_u, rngs_u, zero_update_metrics

        carry, model_, optimizer_, slow_critic_, rngs_, update_metrics = nnx.cond(
            start_training, do_updates, no_updates, carry, model_, optimizer_, slow_critic_, rngs_
        )

        # Accumulate metrics (only when training)
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, update_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, model_, optimizer_, slow_critic_, rngs_

    # --- Stateful eval with RSSM carry ---
    def eval_single(model_, obs_rms, rng_eval):
        """Evaluate one episode maintaining RSSM state across steps."""
        rng_reset, rng_ep = jax.random.split(rng_eval)
        obs, env_st = env.reset(rng_reset, env_params)
        h, z = model_.rssm.initial_state(1)
        last_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

        init_carry = EvalCarry(
            rng=rng_ep,
            env_st=env_st,
            obs=obs,
            h=h,
            z=z,
            last_action=last_action,
            done=jnp.bool_(False),
            ret=jnp.float32(0.0),
            length=jnp.int32(0),
        )

        def step_fn(carry):
            rng, _rng_act, rng_step, rng_rssm = jax.random.split(carry.rng, 4)
            obs_1D = carry.obs
            if config.normalize_observations:
                obs_1D = normalize_obs(obs_rms, obs_1D)
            obs_1BD = jnp.expand_dims(obs_1D, 0)
            embed = model_.encoder(obs_1BD)
            h_new, z_new, _, post_logits = model_.rssm.obs_step(carry.h, carry.z, carry.last_action, embed, rng_rssm)
            z_new = model_.rssm._sample_stoch(post_logits, rng_rssm)
            feat = jnp.concatenate([h_new, z_new], axis=-1)
            action = model_.actor.act_greedy(feat)
            action_env = action.squeeze(0)
            next_obs, new_env_st, reward, done, _ = env.step(rng_step, carry.env_st, action_env, env_params)
            # Action for next RSSM step
            if discrete:
                next_action = jax.nn.one_hot(action, action_dim, dtype=jnp.float32)  # keep (1, A)
            else:
                next_action = action  # (1, A)
            # Reset RSSM on done
            reset = done.astype(jnp.float32)
            h_new = h_new * (1.0 - reset)
            z_new = z_new * (1.0 - reset)
            next_action = next_action * (1.0 - reset)
            return EvalCarry(
                rng=rng,
                env_st=new_env_st,
                obs=next_obs,
                h=h_new,
                z=z_new,
                last_action=next_action,
                done=done,
                ret=carry.ret + reward.squeeze(),
                length=carry.length + 1,
            )

        final = jax.lax.while_loop(
            lambda c: jnp.logical_and(c.length < max_steps, jnp.logical_not(c.done)),
            step_fn,
            init_carry,
        )
        return final.length, final.ret

    def eval_callback(carry, model_, rngs_):
        seeds = jax.random.split(rngs_(), 128)
        return jax.vmap(lambda s: eval_single(model_, carry.obs_rms, s))(seeds)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, model, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _DREAMER_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, model_, optimizer_, slow_critic_, rngs_ = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, model_, optimizer_, slow_critic_, rngs_ = nnx.fori_loop(
            0, steps_per_iter, train_body, (carry, model_, optimizer_, slow_critic_, rngs_)
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, model_, rngs_)
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

        return (carry, model_, optimizer_, slow_critic_, rngs_), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, model, optimizer, slow_critic, rngs), all_metrics = scan_eval(
        (carry, model, optimizer, slow_critic, rngs), jnp.zeros(num_evals)
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "model": model,
        "optimizer": optimizer,
        "slow_critic": slow_critic,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
