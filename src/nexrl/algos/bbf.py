"""BBF — Bigger, Better, Faster: C51 + SPR + periodic resets + annealing.

"Bigger, Better, Faster: Human-level Atari with human-level efficiency"
(Schwarzer et al., ICML 2023)

Standalone training function. Builds on Rainbow (C51 + PER + dueling + DDQN) with:
- SPR self-predictive representation auxiliary loss
- Scaled IMPALA ResNet encoder (width_scale=4)
- Periodic shrink-and-perturb resets with offline warmup
- Receding n-step horizon and increasing gamma annealing
- EMA target network (tau=0.005)
- AdamW weight decay
- Data augmentation (random shift)

Following the reference implementation's compilation strategy: the training
step (gradient update) is JIT-compiled separately from collection and reset
logic. A Python outer loop orchestrates collection, update epochs, periodic
resets, and evaluation — avoiding the enormous XLA graph that results from
compiling the entire training loop end-to-end.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (num_actions)
    N: number of atoms (n_atoms)
    F: features_dim (encoder output; also SPR projection/target/predictor dim)
    J: jumps (SPR prediction horizon)
    K: subsequence length (max(max_n_step, jumps+1))
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

from nexrl.algos.utils import (
    _polyak_update,
    create_encoder,
    create_env,
    create_vec_env_fns,
    make_eval_act,
)
from nexrl.buffers import BBFReplayBuffer, Minibatch
from nexrl.configs import BBFConfig
from nexrl.evaluate import evaluate
from nexrl.networks.bbf import BBFNetwork, _gather_action_atoms, spr_loss, spr_rollout, spr_targets
from nexrl.networks.sgf import random_intensity, random_shift
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_BBF_METRIC_NAMES = ("cross_entropy_loss", "spr_loss", "total_loss", "mean_q_value", "mean_priority", "current_n_step")


class BBFCarry(NamedTuple):
    """Non-module carry state for BBF training."""

    env_state: Any
    last_obs: chex.Array
    global_step: int
    replay_buffer: BBFReplayBuffer
    max_priority: chex.Array
    cycle_grad_steps: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


# ---------------------------------------------------------------------------
# Annealing schedules (pure functions of cycle_grad_steps)
# ---------------------------------------------------------------------------


def _anneal_n_step(
    cycle_grad_steps: jax.Array,
    max_n: int,
    min_n: int,
    cycle_steps: int,
) -> jax.Array:
    """Exponential decay from max_n to min_n over cycle_steps, rounded to an integer.

    Schwarzer rounds the annealed horizon to an int (``int(round(...))``) so the
    n-step return sums exactly ``n`` rewards AND bootstraps with ``gamma**n`` at the
    state ``n`` steps ahead — consistent. Returning a non-integer would sum
    ``ceil(n)`` rewards but discount by ``gamma**n`` (a slight mismatch).
    """
    frac = jnp.minimum(cycle_grad_steps.astype(jnp.float32) / cycle_steps, 1.0)
    log_max = jnp.log(jnp.float32(max_n))
    log_min = jnp.log(jnp.float32(min_n))
    return jnp.round(jnp.exp(log_max + (log_min - log_max) * frac))


def _anneal_gamma(
    cycle_grad_steps: jax.Array,
    max_gamma: float,
    min_gamma: float,
    cycle_steps: int,
) -> jax.Array:
    """Exponential increase from min_gamma to max_gamma over cycle_steps."""
    frac = jnp.minimum(cycle_grad_steps.astype(jnp.float32) / cycle_steps, 1.0)
    log_max = jnp.log(jnp.float32(max_gamma))
    log_min = jnp.log(jnp.float32(min_gamma))
    return jnp.exp(log_min + (log_max - log_min) * frac)


# ---------------------------------------------------------------------------
# On-the-fly n-step return from subsequence
# ---------------------------------------------------------------------------


def _compute_nstep_from_subsequence(
    rewards_BK: jax.Array,
    dones_BK: jax.Array,
    next_obs_BK: jax.Array,
    current_n: jax.Array,
    gamma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute variable n-step return from a max_n_step-length subsequence.

    Args:
        rewards_BK: Rewards for each step. (B, K) where K = max_n_step.
        dones_BK: Done flags. (B, K).
        next_obs_BK: Next observations. (B, K, *obs_shape).
        current_n: Annealed n-step value (traced float).
        gamma: Annealed discount factor (traced float).

    Returns:
        n_step_return_B: Discounted n-step return. (B,)
        bootstrap_discount_B: gamma^n * (1 - done_in_window). (B,)
        effective_next_obs_B: Next obs for bootstrap target. (B, *obs_shape)
    """
    B, K = rewards_BK.shape
    step_indices_K = jnp.arange(K, dtype=jnp.float32)

    # Active steps: those within the current annealed n
    active_BK = (step_indices_K[None, :] < current_n).astype(jnp.float32)  # (1, K) broadcast

    # Episode boundary: cumulative done scan
    dones_float_BK = dones_BK.astype(jnp.float32)
    cum_done_BK = jnp.cumsum(dones_float_BK, axis=1)
    # not_done_yet[k] = 1 if no done at steps 0..k-1
    shifted_BK = jnp.concatenate([jnp.zeros((B, 1), dtype=jnp.float32), cum_done_BK[:, :-1]], axis=1)
    not_done_yet_BK = (shifted_BK == 0).astype(jnp.float32)

    # Combined mask and discounts
    mask_BK = active_BK * not_done_yet_BK
    discounts_K = gamma**step_indices_K

    # N-step return
    n_step_return_B = jnp.sum(rewards_BK * mask_BK * discounts_K[None, :], axis=1)

    # Bootstrap discount: gamma^current_n if no done in window, else 0
    done_in_window_B = jnp.any((dones_float_BK * mask_BK) > 0, axis=1)
    bootstrap_discount_B = jnp.where(done_in_window_B, 0.0, gamma**current_n)

    # Effective next obs: last non-terminal next_obs within the n-step window
    step_weight_BK = mask_BK * step_indices_K[None, :]
    done_and_active_BK = dones_float_BK * active_BK * not_done_yet_BK
    has_terminal_B = jnp.any(done_and_active_BK > 0, axis=1)
    terminal_idx_B = jnp.argmax(done_and_active_BK, axis=1)

    last_active_idx_B = jnp.argmax(step_weight_BK, axis=1)
    eff_idx_B = jnp.where(has_terminal_B, terminal_idx_B, last_active_idx_B)

    effective_next_obs_B = next_obs_BK[jnp.arange(B), eff_idx_B]

    return n_step_return_B, bootstrap_discount_B, effective_next_obs_B


# ---------------------------------------------------------------------------
# Shrink-and-perturb reset
# ---------------------------------------------------------------------------


def _is_sp_path(path) -> bool:
    """True for encoder/transition leaves — the BBF shrink-and-perturb subset.

    Shared by ``_build_sp_mask`` (online/target S&P mask) and ``_reset_opt_leaf``
    inside ``jit_reset`` (Adam moment preservation), so the two splits cannot
    drift apart by accident.
    """
    path_str = ".".join(str(p) for p in path)
    return "encoder" in path_str or "transition" in path_str


def _build_sp_mask(params: nnx.State) -> nnx.State:
    """Build a mask: 1.0 for encoder/transition params (S&P), 0.0 for head/proj/pred (full reset)."""
    return jax.tree.map_with_path(
        lambda path, val: jnp.ones_like(val) if _is_sp_path(path) else jnp.zeros_like(val),
        params,
    )


# ---------------------------------------------------------------------------
# Network creation
# ---------------------------------------------------------------------------


def _create_networks(
    config: BBFConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[BBFNetwork, nnx.Optimizer, BBFNetwork, Callable[[jax.Array], nnx.State]]:
    """Create BBF online network, optimizer, target network, and a fresh-params factory.

    The returned ``make_fresh_params(reset_rng)`` constructs a throwaway
    ``BBFNetwork`` with a fresh ``nnx.Rngs`` and returns its ``nnx.Param`` state.
    It is the shrink-and-perturb random source — draws come from the network's
    actual initializers (Conv/Dense kernel inits, biases, layer scales), matching
    Schwarzer's ``network_def.init(...)`` rather than a hand-rolled fan-in
    heuristic. Closes a subtle fidelity gap inside the most important BBF
    mechanism.
    """
    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)

    rngs = nnx.Rngs(rng)

    compute_dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32

    if config.encoder_kwargs is not None:
        features_dim = config.encoder_kwargs.features_dim
        in_features = 0
    else:
        features_dim = config.hidden_dim
        in_features = int(np.prod(obs_space.shape))

    # Single source of truth for encoder construction — used both at training
    # start (for `network` / `target`) and inside `make_fresh_params` (for each
    # reset's random draw). `return_spatial=True`: encoder outputs the spatial
    # latent so the convolutional SPR transition can operate on it; BBFNetwork
    # owns the shared projection.
    def _make_encoder(encoder_rngs: nnx.Rngs) -> Any:
        if config.encoder_kwargs is None:
            return None
        return create_encoder(
            config.encoder_kwargs,
            obs_space.shape,
            encoder_rngs,
            compute_dtype=compute_dtype,
            return_spatial=True,
        )

    encoder = _make_encoder(rngs)
    encoder_t = _make_encoder(rngs)

    net_kwargs = {
        "action_dim": action_dim,
        "n_atoms": config.n_atoms,
        "v_min": config.v_min,
        "v_max": config.v_max,
        "features_dim": features_dim,
        "in_features": in_features,
    }

    network = BBFNetwork(**net_kwargs, encoder=encoder, rngs=rngs, compute_dtype=compute_dtype)
    target = BBFNetwork(**net_kwargs, encoder=encoder_t, rngs=rngs, compute_dtype=compute_dtype)

    def wd_mask(params):
        def _check(path, _):
            name = str(path[-1]) if path else ""
            return name not in ("bias", "scale")

        return jax.tree.map_with_path(_check, params)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1.5e-4,
            mask=wd_mask,
        ),
    )
    optimizer = nnx.Optimizer(network, tx, wrt=nnx.Param)

    def make_fresh_params(reset_rng: jax.Array) -> nnx.State:
        """Construct a fresh BBFNetwork and return its ``nnx.Param`` state.

        Drawn from the same init path used by `_create_networks` above, so the
        random component of shrink-and-perturb is distributed identically to the
        network's original initialization (kernel inits, bias inits, layer scales).
        Called from Python (not under JIT) — module construction stays in eager
        land; the resulting state is then fed into the jitted reset as data.
        """
        fresh_rngs = nnx.Rngs(reset_rng)
        fresh_encoder = _make_encoder(fresh_rngs)
        fresh_net = BBFNetwork(
            **net_kwargs,
            encoder=fresh_encoder,
            rngs=fresh_rngs,
            compute_dtype=compute_dtype,
        )
        return nnx.state(fresh_net, nnx.Param)

    return network, optimizer, target, make_fresh_params


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_bbf(
    config: BBFConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a BBF agent.

    BBF combines Rainbow (C51 + PER + dueling + DDQN) with SPR auxiliary loss,
    a scaled IMPALA encoder, periodic shrink-and-perturb resets, receding
    n-step/gamma annealing, EMA target network, and AdamW weight decay.

    Uses the reference implementation's compilation strategy: separately JIT'd
    functions for collection, gradient updates, and resets, connected by a
    Python outer loop.

    Args:
        config: BBFConfig with hyperparameters.
        rng: PRNG key. Use ``jax.vmap`` over different keys for independent seeds.
        env: Optional gymnax-compatible environment.
        env_params: Optional environment parameters.
        log_fn: Optional callback invoked at each evaluation point via
            ``jax.debug.callback``.

    Returns:
        A ``(state, metrics)`` tuple.
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]
    assert discrete, "BBF requires discrete action space"
    subseq_len = max(config.max_n_step, config.jumps + 1)

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    rngs = nnx.Rngs(rng)
    network, optimizer, target, make_fresh_params = _create_networks(config, env, env_params, rngs())

    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    buf = BBFReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    # Pre-compute shrink-and-perturb mask once (pytree of 0/1 matching param shapes)
    sp_mask = _build_sp_mask(nnx.state(network, nnx.Param))

    carry = BBFCarry(
        env_state=env_state,
        last_obs=obs,
        global_step=0,
        replay_buffer=buf,
        max_priority=jnp.float32(1.0),
        cycle_grad_steps=0,
        running_return=jnp.zeros(num_envs, dtype=jnp.float32),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # -----------------------------------------------------------------------
    # JIT'd collection: one env step + buffer extend
    # -----------------------------------------------------------------------
    @nnx.jit
    def jit_collect(carry, q_net, target_net, rngs):
        rng_action = rngs()

        epsilon = jnp.where(
            carry.global_step < config.epsilon_decay_period,
            config.epsilon_train + (1.0 - config.epsilon_train) * (1.0 - carry.global_step / config.epsilon_decay_period),
            config.epsilon_train,
        )

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs))

        def sample_policy(rng):
            select_net = target_net if config.target_action_selection else q_net
            return select_net.act(carry.last_obs, rng, epsilon=epsilon)

        uniform = carry.global_step < config.fill_buffer
        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, rewards, dones, _ = vmap_step(rng_steps, carry.env_state, actions, env_params)

        # Reward clipping for the TRAINING signal only (matches Schwarzer's runner,
        # which clips the agent/buffer reward to [-1, 1]). running_return below uses
        # the RAW reward so train_return stays the true game score — the reference
        # likewise reports cum_reward unclipped. Decoupling here keeps the env
        # adapter untouched.
        train_rewards = jnp.clip(rewards, -1.0, 1.0) if config.clip_rewards else rewards

        new_running_B = carry.running_return + rewards  # RAW reward for the reported metric
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * dones)
        completed_count = carry.completed_returns_count + jnp.sum(dones.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - dones)

        mb = Minibatch(
            obs=carry.last_obs,
            action=actions,
            reward=train_rewards,
            done=dones,
            next_obs=next_obs,
        )
        priorities_B = jnp.full(num_envs, carry.max_priority**config.per_alpha)
        new_buf = carry.replay_buffer.extend_with_priorities(mb, priorities_B)

        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            replay_buffer=new_buf,
            running_return=reset_running_B,
            completed_returns_sum=completed_sum,
            completed_returns_count=completed_count,
        )
        return carry

    # -----------------------------------------------------------------------
    # Single gradient update step (used inside jit_update_epoch's fori_loop)
    # -----------------------------------------------------------------------
    def _update_q(carry, q_net, q_opt, q_tgt, rngs):
        support_N = q_net.support

        subseq_mb, same_traj_BK, indices, weights_B = carry.replay_buffer.sample_with_subsequences(
            config.batch_size, rngs(), config.per_beta, subseq_len, num_envs
        )

        current_n = _anneal_n_step(carry.cycle_grad_steps, config.max_n_step, config.n_step, config.cycle_steps)
        current_gamma = _anneal_gamma(carry.cycle_grad_steps, config.gamma, config.min_gamma, config.cycle_steps)

        # random_shift is dtype-agnostic (pad + dynamic_slice). Augment in uint8
        # so the spatial crop is precision-free, then apply intensity aug
        # (matches Schwarzer's drq_image_aug: per-image multiplicative noise on
        # [0, 1] floats). uint8 * float -> float promotes via JAX type rules; the
        # encoder's `.astype(compute_dtype) / 255.0` then makes it [0, ~1.1].
        current_obs_BCHW = subseq_mb.obs[:, 0]  # (B, C, H, W) uint8
        if config.data_augmentation:
            current_obs_BCHW = random_shift(rngs(), current_obs_BCHW, pad=config.shift_pad)
            current_obs_BCHW = random_intensity(rngs(), current_obs_BCHW)

        n_step_return_B, bootstrap_discount_B, effective_next_obs_B = _compute_nstep_from_subsequence(
            subseq_mb.reward[:, : config.max_n_step],
            subseq_mb.done[:, : config.max_n_step],
            subseq_mb.next_obs[:, : config.max_n_step],
            current_n,
            current_gamma,
        )

        next_obs_aug_BCHW = effective_next_obs_B
        if config.data_augmentation:
            next_obs_aug_BCHW = random_shift(rngs(), next_obs_aug_BCHW, pad=config.shift_pad)
            next_obs_aug_BCHW = random_intensity(rngs(), next_obs_aug_BCHW)

        # C51 target (DDQN: online selects action, target provides distribution)
        next_features_BF = q_tgt.encode(next_obs_aug_BCHW)
        online_next_features_BF = q_net.encode(next_obs_aug_BCHW)

        online_next_q_BA = jnp.sum(jax.nn.softmax(q_net.q_logits(online_next_features_BF), axis=-1) * support_N, axis=-1)
        next_action_B = jnp.argmax(online_next_q_BA, axis=1)

        target_probs_BN = q_tgt.take_probs(next_features_BF, next_action_B)
        target_support_BN = n_step_return_B[:, None] + bootstrap_discount_B[:, None] * support_N[None, :]

        projected_BN = jnp.asarray(
            jax.vmap(rlax.categorical_l2_project, in_axes=(0, 0, None))(target_support_BN, target_probs_BN, support_N)
        )
        projected_BN = jax.lax.stop_gradient(projected_BN)

        # SPR targets from target encoder
        future_obs_JB = subseq_mb.obs[:, 1 : config.jumps + 1]  # (B, J, C, H, W)
        future_obs_JBCHW = jnp.transpose(future_obs_JB, (1, 0, *range(2, future_obs_JB.ndim)))  # (J, B, C, H, W)
        spr_tgt_JBP = spr_targets(q_tgt, future_obs_JBCHW)

        spr_actions_BJ = subseq_mb.action[:, : config.jumps]  # (B, J)
        spr_actions_JB = spr_actions_BJ.T  # (J, B)

        spr_mask_BJ = same_traj_BK[:, 1 : config.jumps + 1]
        spr_mask_JB = spr_mask_BJ.T  # (J, B)

        def loss_fn(model):
            # Encode to the spatial latent once: project it for the Q-head, and
            # roll the convolutional SPR transition forward from the same latent.
            spatial_BHWC = model.encode_spatial(current_obs_BCHW)
            features_BF = model.project(spatial_BHWC)

            # Compute q_logits once and derive both log_probs (for CE) and
            # mean_q metric from it. XLA likely CSEs this anyway but explicit
            # is clearer. Cast to fp32 — bf16 log_softmax collapses CE training
            # (Flax #677) and the C51 projected target is fp32.
            logits_BAN = model.q_logits(features_BF).astype(jnp.float32)
            log_probs_BAN = jax.nn.log_softmax(logits_BAN, axis=-1)
            probs_BAN = jax.nn.softmax(logits_BAN, axis=-1)

            action_B = subseq_mb.action[:, 0]
            log_probs_BN = _gather_action_atoms(log_probs_BAN, action_B)
            safe_log_BN = jnp.where(projected_BN > 0, log_probs_BN, 0.0)
            ce_B = -jnp.sum(projected_BN * safe_log_BN, axis=-1)

            predictions_JBP = spr_rollout(model, spatial_BHWC, spr_actions_JB)
            spr_loss_B = spr_loss(predictions_JBP, spr_tgt_JBP, spr_mask_JB)

            total_B = ce_B + config.spr_weight * spr_loss_B
            loss = jnp.mean(weights_B * total_B)

            q_BA = jnp.sum(probs_BAN * support_N, axis=-1)
            mean_q = q_BA.mean()
            return loss, (ce_B, spr_loss_B, mean_q)

        (loss, (ce_B, spr_loss_B, mean_q)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_net)
        q_opt.update(q_net, grads)

        new_priorities_B = ce_B + config.per_eps
        new_buf = carry.replay_buffer.update_priorities(indices, new_priorities_B**config.per_alpha)
        new_max_priority = jnp.maximum(carry.max_priority, new_priorities_B.max())

        _polyak_update(q_net, q_tgt, 1.0 - config.target_update_tau)

        carry = carry._replace(
            replay_buffer=new_buf,
            max_priority=new_max_priority,
            cycle_grad_steps=carry.cycle_grad_steps + 1,
        )

        iter_metrics = {
            "cross_entropy_loss": jnp.mean(ce_B),
            "spr_loss": jnp.mean(spr_loss_B),
            "total_loss": loss,
            "mean_q_value": mean_q,
            "mean_priority": new_priorities_B.mean(),
            "current_n_step": current_n,
        }
        return carry, iter_metrics

    # -----------------------------------------------------------------------
    # JIT'd update epoch: num_epochs gradient steps via nnx.fori_loop
    # No nnx.cond for start_training (Python handles it).
    # No nnx.cond for reset (Python handles it).
    # -----------------------------------------------------------------------
    @nnx.jit
    def jit_update_epoch(carry, q_net, q_opt, q_tgt, rngs):
        zero_metrics = {k: jnp.float32(0.0) for k in _BBF_METRIC_NAMES}

        def update_body(_, state):
            carry_u, q_net_u, q_opt_u, q_tgt_u, rngs_u, metrics_sum = state
            carry_u, iter_m = _update_q(carry_u, q_net_u, q_opt_u, q_tgt_u, rngs_u)
            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, iter_m)
            return (carry_u, q_net_u, q_opt_u, q_tgt_u, rngs_u, metrics_sum)

        carry, q_net, q_opt, q_tgt, rngs, metrics_sum = nnx.fori_loop(
            0, config.num_epochs, update_body, (carry, q_net, q_opt, q_tgt, rngs, zero_metrics)
        )
        avg_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
        return carry, avg_metrics

    # -----------------------------------------------------------------------
    # JIT'd shrink-and-perturb reset
    #
    # `fresh` and `fresh_target` are drawn in Python by `make_fresh_params`
    # before this call (see the outer-loop reset trigger). Drawing outside the
    # jit lets the fresh state come from a real BBFNetwork construction (real
    # initializers) rather than a fan-in heuristic; the jit body then only does
    # the shrink-and-perturb arithmetic and the optimizer-moment surgery.
    # -----------------------------------------------------------------------
    @nnx.jit
    def jit_reset(q_net, q_opt, q_tgt, carry, fresh, fresh_target):
        # --- Online net: shrink-and-perturb. Encoder/transition (sp_mask==1) keep
        # `shrink*old + perturb*random`; head/projection/predictor (sp_mask==0) are
        # fully re-initialised to `random`. ---
        params = nnx.state(q_net, nnx.Param)

        new_params = jax.tree.map(
            lambda m, old, new: m * (config.shrink_factor * old + config.perturb_factor * new) + (1.0 - m) * new,
            sp_mask,
            params,
            fresh,
        )
        nnx.update(q_net, new_params)

        # --- Deviation #9: preserve encoder/transition Adam moments across resets.
        # Schwarzer re-initialises the optimizer then copies back the
        # (encoder, transition) subtree of the OLD optimizer state. We achieve the
        # same selectively on our single AdamW: keep mu/nu for any leaf identified
        # by `_is_sp_path` (the shrink-perturbed params, whose accumulated moments
        # stay roughly valid), and re-initialise everything else — head/projection/
        # predictor moments AND the shared Adam `count`/`step` (so the preserved
        # encoder moments get a brief bias-correction re-warmup; closing that
        # residual gap exactly needs the two-optimizer split, #10). Sharing
        # `_is_sp_path` with `_build_sp_mask` keeps param S&P and optimizer
        # preservation in lock-step by construction. ---
        def _reset_opt_leaf(path, val):
            if not hasattr(val, "shape"):
                return val
            if _is_sp_path(path):
                return val  # preserve encoder/transition Adam mu, nu
            return jnp.zeros_like(val)  # re-init head/proj/predictor moments + count + step

        new_opt_state = jax.tree.map_with_path(_reset_opt_leaf, nnx.state(q_opt))
        nnx.update(q_opt, new_opt_state)

        # --- Deviation #11: independent target re-initialisation (not a hard copy
        # of the online net). The target undergoes its OWN shrink-and-perturb on its
        # current EMA params, using a SEPARATE random draw — `fresh_target` was
        # drawn from an independent rngs() advance in the Python outer loop, so it
        # differs from the online `fresh`. Same sp_mask split: encoder/transition
        # keep `shrink*target_old + perturb*target_random`, the rest take the
        # target's own random init — a different init than online. ---
        target_params = nnx.state(q_tgt, nnx.Param)
        new_target_params = jax.tree.map(
            lambda m, old, new: m * (config.shrink_factor * old + config.perturb_factor * new) + (1.0 - m) * new,
            sp_mask,
            target_params,
            fresh_target,
        )
        nnx.update(q_tgt, new_target_params)

        carry = carry._replace(cycle_grad_steps=jnp.int32(0))
        return carry

    # -----------------------------------------------------------------------
    # Eval helper
    # -----------------------------------------------------------------------
    def run_eval(q_net, rngs):
        act_fn = make_eval_act(lambda obs, rng: q_net.act(obs, rng, epsilon=config.epsilon_eval), config, None)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # -----------------------------------------------------------------------
    # Python outer loop
    # -----------------------------------------------------------------------
    fill_steps = config.fill_buffer // num_envs
    eval_period = config.eval_freq // num_envs
    total_env_steps = config.total_timesteps // num_envs
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    # Python-tracked reset trigger, in ENV STEPS to match Schwarzer's reference
    # (which compares training_steps — env-step count — to reset_every). Counting
    # in grad steps fires resets RR-times too frequently at high replay ratios.
    next_reset_at = config.reset_every  # in env steps

    # Metrics accumulators stay on-device — only materialized inside the
    # eval-boundary jax.debug.callback. Materializing per-iter (the old
    # `float(update_metrics[k])` pattern) forces a GPU sync every step and
    # blocks JAX from pipelining successive iterations.
    metrics_accum = {k: jnp.float32(0.0) for k in _BBF_METRIC_NAMES}
    metrics_count = jnp.int32(0)

    all_metrics_list = []

    # Optional initial evaluation
    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = run_eval(network, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _BBF_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    for step in range(total_env_steps):
        # --- Collection ---
        carry = jit_collect(carry, network, target, rngs)
        env_steps_collected = (step + 1) * num_envs

        # --- Gradient updates (only after buffer is filled) ---
        if step >= fill_steps:
            carry, update_metrics = jit_update_epoch(carry, network, optimizer, target, rngs)

            metrics_accum = jax.tree.map(lambda a, m: a + m, metrics_accum, update_metrics)
            metrics_count = metrics_count + 1

            # --- Periodic shrink-and-perturb reset (env-step semantics) ---
            if config.reset_every > 0 and env_steps_collected >= next_reset_at and next_reset_at <= config.no_resets_after:
                # Draw the random component of S&P outside JIT — eager construction
                # of a fresh BBFNetwork lets us re-use the network's real init path
                # (kernel inits, biases) rather than a fan-in heuristic. Independent
                # `rngs()` advances for online and target yield independent draws.
                fresh = make_fresh_params(rngs())
                fresh_target = make_fresh_params(rngs())
                carry = jit_reset(network, optimizer, target, carry, fresh, fresh_target)
                next_reset_at += config.reset_every

        # --- Periodic evaluation ---
        if (step + 1) % eval_period == 0:
            # `final_eval_only` runs the costly 128-episode Atari eval just once, at the
            # last eval point; the train_return curve is still logged every eval_period.
            is_final_eval = (step + 1) == total_env_steps
            do_eval = config.run_evaluation and (not config.final_eval_only or is_final_eval)
            if do_eval:
                eval_lengths, eval_returns = run_eval(network, rngs)
            else:
                eval_lengths = jnp.zeros(128, dtype=jnp.float32)
                eval_returns = jnp.zeros(128, dtype=jnp.float32)

            # Divisions stay on-device; jax.debug.callback materializes inside log_fn
            safe_returns_count_f = jnp.maximum(carry.completed_returns_count.astype(jnp.float32), 1.0)
            safe_metrics_count_f = jnp.maximum(metrics_count.astype(jnp.float32), 1.0)
            train_return = carry.completed_returns_sum / safe_returns_count_f

            metrics = {
                "eval_lengths": eval_lengths,
                "eval_returns": eval_returns,
                "train_return": train_return,
                "global_step": jnp.int32((step + 1) * num_envs),
                **{k: metrics_accum[k] / safe_metrics_count_f for k in _BBF_METRIC_NAMES},
            }

            if log_fn is not None:
                jax.debug.callback(log_fn, metrics)

            all_metrics_list.append(metrics)

            # Reset accumulators — stay on device, no host sync
            metrics_accum = {k: jnp.float32(0.0) for k in _BBF_METRIC_NAMES}
            metrics_count = jnp.int32(0)
            carry = carry._replace(
                completed_returns_sum=jnp.float32(0.0),
                completed_returns_count=jnp.int32(0),
            )

    # Pad if fewer evals than expected
    while len(all_metrics_list) < num_evals:
        all_metrics_list.append(
            all_metrics_list[-1]
            if all_metrics_list
            else {
                "eval_lengths": jnp.zeros(128, dtype=jnp.float32),
                "eval_returns": jnp.zeros(128, dtype=jnp.float32),
                "train_return": jnp.float32(0.0),
                "global_step": jnp.int32(0),
                **{k: jnp.float32(0.0) for k in _BBF_METRIC_NAMES},
            }
        )

    all_metrics = jax.tree.map(lambda *args: jnp.stack(args), *all_metrics_list)

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "q_network": network,
        "q_optimizer": optimizer,
        "q_target": target,
        "global_step": carry.global_step,
    }
    return state, all_metrics
