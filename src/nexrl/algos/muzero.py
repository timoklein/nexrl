"""MuZero — MCTS with a learned world model.

"Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
(Schrittwieser et al., 2020)

Replaces AlphaZero's perfect environment model with learned representation,
dynamics, and prediction networks. MCTS simulations happen entirely in the
learned latent space.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    S: latent state dimension (latent_dim)
    A: action dimension (num_actions)
    K: unroll steps
    N: number of atoms (value or reward)
"""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import gymnax
import jax
import mctx
import numpy as np
import optax
import rlax
from flax import nnx
from jax import numpy as jnp

from nexrl.algos.utils import (
    RewardRMSState,
    RMSState,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    normalize_obs,
    resolve_network_kwargs,
    update_rew_rms,
    update_rms,
)
from nexrl.buffers import CircularBuffer
from nexrl.configs import MuZeroConfig
from nexrl.evaluate import evaluate
from nexrl.networks import MuZeroModel
from nexrl.networks.world_model import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_MZ_METRIC_NAMES = ("policy_loss", "value_loss", "reward_loss", "entropy")


class MZTransition(NamedTuple):
    """Transition stored in CircularBuffer for MuZero."""

    obs: chex.Array  # (D,) raw observation
    action: chex.Array  # () discrete action taken
    reward: chex.Array  # () reward received
    done: chex.Array  # () episode boundary flag
    search_policy: chex.Array  # (A,) MCTS visit distribution
    search_value: chex.Array  # () MCTS root value (scalar, pre-transform)


class MZCarry(NamedTuple):
    """Non-module carry state for MuZero training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    buffer: CircularBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def scale_gradient(x: jax.Array, scale: float) -> jax.Array:
    """Scale gradient on backward pass without affecting forward pass."""
    return x * scale + jax.lax.stop_gradient(x) * (1 - scale)


def scalar_to_categorical(scalar: jax.Array, min_val: float, max_val: float, num_atoms: int) -> jax.Array:
    """Convert scalar values to categorical 2-hot targets via signed hyperbolic transform.

    Pipeline: scalar -> signed_hyperbolic -> transform_to_2hot -> categorical target
    """
    transformed = rlax.signed_hyperbolic(scalar)
    return rlax.transform_to_2hot(transformed, min_val, max_val, num_atoms)


def categorical_to_scalar(probs: jax.Array, min_val: float, max_val: float, num_atoms: int) -> jax.Array:
    """Convert categorical probabilities back to scalar values.

    Pipeline: probs -> transform_from_2hot -> signed_parabolic -> scalar

    Input probs shape (*, N) -> output shape (*) stripping the atoms dimension.
    """
    expected = rlax.transform_from_2hot(probs, min_val, max_val, num_atoms)
    result = rlax.signed_parabolic(expected)
    return result.reshape(probs.shape[:-1])


def compute_n_step_targets(
    rewards: jax.Array,
    dones: jax.Array,
    search_values: jax.Array,
    gamma: float,
    n_steps: int,
) -> jax.Array:
    """Compute n-step bootstrapped value targets from sequence data.

    target[t] = sum(gamma^i * r[t+i] * mask[t+i], i=0..n-1) + gamma^n * v[t+n] * mask_prod[t+n]

    Args:
        rewards: shape (B, L)
        dones: shape (B, L)
        search_values: shape (B, L) — MCTS root values
        gamma: discount factor
        n_steps: number of steps to bootstrap over

    Returns:
        targets: shape (B, L) — n-step value targets
    """
    B, L = rewards.shape
    masks = 1.0 - dones.astype(jnp.float32)

    def compute_target_t(t):
        # Accumulate discounted rewards
        def scan_body(carry, i):
            acc, mask_prod = carry
            idx = t + i
            # Clamp index to valid range
            safe_idx = jnp.minimum(idx, L - 1)
            r = rewards[:, safe_idx]
            m = masks[:, safe_idx]
            valid = (idx < L).astype(jnp.float32)
            acc = acc + (gamma**i) * r * mask_prod * valid
            mask_prod = mask_prod * m * valid + (1 - valid) * mask_prod
            return (acc, mask_prod), None

        (acc, mask_prod), _ = jax.lax.scan(
            scan_body,
            (jnp.zeros(B), jnp.ones(B)),
            jnp.arange(n_steps),
        )

        # Bootstrap value
        boot_idx = jnp.minimum(t + n_steps, L - 1)
        boot_valid = (t + n_steps < L).astype(jnp.float32)
        bootstrap = (gamma**n_steps) * search_values[:, boot_idx] * mask_prod * boot_valid
        return acc + bootstrap

    targets = jax.vmap(compute_target_t)(jnp.arange(L))  # (L, B)
    return targets.T  # (B, L)


def _create_networks(
    config: MuZeroConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[MuZeroModel, nnx.Optimizer]:
    """Create MuZeroModel (repr + dynamics + prediction) and single optimizer."""
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    rngs = nnx.Rngs(rng)

    repr_net = RepresentationNetwork(
        in_features=in_features,
        latent_dim=config.latent_dim,
        rngs=rngs,
        **agent_kwargs,
    )
    dynamics_net = DynamicsNetwork(
        latent_dim=config.latent_dim,
        num_actions=action_dim,
        reward_num_atoms=config.reward_num_atoms,
        rngs=rngs,
        **agent_kwargs,
    )
    prediction_net = PredictionNetwork(
        latent_dim=config.latent_dim,
        num_actions=action_dim,
        value_num_atoms=config.value_num_atoms,
        rngs=rngs,
        **agent_kwargs,
    )

    model = MuZeroModel(repr_net, dynamics_net, prediction_net)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    return model, optimizer


def train_muzero(
    config: MuZeroConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a MuZero agent.

    MuZero with MCTS and a learned dynamics model.

    Args:
        config: MuZeroConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, MuZeroConfig

        config = MuZeroConfig.from_yaml("configs/muzero/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("muzero"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    num_actions = action_space.n
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]
    assert discrete, "MuZero requires discrete action space"

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks
    rngs = nnx.Rngs(rng)
    model, optimizer = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init circular buffer for MZTransitions
    seq_len = config.unroll_steps + config.n_steps + 1
    dummy_transition = MZTransition(
        obs=jnp.empty((config.buffer_size, *obs_space.shape), dtype=obs_space.dtype),
        action=jnp.empty(config.buffer_size, dtype=jnp.int32),
        reward=jnp.empty(config.buffer_size, dtype=jnp.float32),
        done=jnp.empty(config.buffer_size, dtype=bool),
        search_policy=jnp.empty((config.buffer_size, num_actions), dtype=jnp.float32),
        search_value=jnp.empty(config.buffer_size, dtype=jnp.float32),
    )
    buf = CircularBuffer.empty(config.buffer_size, dummy_transition)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _MZ_METRIC_NAMES}

    carry = MZCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    # --- MCTS with learned model ---
    if config.mcts_policy == "gumbel":
        search_fn = functools.partial(
            mctx.gumbel_muzero_policy,
            num_simulations=config.num_simulations,
            max_depth=config.max_depth,
        )
    else:
        search_fn = functools.partial(
            mctx.muzero_policy,
            num_simulations=config.num_simulations,
            max_depth=config.max_depth,
        )

    # --- Collect transitions ---
    def collect_transitions(carry, model_, rngs_):
        rng_search = rngs_()

        def recurrent_fn(params, rng_key, action_B, embedding):
            """MCTS recurrent function — uses learned dynamics model."""
            latent_BS = embedding
            next_latent_BS, reward_logits_BN = model_.dynamics(latent_BS, action_B)
            policy_logits_BA, value_logits_BN = model_.prediction(next_latent_BS)

            reward_probs_BN = jax.nn.softmax(reward_logits_BN)
            reward_B = categorical_to_scalar(reward_probs_BN, config.reward_min, config.reward_max, config.reward_num_atoms)

            value_probs_BN = jax.nn.softmax(value_logits_BN)
            value_B = categorical_to_scalar(value_probs_BN, config.value_min, config.value_max, config.value_num_atoms)

            discount_B = jnp.ones_like(reward_B) * config.gamma

            recurrent_output = mctx.RecurrentFnOutput(
                reward=reward_B,  # pyright: ignore[reportCallIssue]
                discount=discount_B,  # pyright: ignore[reportCallIssue]
                prior_logits=policy_logits_BA,  # pyright: ignore[reportCallIssue]
                value=value_B,  # pyright: ignore[reportCallIssue]
            )
            return recurrent_output, next_latent_BS

        obs_for_net = carry.last_obs
        if config.normalize_observations:
            obs_for_net = normalize_obs(carry.obs_rms, obs_for_net)

        latent_BS = model_.representation(obs_for_net)
        policy_logits_BA, value_logits_BN = model_.prediction(latent_BS)

        value_probs_BN = jax.nn.softmax(value_logits_BN)
        value_B = categorical_to_scalar(value_probs_BN, config.value_min, config.value_max, config.value_num_atoms)

        root = mctx.RootFnOutput(
            prior_logits=policy_logits_BA,  # pyright: ignore[reportCallIssue]
            value=value_B,  # pyright: ignore[reportCallIssue]
            embedding=latent_BS,  # pyright: ignore[reportCallIssue]
        )

        policy_output = search_fn(
            params={},
            rng_key=rng_search,
            root=root,
            recurrent_fn=recurrent_fn,
        )

        search_policy_BA = policy_output.action_weights
        search_value_B = policy_output.search_tree.node_values[:, 0]  # pyright: ignore[reportIndexIssue]
        action_B = policy_output.action

        # Step environment with MCTS-selected action
        rng_steps = jax.random.split(rngs_(), num_envs)
        next_obs, new_env_state, rewards, dones, _ = vmap_step(rng_steps, carry.env_state, action_B, env_params)

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

        transition = MZTransition(
            obs=carry.last_obs,
            action=action_B,
            reward=rewards.astype(jnp.float32),
            done=dones,
            search_policy=search_policy_BA,
            search_value=search_value_B,
        )
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
        return carry, transition

    # --- Update networks ---
    def update_networks(sequences, model_, optimizer_, obs_rms):
        """Update MuZero model with sampled sequences.

        sequences: pytree with shape (B, seq_len, ...) where seq_len = unroll_steps + n_steps + 1
        """
        K = config.unroll_steps

        # Extract obs, actions, rewards, dones, search_policy, search_value
        obs_BL = sequences.obs  # (B, L, D)
        action_BL = sequences.action  # (B, L)
        reward_BL = sequences.reward  # (B, L)
        done_BL = sequences.done  # (B, L)
        search_policy_BLA = sequences.search_policy  # (B, L, A)
        search_value_BL = sequences.search_value  # (B, L)

        # Compute n-step value targets for positions 0..K
        value_targets_BL = compute_n_step_targets(reward_BL, done_BL, search_value_BL, config.gamma, config.n_steps)

        # Normalize root observations
        root_obs_BD = obs_BL[:, 0]
        if config.normalize_observations:
            root_obs_BD = normalize_obs(obs_rms, root_obs_BD)

        # Convert value and reward targets to categorical
        # Value targets for steps 0..K
        value_target_cats = jax.vmap(
            lambda v: scalar_to_categorical(v, config.value_min, config.value_max, config.value_num_atoms)
        )(value_targets_BL[:, : K + 1].reshape(-1))
        value_target_cats = value_target_cats.reshape(obs_BL.shape[0], K + 1, config.value_num_atoms)

        # Reward targets for steps 0..K-1
        reward_target_cats = jax.vmap(
            lambda r: scalar_to_categorical(r, config.reward_min, config.reward_max, config.reward_num_atoms)
        )(reward_BL[:, :K].reshape(-1))
        reward_target_cats = reward_target_cats.reshape(obs_BL.shape[0], K, config.reward_num_atoms)

        # Done mask: mask losses after episode boundaries
        # mask[t] = product of (1-done) for steps 0..t-1 (mask[0]=1 always)
        mask_BK1 = jnp.ones((obs_BL.shape[0], K + 1))
        running_mask = jnp.ones(obs_BL.shape[0])
        masks = [running_mask]
        for t in range(K):
            running_mask = running_mask * (1 - done_BL[:, t].astype(jnp.float32))
            masks.append(running_mask)
        mask_BK1 = jnp.stack(masks, axis=1)  # (B, K+1)

        def loss_fn(model_inner):
            # Step 0: representation -> prediction
            latent_BS = model_inner.representation(root_obs_BD)
            policy_logits_BA, value_logits_BN = model_inner.prediction(latent_BS)

            # Policy loss: KL divergence (cross-entropy - target entropy)
            log_probs_BA = jax.nn.log_softmax(policy_logits_BA)
            target_policy_BA = search_policy_BLA[:, 0]
            policy_loss_B = -jnp.sum(target_policy_BA * log_probs_BA, axis=-1)

            # Value loss: softmax cross-entropy
            value_log_probs_BN = jax.nn.log_softmax(value_logits_BN)
            value_loss_B = -jnp.sum(value_target_cats[:, 0] * value_log_probs_BN, axis=-1)

            # Entropy bonus
            probs_BA = jax.nn.softmax(policy_logits_BA)
            entropy_B = -jnp.sum(probs_BA * log_probs_BA, axis=-1)

            total_policy_loss = (policy_loss_B * mask_BK1[:, 0]).mean()
            total_value_loss = (value_loss_B * mask_BK1[:, 0]).mean()
            total_reward_loss = jnp.float32(0.0)
            total_entropy = entropy_B.mean()

            # Steps 1..K: dynamics unrolling (read-only scan over model)
            def unroll_step(carry_inner, t):
                lat, pol_loss, val_loss, rew_loss, ent = carry_inner

                lat = scale_gradient(lat, config.gradient_scale)
                action_t_B = action_BL[:, t]
                next_lat, reward_logits_BN = model_inner.dynamics(lat, action_t_B)
                pol_logits_BA, val_logits_BN = model_inner.prediction(next_lat)

                # Reward loss
                rew_log_probs_BN = jax.nn.log_softmax(reward_logits_BN)
                rew_loss_B = -jnp.sum(reward_target_cats[:, t] * rew_log_probs_BN, axis=-1)

                # Policy loss
                log_p_BA = jax.nn.log_softmax(pol_logits_BA)
                target_p_BA = search_policy_BLA[:, t + 1]
                pol_loss_B = -jnp.sum(target_p_BA * log_p_BA, axis=-1)

                # Value loss
                val_log_p_BN = jax.nn.log_softmax(val_logits_BN)
                val_loss_B = -jnp.sum(value_target_cats[:, t + 1] * val_log_p_BN, axis=-1)

                # Entropy
                p_BA = jax.nn.softmax(pol_logits_BA)
                ent_B = -jnp.sum(p_BA * log_p_BA, axis=-1)

                step_mask = mask_BK1[:, t + 1]
                pol_loss = pol_loss + (pol_loss_B * step_mask).mean()
                val_loss = val_loss + (val_loss_B * step_mask).mean()
                rew_loss = rew_loss + (rew_loss_B * step_mask).mean()
                ent = ent + (ent_B * step_mask).mean()

                return (next_lat, pol_loss, val_loss, rew_loss, ent), None

            (_, total_policy_loss, total_value_loss, total_reward_loss, total_entropy), _ = jax.lax.scan(
                unroll_step,
                (latent_BS, total_policy_loss, total_value_loss, total_reward_loss, total_entropy),
                jnp.arange(K),
            )

            # Average over unroll steps
            total_policy_loss = total_policy_loss / (K + 1)
            total_value_loss = total_value_loss / (K + 1)
            total_reward_loss = total_reward_loss / jnp.maximum(K, 1)
            total_entropy = total_entropy / (K + 1)

            loss = (
                total_policy_loss
                + config.vf_coef * total_value_loss
                + config.reward_coef * total_reward_loss
                - config.ent_coef * total_entropy
            )
            return loss, (total_policy_loss, total_value_loss, total_reward_loss, total_entropy)

        (_, (policy_loss, value_loss, reward_loss, entropy)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model_)
        optimizer_.update(model_, grads)

        return policy_loss, value_loss, reward_loss, entropy

    # --- Train iteration ---
    def train_iteration(carry, model_, optimizer_, rngs_):
        start_training = carry.global_step > config.fill_buffer

        carry, batch = collect_transitions(carry, model_, rngs_)
        carry = carry._replace(buffer=carry.buffer.extend(batch))

        zero_update_metrics = {k: jnp.float32(0.0) for k in _MZ_METRIC_NAMES}

        def sample_strided_sequences(buffer, rng_sample):
            """Sample temporal sequences accounting for interleaved multi-env storage.

            Buffer layout: [env0_t0, env1_t0, ..., envN_t0, env0_t1, env1_t1, ...]
            To get seq_len consecutive timesteps from the same env, stride by num_envs.
            """
            n = buffer.num_entries
            num_timesteps = n // num_envs
            max_start_t = num_timesteps - seq_len
            rng_t, rng_e = jax.random.split(rng_sample)
            start_timesteps = jax.random.randint(rng_t, (config.batch_size,), 0, max_start_t)
            env_ids = jax.random.randint(rng_e, (config.batch_size,), 0, num_envs)
            offsets = jnp.arange(seq_len)
            # indices[i, j] = env_ids[i] + (start_timesteps[i] + j) * num_envs
            indices = env_ids[:, None] + (start_timesteps[:, None] + offsets[None, :]) * num_envs
            indices = indices % buffer.size
            return jax.tree.map(lambda arr: arr[indices], buffer.data)

        def update_iteration(_, state):
            carry_inner, model_inner, optimizer_inner, rngs_inner, metrics_sum = state
            rng_sample = rngs_inner()
            seqs = sample_strided_sequences(carry_inner.buffer, rng_sample)
            p_loss, v_loss, r_loss, ent = update_networks(seqs, model_inner, optimizer_inner, carry_inner.obs_rms)
            metrics_sum = jax.tree.map(
                lambda s, m: s + m,
                metrics_sum,
                {"policy_loss": p_loss, "value_loss": v_loss, "reward_loss": r_loss, "entropy": ent},
            )
            return (carry_inner, model_inner, optimizer_inner, rngs_inner, metrics_sum)

        def do_updates(carry_u, model_u, optimizer_u, rngs_u):
            carry_u, model_u, optimizer_u, rngs_u, metrics_sum = nnx.fori_loop(
                0,
                config.num_epochs,
                update_iteration,
                (carry_u, model_u, optimizer_u, rngs_u, zero_update_metrics),
            )
            iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry_u, model_u, optimizer_u, rngs_u, iter_metrics

        def no_updates(carry_u, model_u, optimizer_u, rngs_u):
            return carry_u, model_u, optimizer_u, rngs_u, zero_update_metrics

        carry, model_, optimizer_, rngs_, update_metrics = nnx.cond(
            start_training, do_updates, no_updates, carry, model_, optimizer_, rngs_
        )

        # Accumulate metrics (only when training)
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, update_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, model_, optimizer_, rngs_

    # --- Eval callback using MCTS with learned model ---
    def muzero_eval_act(obs_BD, rng, model_):
        """Eval policy: run full MCTS with learned model.

        Note: obs normalization is handled by make_eval_act wrapper.
        """
        latent_BS = model_.representation(obs_BD)
        policy_logits_BA, value_logits_BN = model_.prediction(latent_BS)

        value_probs_BN = jax.nn.softmax(value_logits_BN)
        value_B = categorical_to_scalar(value_probs_BN, config.value_min, config.value_max, config.value_num_atoms)

        def recurrent_fn(params, rng_key, action_B, embedding):
            lat = embedding
            next_lat, reward_logits = model_.dynamics(lat, action_B)
            pol_logits, val_logits = model_.prediction(next_lat)
            reward_probs = jax.nn.softmax(reward_logits)
            reward = categorical_to_scalar(reward_probs, config.reward_min, config.reward_max, config.reward_num_atoms)
            val_probs = jax.nn.softmax(val_logits)
            val = categorical_to_scalar(val_probs, config.value_min, config.value_max, config.value_num_atoms)
            discount = jnp.ones_like(reward) * config.gamma
            return mctx.RecurrentFnOutput(
                reward=reward,  # pyright: ignore[reportCallIssue]
                discount=discount,  # pyright: ignore[reportCallIssue]
                prior_logits=pol_logits,  # pyright: ignore[reportCallIssue]
                value=val,  # pyright: ignore[reportCallIssue]
            ), next_lat

        root = mctx.RootFnOutput(
            prior_logits=policy_logits_BA,  # pyright: ignore[reportCallIssue]
            value=value_B,  # pyright: ignore[reportCallIssue]
            embedding=latent_BS,  # pyright: ignore[reportCallIssue]
        )
        policy_output = search_fn(params={}, rng_key=rng, root=root, recurrent_fn=recurrent_fn)
        return policy_output.action

    def eval_callback(carry, model_, rngs_):
        act_fn = make_eval_act(
            lambda obs, rng: muzero_eval_act(obs, rng, model_),
            config,
            carry.obs_rms,
        )
        return evaluate(act_fn, rngs_(), env, env_params, 128, max_steps)

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
            **{k: jnp.float32(0.0) for k in _MZ_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, model_, optimizer_, rngs_ = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, model_, optimizer_, rngs_ = nnx.fori_loop(0, steps_per_iter, train_body, (carry, model_, optimizer_, rngs_))

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

        return (carry, model_, optimizer_, rngs_), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, model, optimizer, rngs), all_metrics = scan_eval((carry, model, optimizer, rngs), jnp.zeros(num_evals))

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "model": model,
        "optimizer": optimizer,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
