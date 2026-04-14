"""AlphaZero — MCTS-based policy improvement with a perfect environment model.

"A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go
through Self-Play" (Silver et al., 2018)

Uses Monte Carlo Tree Search (via mctx) to generate expert policies,
then distills them into neural networks (Expert Iteration).

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    A: action dimension (num_actions)
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
from nexrl.configs import AlphaZeroConfig
from nexrl.evaluate import evaluate
from nexrl.networks import DiscretePolicy, VNetwork
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_AZ_METRIC_NAMES = ("actor_loss", "value_loss", "entropy")


class AZTransition(NamedTuple):
    """Transition stored in CircularBuffer for AlphaZero."""

    obs: chex.Array  # (D,) raw obs, normalized at train time
    search_policy: chex.Array  # (A,) MCTS visit distribution
    search_value: chex.Array  # scalar, MCTS root value


class AZCarry(NamedTuple):
    """Non-module carry state for AlphaZero training."""

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


def _create_networks(
    config: AlphaZeroConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[DiscretePolicy, nnx.Optimizer, VNetwork, nnx.Optimizer]:
    """Create actor (DiscretePolicy), critic (VNetwork), and their optimizers."""
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    rngs = nnx.Rngs(rng)
    actor = DiscretePolicy(in_features=in_features, action_dim=action_dim, rngs=rngs, **agent_kwargs)
    critic = VNetwork(in_features=in_features, rngs=rngs, **agent_kwargs)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    actor_opt = nnx.Optimizer(actor, tx, wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, tx, wrt=nnx.Param)

    return actor, actor_opt, critic, critic_opt


def train_alphazero(
    config: AlphaZeroConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train an AlphaZero agent.

    AlphaZero with MCTS-based policy improvement using a perfect environment model.

    Args:
        config: AlphaZeroConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, AlphaZeroConfig

        config = AlphaZeroConfig.from_yaml("configs/alphazero/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("alphazero"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    # --- Setup (traced once) ---
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    num_actions = action_space.n
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]
    assert discrete, "AlphaZero requires discrete action space"

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    actor, actor_opt, critic, critic_opt = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init circular buffer for AZTransitions
    dummy_transition = AZTransition(
        obs=jnp.empty((config.buffer_size, *obs_space.shape), dtype=obs_space.dtype),
        search_policy=jnp.empty((config.buffer_size, num_actions), dtype=jnp.float32),
        search_value=jnp.empty(config.buffer_size, dtype=jnp.float32),
    )
    buf = CircularBuffer.empty(config.buffer_size, dummy_transition)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _AZ_METRIC_NAMES}

    carry = AZCarry(
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

    # --- MCTS ---
    _single_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Select MCTS policy
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
    def collect_transitions(carry, actor_, critic_, rngs):
        rng_search = rngs()

        # recurrent_fn defined here to capture current actor_/critic_/obs_rms
        def recurrent_fn(params, rng_key, action_B, embedding):
            """MCTS recurrent function — steps env and evaluates next state."""
            cur_env_state = embedding
            rng_steps = jax.random.split(rng_key, num_envs)
            next_obs_BD, next_env_state, reward_B, done_B, _ = _single_step(rng_steps, cur_env_state, action_B, env_params)

            obs_for_net = next_obs_BD
            if config.normalize_observations:
                obs_for_net = normalize_obs(carry.obs_rms, obs_for_net)

            prior_logits_BA = actor_.get_logits(obs_for_net)
            value_B = critic_(obs_for_net)

            discount_B = (1 - done_B.astype(jnp.float32)) * config.gamma
            value_B = jnp.where(done_B, 0.0, value_B)

            recurrent_output = mctx.RecurrentFnOutput(
                reward=reward_B.astype(jnp.float32),  # pyright: ignore[reportCallIssue]
                discount=discount_B,  # pyright: ignore[reportCallIssue]
                prior_logits=prior_logits_BA,  # pyright: ignore[reportCallIssue]
                value=value_B,  # pyright: ignore[reportCallIssue]
            )
            return recurrent_output, next_env_state

        obs_for_net = carry.last_obs
        if config.normalize_observations:
            obs_for_net = normalize_obs(carry.obs_rms, obs_for_net)

        prior_logits_BA = actor_.get_logits(obs_for_net)
        value_B = critic_(obs_for_net)

        root = mctx.RootFnOutput(
            prior_logits=prior_logits_BA,  # pyright: ignore[reportCallIssue]
            value=value_B,  # pyright: ignore[reportCallIssue]
            embedding=carry.env_state,  # pyright: ignore[reportCallIssue]
        )

        policy_output = search_fn(
            params={},
            rng_key=rng_search,
            root=root,
            recurrent_fn=recurrent_fn,
        )

        # action_weights_BA is the MCTS visit distribution (sums to 1)
        search_policy_BA = policy_output.action_weights
        # Root value from MCTS tree
        search_value_B = policy_output.search_tree.node_values[:, 0]  # pyright: ignore[reportIndexIssue]
        action_B = policy_output.action

        # Step environment with MCTS-selected action
        rng_steps = jax.random.split(rngs(), num_envs)
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

        transition = AZTransition(
            obs=carry.last_obs,
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
    def update_networks(batch, actor_, actor_opt_, critic_, critic_opt_, obs_rms):
        obs_BD = batch.obs
        if config.normalize_observations:
            obs_BD = normalize_obs(obs_rms, obs_BD)

        search_policy_BA = batch.search_policy
        search_value_B = batch.search_value

        # Actor loss: cross-entropy between MCTS policy and network logits + entropy bonus
        def actor_loss_fn(model):
            logits_BA = model.get_logits(obs_BD)
            log_probs_BA = jax.nn.log_softmax(logits_BA)
            # Cross-entropy: -sum(target * log_softmax(logits))
            ce_loss_B = -jnp.sum(search_policy_BA * log_probs_BA, axis=-1)
            # Entropy bonus: -sum(p * log(p)) where p = softmax(logits)
            probs_BA = jax.nn.softmax(logits_BA)
            entropy_B = -jnp.sum(probs_BA * log_probs_BA, axis=-1)
            loss = (ce_loss_B - config.ent_coef * entropy_B).mean()
            return loss, entropy_B.mean()

        (actor_loss, entropy), actor_grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor_)
        actor_opt_.update(actor_, actor_grads)

        # Critic loss: MSE between network value and MCTS root value
        def critic_loss_fn(model):
            value_B = model(obs_BD)
            return config.vf_coef * jnp.square(value_B - search_value_B).mean()

        value_loss, critic_grads = nnx.value_and_grad(critic_loss_fn)(critic_)
        critic_opt_.update(critic_, critic_grads)

        return actor_loss, value_loss, entropy

    # --- Train iteration ---
    def train_iteration(carry, actor_, actor_opt_, critic_, critic_opt_, rngs):
        start_training = carry.global_step > config.fill_buffer

        carry, batch = collect_transitions(carry, actor_, critic_, rngs)
        carry = carry._replace(buffer=carry.buffer.extend(batch))

        zero_update_metrics = {"actor_loss": jnp.float32(0.0), "value_loss": jnp.float32(0.0), "entropy": jnp.float32(0.0)}

        def update_iteration(_, state):
            carry_inner, actor_inner, actor_opt_inner, critic_inner, critic_opt_inner, rngs_inner, metrics_sum = state
            # Sample from buffer
            rng_sample = rngs_inner()
            n = carry_inner.buffer.num_entries
            idx = jax.random.randint(rng_sample, (config.batch_size,), 0, n)
            minibatch = jax.tree.map(lambda arr: arr[idx], carry_inner.buffer.data)
            a_loss, v_loss, ent = update_networks(
                minibatch, actor_inner, actor_opt_inner, critic_inner, critic_opt_inner, carry_inner.obs_rms
            )
            metrics_sum = jax.tree.map(
                lambda s, m: s + m, metrics_sum, {"actor_loss": a_loss, "value_loss": v_loss, "entropy": ent}
            )
            return (carry_inner, actor_inner, actor_opt_inner, critic_inner, critic_opt_inner, rngs_inner, metrics_sum)

        def do_updates(carry_u, actor_u, actor_opt_u, critic_u, critic_opt_u, rngs_u):
            carry_u, actor_u, actor_opt_u, critic_u, critic_opt_u, rngs_u, metrics_sum = nnx.fori_loop(
                0,
                config.num_epochs,
                update_iteration,
                (carry_u, actor_u, actor_opt_u, critic_u, critic_opt_u, rngs_u, zero_update_metrics),
            )
            iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry_u, actor_u, actor_opt_u, critic_u, critic_opt_u, rngs_u, iter_metrics

        def no_updates(carry_u, actor_u, actor_opt_u, critic_u, critic_opt_u, rngs_u):
            return carry_u, actor_u, actor_opt_u, critic_u, critic_opt_u, rngs_u, zero_update_metrics

        carry, actor_, actor_opt_, critic_, critic_opt_, rngs, update_metrics = nnx.cond(
            start_training, do_updates, no_updates, carry, actor_, actor_opt_, critic_, critic_opt_, rngs
        )

        # Accumulate metrics (only when training)
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, update_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, actor_, actor_opt_, critic_, critic_opt_, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor_, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor_.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, actor, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _AZ_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, actor_, actor_opt_, critic_, critic_opt_, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, actor_, actor_opt_, critic_, critic_opt_, rngs = nnx.fori_loop(
            0, steps_per_iter, train_body, (carry, actor_, actor_opt_, critic_, critic_opt_, rngs)
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        # Mean training episode return
        train_return = carry.completed_returns_sum / jnp.maximum(carry.completed_returns_count, 1)

        if config.run_evaluation:
            eval_lengths, eval_returns = eval_callback(carry, actor_, rngs)
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

        return (carry, actor_, actor_opt_, critic_, critic_opt_, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor, actor_opt, critic, critic_opt, rngs), all_metrics = scan_eval(
        (carry, actor, actor_opt, critic, critic_opt, rngs), jnp.zeros(num_evals)
    )

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "actor": actor,
        "actor_optimizer": actor_opt,
        "critic": critic,
        "critic_optimizer": critic_opt,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
