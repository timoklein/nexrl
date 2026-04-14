"""Proximal Q-Network (PQN) — standalone training function.

"Simplifying Deep Temporal Difference Learning" (Gallici et al., 2024)

Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
by Matteo Gallici et al.

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    A: action dimension (num_actions)
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
from optax import linear_schedule

from nexrl.algos.utils import (
    RewardRMSState,
    RMSState,
    create_encoder,
    create_env,
    create_vec_env_fns,
    make_eval_act,
    resolve_network_kwargs,
    shuffle_and_split,
    update_and_normalize_obs,
    update_and_normalize_rew,
)
from nexrl.configs import PQNConfig
from nexrl.evaluate import evaluate
from nexrl.networks import DiscreteQNetwork, NormalizedDiscreteQNetwork
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


class Trajectory(struct.PyTreeNode):
    """Container for trajectory data collected during rollout."""

    obs: chex.Array
    action: chex.Array
    next_q: chex.Array
    reward: chex.Array
    done: chex.Array


class TargetMinibatch(struct.PyTreeNode):
    """Container for minibatch data with computed targets."""

    trajectories: Trajectory
    targets: chex.Array


_PQN_METRIC_NAMES = ("td_loss", "mean_q_value", "epsilon")


class PQNCarry(NamedTuple):
    """Non-module carry state for PQN training."""

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
    config: PQNConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[DiscreteQNetwork | NormalizedDiscreteQNetwork, nnx.Optimizer]:
    """Create Q-network and optimizer."""
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None), default_activation="relu")

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    q_kwargs = {"in_features": in_features, "action_dim": action_dim, **agent_kwargs}

    rngs = nnx.Rngs(rng)
    if config.norm_type == "layer_norm":
        q_network = NormalizedDiscreteQNetwork(**q_kwargs, norm_input=config.norm_input, rngs=rngs)
    else:
        encoder = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
        q_network = DiscreteQNetwork(**q_kwargs, encoder=encoder, rngs=rngs)

    tx = optax.chain(
        optax.clip(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    q_optimizer = nnx.Optimizer(q_network, tx, wrt=nnx.Param)
    return q_network, q_optimizer


def train_pqn(
    config: PQNConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a PQN agent.

    Proximal Q-Network combining on-policy updates with Q-learning.

    Args:
        config: PQNConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, PQNConfig

        config = PQNConfig.from_yaml("configs/pqn/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("pqn"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    # --- Setup ---
    env, env_params = create_env(config, env, env_params)

    eps_schedule = linear_schedule(
        config.eps_start,
        config.eps_end,
        int(config.exploration_fraction * config.total_timesteps),
    )

    num_envs = config.num_envs
    num_steps = config.num_steps
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)  # pyright: ignore[reportAttributeAccessIssue]
    assert discrete, "PQN requires discrete action space"

    iteration_size = num_envs * num_steps
    assert iteration_size % config.num_minibatches == 0

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    q_network, q_optimizer = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _PQN_METRIC_NAMES}

    carry = PQNCarry(
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
    def collect_trajectories(carry, q_network, rngs):
        """Roll out epsilon-greedy policy for num_steps. Q-network is read-only.
        Pre-split keys to keep jax.lax.scan (no module mutation needed)."""
        epsilon = eps_schedule(carry.global_step)
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_action, rng_step = jax.random.split(step_key)

            # Sample action using epsilon-greedy policy
            action_B = q_network.act(carry.last_obs, rng_action, epsilon=epsilon)

            # Step environment
            rng_steps = jax.random.split(rng_step, num_envs)
            next_obs, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_B, env_params)

            # Track episode returns using raw rewards (before normalization)
            new_running_B = carry.running_return + reward_B
            completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * done_B)
            completed_count = carry.completed_returns_count + jnp.sum(done_B.astype(jnp.int32))
            reset_running_B = new_running_B * (1 - done_B)

            # Compute Q-values for next state
            next_q_BA = q_network(next_obs)

            obs_rms = carry.obs_rms
            rew_rms = carry.rew_rms
            if config.normalize_observations:
                obs_rms, next_obs = update_and_normalize_obs(obs_rms, next_obs)
            if config.normalize_rewards:
                rew_rms, reward_B = update_and_normalize_rew(rew_rms, reward_B, done_B, config.gamma)

            transition = Trajectory(carry.last_obs, action_B, next_q_BA, reward_B, done_B)
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs,
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

    # --- Calculate TD-lambda targets ---
    def calculate_targets(trajectories, max_last_q_B):
        def get_target(lambda_return_and_next_q, t):
            lambda_return_B, next_q_B = lambda_return_and_next_q
            return_bootstrap_B = next_q_B + config.td_lambda * (lambda_return_B - next_q_B)
            lambda_return_B = t.reward + (1 - t.done) * config.gamma * return_bootstrap_B
            max_next_q_B = t.next_q.max(axis=1)
            return (lambda_return_B, max_next_q_B), lambda_return_B

        max_last_q_B = jnp.where(trajectories.done[-1], 0, max_last_q_B)
        lambda_returns_B = trajectories.reward[-1] + config.gamma * max_last_q_B  # type: ignore[operator]
        _, targets_TB = jax.lax.scan(
            get_target,
            (lambda_returns_B, max_last_q_B),
            jax.tree.map(lambda x: x[:-1], trajectories),
            reverse=True,
        )
        targets_TB = jnp.concatenate((targets_TB, lambda_returns_B[None]))
        return targets_TB

    # --- Update Q-network ---
    def update_q(mb, q_network, q_optimizer):
        tr, ta = mb.trajectories, mb.targets

        def loss_fn(model):
            q_B = model.take(tr.obs, tr.action)
            return optax.l2_loss(q_B, ta).mean(), q_B.mean()

        (loss, mean_q), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_network)
        q_optimizer.update(q_network, grads)
        return loss, mean_q

    # --- Train iteration ---
    def train_iteration(carry, q_network, q_optimizer, rngs):
        carry, trajectories = collect_trajectories(carry, q_network, rngs)

        last_q_BA = q_network(carry.last_obs)
        max_last_q_B = last_q_BA.max(axis=1)
        max_last_q_B = jnp.where(carry.last_done, 0, max_last_q_B)
        targets = calculate_targets(trajectories, max_last_q_B)

        epsilon = eps_schedule(carry.global_step)
        zero_update_metrics = {"td_loss": jnp.float32(0.0), "mean_q_value": jnp.float32(0.0)}

        def update_epoch(_, state):
            q_net, q_opt, rngs, metrics_sum = state

            batch = TargetMinibatch(trajectories, targets)
            minibatches = shuffle_and_split(batch, rngs(), iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def update_step(state, mb):
                q_net, q_opt = state
                td_loss, mean_q = update_q(mb, q_net, q_opt)
                return (q_net, q_opt), {"td_loss": td_loss, "mean_q_value": mean_q}

            (q_net, q_opt), mb_metrics = update_step((q_net, q_opt), minibatches)
            epoch_metrics = jax.tree.map(jnp.mean, mb_metrics)
            metrics_sum = jax.tree.map(lambda s, e: s + e, metrics_sum, epoch_metrics)
            return (q_net, q_opt, rngs, metrics_sum)

        q_network, q_optimizer, rngs, epoch_metrics_sum = nnx.fori_loop(
            0, config.num_epochs, update_epoch, (q_network, q_optimizer, rngs, zero_update_metrics)
        )

        # Average over epochs, add epsilon
        iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, epoch_metrics_sum)
        iter_metrics["epsilon"] = epsilon

        # Accumulate in carry
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, iter_metrics),
            metrics_count=carry.metrics_count + 1,
        )

        return carry, q_network, q_optimizer, rngs

    # --- Eval callback ---
    def eval_callback(carry, q_network, rngs):
        act_fn = make_eval_act(lambda obs, rng: q_network.act(obs, rng, epsilon=0.005), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if config.run_evaluation and not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, q_network, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "train_return": jnp.float32(0.0),
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _PQN_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, q_net, q_opt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, q_net, q_opt, rngs = nnx.fori_loop(0, num_iters_per_eval, train_body, (carry, q_net, q_opt, rngs))

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

        return (carry, q_net, q_opt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, q_network, q_optimizer, rngs), all_metrics = scan_eval((carry, q_network, q_optimizer, rngs), jnp.zeros(num_evals))

    if config.run_evaluation and not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics
