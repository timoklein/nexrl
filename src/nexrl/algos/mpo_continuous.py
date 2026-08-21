"""Maximum a-Posteriori Policy Optimization (MPO) — continuous action spaces.

Off-policy with replay buffer. Uses a sample-based E-step with importance weights
from Q-values, and a decomposed M-step that separately updates the policy mean
and std with per-dimension KL trust-region constraints.

The temperature dual loss computes the full gradient through logsumexp(Q/eta),
giving the correct equilibrium KL(weights || uniform) = epsilon. This is critical
for continuous MPO where Q-value magnitudes can make the stop_gradient version
converge to useless uniform weights.

Reference: Abdolmaleki et al., "Maximum a Posteriori Policy Optimisation" (2018).

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension
    N: num_action_samples (E-step)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import distrax
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

from nexrl.algos.utils import (
    RewardRMSState,
    RMSState,
    create_encoder,
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
from nexrl.configs import MPOConfig
from nexrl.evaluate import evaluate
from nexrl.networks import GaussianPolicy, QNetwork
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_METRIC_NAMES = ("actor_loss", "critic_loss", "temperature", "alpha_mean", "alpha_stddev", "mean_q_value")


class MPOContinuousCarry(NamedTuple):
    """Non-module carry state for continuous MPO training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    dual_params: dict
    dual_opt_state: Any
    train_metrics: dict
    metrics_count: int
    running_return: chex.Array
    completed_returns_sum: chex.Array
    completed_returns_count: chex.Array


def _create_networks(
    config: MPOConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[GaussianPolicy, nnx.Optimizer, GaussianPolicy, QNetwork, nnx.Optimizer, QNetwork]:
    """Create continuous actor (+ target), critic (+ target), and optimizers."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    actor_kwargs = resolve_network_kwargs(config.actor_kwargs)
    critic_kwargs = resolve_network_kwargs(config.critic_kwargs)

    rngs = nnx.Rngs(rng)

    actor_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    target_actor_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    q_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    qt_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None

    actor = GaussianPolicy(
        in_features=in_features, action_dim=action_dim, action_range=action_range, **actor_kwargs, rngs=rngs, encoder=actor_enc
    )
    target_actor = GaussianPolicy(
        in_features=in_features,
        action_dim=action_dim,
        action_range=action_range,
        **actor_kwargs,
        rngs=rngs,
        encoder=target_actor_enc,
    )
    nnx.update(target_actor, nnx.state(actor, nnx.Param))

    actor_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate=config.learning_rate))
    actor_opt = nnx.Optimizer(actor, actor_tx, wrt=nnx.Param)

    q_network = QNetwork(obs_dim=in_features, action_dim=action_dim, **critic_kwargs, rngs=rngs, encoder=q_enc)
    q_target = QNetwork(obs_dim=in_features, action_dim=action_dim, **critic_kwargs, rngs=rngs, encoder=qt_enc)
    nnx.update(q_target, nnx.state(q_network, nnx.Param))

    q_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate=config.critic_learning_rate))
    q_opt = nnx.Optimizer(q_network, q_tx, wrt=nnx.Param)

    return actor, actor_opt, target_actor, q_network, q_opt, q_target


def train_mpo_continuous(
    config: MPOConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a continuous MPO agent.

    Maximum a-Posteriori Policy Optimization for continuous action spaces.

    Args:
        config: MPOConfig with hyperparameters and environment specification.
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

        from nexrl import get_train_fn, MPOConfig

        config = MPOConfig.from_yaml("configs/mpo_continuous/gymnax/pendulum.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("mpo_continuous"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)
    # Dedicated eval instances — sharing stateful Atari/DMC envs corrupts training (see make_eval_env).
    eval_env = make_eval_env(env, config.run_evaluation)

    num_envs = config.num_envs
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    action_dim = int(np.prod(action_space.shape))
    max_steps = env_params.max_steps_in_episode

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    rngs = nnx.Rngs(rng)
    actor, actor_opt, target_actor, q_network, q_opt, q_target = _create_networks(config, env, env_params, rngs())

    # Dual params: per-dimension alpha for mean and std when per_dim_constraining
    alpha_shape = (action_dim,) if config.per_dim_constraining else ()
    dual_params: dict[str, jax.Array] = {
        "log_temperature": jnp.array(config.init_log_temperature),
        "log_alpha_mean": jnp.full(alpha_shape, config.init_log_alpha_mean),
        "log_alpha_stddev": jnp.full(alpha_shape, config.init_log_alpha_stddev),
    }
    dual_tx = optax.adam(config.dual_learning_rate)
    dual_opt_state = dual_tx.init(dual_params)

    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _METRIC_NAMES}

    carry = MPOContinuousCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        dual_params=dual_params,
        dual_opt_state=dual_opt_state,
        train_metrics=zero_train_metrics,
        metrics_count=0,
        running_return=jnp.zeros(num_envs),
        completed_returns_sum=jnp.float32(0.0),
        completed_returns_count=jnp.int32(0),
    )

    N = config.num_action_samples

    # --- Collect transitions ---
    def collect_transitions(carry, actor, rngs):
        last_obs = carry.last_obs
        if config.normalize_observations:
            last_obs = normalize_obs(carry.obs_rms, last_obs)

        action_BA = actor.act(last_obs, rngs())  # (B, A) — clipped to action_range

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_BA, env_params)

        new_running_B = carry.running_return + reward_B
        completed_sum = carry.completed_returns_sum + jnp.sum(new_running_B * done_B)
        completed_count = carry.completed_returns_count + jnp.sum(done_B.astype(jnp.int32))
        reset_running_B = new_running_B * (1 - done_B)

        obs_rms = carry.obs_rms
        rew_rms = carry.rew_rms
        if config.normalize_observations:
            obs_rms = update_rms(obs_rms, next_obs)
        if config.normalize_rewards:
            rew_rms = update_rew_rms(rew_rms, reward_B, done_B, config.gamma)

        mb = Minibatch(obs=carry.last_obs, action=action_BA, reward=reward_B, next_obs=next_obs, done=done_B)
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

    # --- Critic update ---
    def update_critic(mb, q_network, q_opt, q_target, target_actor, rng):
        # Single-sample target value from target policy
        next_action_BA = target_actor.act(mb.next_obs, rng)  # (B, A)
        v_next_B = q_target(mb.next_obs, next_action_BA)  # (B,)
        target_B = mb.reward + (1 - mb.done) * config.gamma * jax.lax.stop_gradient(v_next_B)

        def critic_loss_fn(q_model):
            q_B = q_model(mb.obs, mb.action)  # (B,)
            return optax.l2_loss(q_B, target_B).mean(), q_B.mean()

        (critic_loss, mean_q), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(q_network)
        q_opt.update(q_network, grads)
        return critic_loss, mean_q

    # --- Actor + dual update ---
    def update_actor_and_dual(carry, mb, actor, actor_opt, q_target, target_actor, rng):
        dual_params = carry.dual_params
        obs_BD = mb.obs  # (B, D)
        B = obs_BD.shape[0]

        # E-step: sample N actions from target policy, compute importance weights
        rngs_N = jax.random.split(rng, N)
        # For each sample key, call target_actor.act on the full batch
        obs_rep_BD = obs_BD  # broadcast below via vmap
        actions_NBA = jax.vmap(lambda k: target_actor.act(obs_rep_BD, k))(rngs_N)  # (N, B, A)

        # Evaluate Q for all (N*B) state-action pairs
        obs_flat_NB_D = jnp.tile(obs_BD[None, :, :], (N, 1, 1)).reshape(N * B, -1)  # (N*B, D)
        act_flat_NB_A = actions_NBA.reshape(N * B, -1)  # (N*B, A)
        q_NB = jax.lax.stop_gradient(q_target(obs_flat_NB_D, act_flat_NB_A)).reshape(N, B)  # (N, B)

        temperature = jax.nn.softplus(dual_params["log_temperature"]) + 1e-8
        weights_NB = jax.nn.softmax(q_NB / temperature, axis=0)  # (N, B) — non-parametric improved policy

        # M-step: decomposed policy update (separate mean and std losses)
        # Get target and online mean/std
        target_mean_BA, target_std_BA = target_actor.get_mean_std(obs_BD)  # (B, A), (B, A)

        def actor_loss_fn(actor_model):
            online_mean_BA, online_std_BA = actor_model.get_mean_std(obs_BD)  # (B, A), (B, A)

            # Log-probs under fixed-std distribution (only mean varies)
            fixed_std_dist = distrax.MultivariateNormalDiag(loc=online_mean_BA, scale_diag=target_std_BA)
            # actions_NBA: (N, B, A), need log_prob for each -> (N, B)
            log_prob_mean_NB = jax.vmap(lambda a: fixed_std_dist.log_prob(a))(actions_NBA)  # (N, B)

            # Log-probs under fixed-mean distribution (only std varies)
            fixed_mean_dist = distrax.MultivariateNormalDiag(loc=target_mean_BA, scale_diag=online_std_BA)
            log_prob_std_NB = jax.vmap(lambda a: fixed_mean_dist.log_prob(a))(actions_NBA)  # (N, B)

            # Weighted negative log-likelihood losses
            sg_weights_NB = jax.lax.stop_gradient(weights_NB)
            loss_policy_mean = -jnp.sum(sg_weights_NB * log_prob_mean_NB, axis=0).mean()
            loss_policy_std = -jnp.sum(sg_weights_NB * log_prob_std_NB, axis=0).mean()

            # Per-dimension KL penalties
            # KL(target || online) for mean: use target_std for both
            kl_mean_BA = 0.5 * (target_mean_BA - online_mean_BA) ** 2 / (target_std_BA**2 + 1e-8)

            # KL(target || online) for std: use target_mean for both
            kl_std_BA = jnp.log(online_std_BA / target_std_BA) + (target_std_BA**2) / (2 * online_std_BA**2 + 1e-8) - 0.5

            alpha_mean = jax.nn.softplus(dual_params["log_alpha_mean"]) + 1e-8
            alpha_stddev = jax.nn.softplus(dual_params["log_alpha_stddev"]) + 1e-8

            loss_kl_mean = jnp.sum(jax.lax.stop_gradient(alpha_mean) * jnp.mean(kl_mean_BA, axis=0))
            loss_kl_std = jnp.sum(jax.lax.stop_gradient(alpha_stddev) * jnp.mean(kl_std_BA, axis=0))

            total_loss = loss_policy_mean + loss_policy_std + loss_kl_mean + loss_kl_std

            return total_loss, (jnp.mean(kl_mean_BA), jnp.mean(kl_std_BA))

        (actor_loss, (_mean_kl_mean, _mean_kl_std)), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)
        actor_opt.update(actor, grads)

        # Dual loss — CRITICAL: recompute logsumexp inside so gradient flows through Q/eta
        def dual_loss_fn(dp):
            temp = jax.nn.softplus(dp["log_temperature"]) + 1e-8
            tempered_NB = jax.lax.stop_gradient(q_NB) / temp  # recompute with dp's temperature
            lse_B = jax.scipy.special.logsumexp(tempered_NB, axis=0)  # (B,)
            t_loss = temp * (config.epsilon + jnp.mean(lse_B) - jnp.log(N))

            # Alpha losses for mean and std (stop_gradient on KL is correct for these)
            a_mean = jax.nn.softplus(dp["log_alpha_mean"]) + 1e-8
            a_loss_mean = jnp.sum(a_mean * (config.epsilon_mean - jax.lax.stop_gradient(jnp.mean(kl_mean_BA, axis=0))))

            a_std = jax.nn.softplus(dp["log_alpha_stddev"]) + 1e-8
            a_loss_std = jnp.sum(a_std * (config.epsilon_stddev - jax.lax.stop_gradient(jnp.mean(kl_std_BA, axis=0))))

            return t_loss + a_loss_mean + a_loss_std

        # Need kl stats from actor_loss_fn — recompute here with stop_gradient for the dual
        target_mean_BA_sg, target_std_BA_sg = jax.lax.stop_gradient(target_mean_BA), jax.lax.stop_gradient(target_std_BA)
        online_mean_BA_sg, online_std_BA_sg = jax.lax.stop_gradient(actor.get_mean_std(obs_BD))

        kl_mean_BA = 0.5 * (target_mean_BA_sg - online_mean_BA_sg) ** 2 / (target_std_BA_sg**2 + 1e-8)
        kl_std_BA = (
            jnp.log(online_std_BA_sg / target_std_BA_sg) + (target_std_BA_sg**2) / (2 * online_std_BA_sg**2 + 1e-8) - 0.5
        )

        _dual_loss, dual_grads = jax.value_and_grad(dual_loss_fn)(dual_params)
        dual_updates, new_dual_opt_state = dual_tx.update(dual_grads, carry.dual_opt_state, dual_params)
        new_dual_params = optax.apply_updates(dual_params, dual_updates)
        new_dual_params = jax.tree.map(lambda x: jnp.maximum(x, -18.0), new_dual_params)

        carry = carry._replace(dual_params=new_dual_params, dual_opt_state=new_dual_opt_state)
        return carry, actor_loss, temperature

    def extract_step_metrics(carry, actor_loss, critic_loss, temperature, mean_q):
        alpha_mean = jax.nn.softplus(carry.dual_params["log_alpha_mean"]) + 1e-8
        alpha_stddev = jax.nn.softplus(carry.dual_params["log_alpha_stddev"]) + 1e-8
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "temperature": temperature,
            "alpha_mean": jnp.mean(alpha_mean),
            "alpha_stddev": jnp.mean(alpha_stddev),
            "mean_q_value": mean_q,
        }

    # --- Train iteration ---
    def train_iteration(carry, actor, actor_opt, target_actor, q_network, q_opt, q_target, rngs):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        carry, batch = collect_transitions(carry, actor, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        zero_update_metrics = {k: jnp.float32(0.0) for k in _METRIC_NAMES}

        def update_iteration(_, state):
            carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs, metrics_sum = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)

            critic_loss, mean_q = update_critic(minibatch, q_net, q_o, q_tgt, t_act, rngs())
            carry, actor_loss, temperature = update_actor_and_dual(carry, minibatch, act, act_opt, q_tgt, t_act, rngs())

            step_metrics = extract_step_metrics(carry, actor_loss, critic_loss, temperature, mean_q)
            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, step_metrics)
            return (carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs, metrics_sum)

        def do_updates(carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs):
            carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs, metrics_sum = nnx.fori_loop(
                0,
                config.num_epochs,
                update_iteration,
                (carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs, zero_update_metrics),
            )
            iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs, iter_metrics

        def no_updates(carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs):
            return carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs, zero_update_metrics

        carry, actor, actor_opt, target_actor, q_network, q_opt, q_target, rngs, update_metrics = nnx.cond(
            start_training,
            do_updates,
            no_updates,
            carry,
            actor,
            actor_opt,
            target_actor,
            q_network,
            q_opt,
            q_target,
            rngs,
        )

        q_target = maybe_update_targets(
            q_network, q_target, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )
        target_actor = maybe_update_targets(
            actor, target_actor, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )

        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, update_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, actor, actor_opt, target_actor, q_network, q_opt, q_target, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), eval_env, env_params, 128, max_steps)

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
            **{k: jnp.float32(0.0) for k in _METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs = nnx.fori_loop(
            0, steps_per_iter, train_body, (carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs)
        )

        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)
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

        carry = carry._replace(
            train_metrics=jax.tree.map(lambda x: x * 0, carry.train_metrics),
            metrics_count=carry.metrics_count * 0,
            completed_returns_sum=carry.completed_returns_sum * 0,
            completed_returns_count=carry.completed_returns_count * 0,
        )

        return (carry, act, act_opt, t_act, q_net, q_o, q_tgt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor, actor_opt, target_actor, q_network, q_opt, q_target, rngs), all_metrics = scan_eval(
        (carry, actor, actor_opt, target_actor, q_network, q_opt, q_target, rngs), jnp.zeros(num_evals)
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
        "target_actor": target_actor,
        "q_network": q_network,
        "q_optimizer": q_opt,
        "q_target": q_target,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "dual_params": carry.dual_params,
        "global_step": carry.global_step,
    }
    return state, all_metrics
