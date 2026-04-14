"""Maximum a-Posteriori Policy Optimization (MPO) — discrete action spaces.

Off-policy with replay buffer. Uses a two-step EM procedure: E-step computes
a non-parametric improved policy from Q-values over all actions, M-step fits
the parametric actor via supervised cross-entropy with KL trust-region
constraints enforced by Lagrange dual variables.

Reference: Abdolmaleki et al., "Maximum a Posteriori Policy Optimisation" (2018).

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (num_actions)
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
    create_encoder,
    create_env,
    create_vec_env_fns,
    make_eval_act,
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
from nexrl.networks import DiscretePolicy, DiscreteQNetwork
from nexrl.types import EnvParams, GymnaxEnv, Metrics, TrainState


_METRIC_NAMES = ("actor_loss", "critic_loss", "temperature", "alpha", "mean_q_value")


class MPOCarry(NamedTuple):
    """Non-module carry state for discrete MPO training."""

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
) -> tuple[DiscretePolicy, nnx.Optimizer, DiscretePolicy, DiscreteQNetwork, nnx.Optimizer, DiscreteQNetwork]:
    """Create discrete actor (+ target), critic (+ target), and optimizers."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(action_space.n)

    actor_kwargs = resolve_network_kwargs(config.actor_kwargs)
    critic_kwargs = resolve_network_kwargs(config.critic_kwargs)

    rngs = nnx.Rngs(rng)

    actor_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    target_actor_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    q_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None
    qt_enc = create_encoder(config.encoder_kwargs, obs_space.shape, rngs) if config.encoder_kwargs else None

    actor = DiscretePolicy(in_features=in_features, action_dim=action_dim, **actor_kwargs, rngs=rngs, encoder=actor_enc)
    target_actor = DiscretePolicy(
        in_features=in_features, action_dim=action_dim, **actor_kwargs, rngs=rngs, encoder=target_actor_enc
    )
    nnx.update(target_actor, nnx.state(actor, nnx.Param))

    actor_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate=config.learning_rate))
    actor_opt = nnx.Optimizer(actor, actor_tx, wrt=nnx.Param)

    q_network = DiscreteQNetwork(in_features=in_features, action_dim=action_dim, **critic_kwargs, rngs=rngs, encoder=q_enc)
    q_target = DiscreteQNetwork(in_features=in_features, action_dim=action_dim, **critic_kwargs, rngs=rngs, encoder=qt_enc)
    nnx.update(q_target, nnx.state(q_network, nnx.Param))

    q_tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate=config.critic_learning_rate))
    q_opt = nnx.Optimizer(q_network, q_tx, wrt=nnx.Param)

    return actor, actor_opt, target_actor, q_network, q_opt, q_target


def train_mpo(
    config: MPOConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train a discrete MPO agent.

    Maximum a-Posteriori Policy Optimization for discrete action spaces.

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

        config = MPOConfig.from_yaml("configs/mpo/gymnax/cartpole.yaml")
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        state, metrics = jax.jit(
            jax.vmap(get_train_fn("mpo"), in_axes=(None, 0)),
            static_argnums=(0,),
        )(config, keys)
    """
    config.validate()
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    max_steps = env_params.max_steps_in_episode

    vmap_step, vmap_reset = create_vec_env_fns(env, config)

    rngs = nnx.Rngs(rng)
    actor, actor_opt, target_actor, q_network, q_opt, q_target = _create_networks(config, env, env_params, rngs())

    dual_params: dict[str, jax.Array] = {
        "log_temperature": jnp.array(config.init_log_temperature),
        "log_alpha": jnp.array(config.init_log_alpha),
    }
    dual_tx = optax.adam(config.dual_learning_rate)
    dual_opt_state = dual_tx.init(dual_params)

    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _METRIC_NAMES}

    carry = MPOCarry(
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

    # --- Collect transitions ---
    def collect_transitions(carry, actor, rngs):
        last_obs = carry.last_obs
        if config.normalize_observations:
            last_obs = normalize_obs(carry.obs_rms, last_obs)

        action_B = actor.act(last_obs, rngs())  # (B,)

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_B, env_params)

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

        mb = Minibatch(obs=carry.last_obs, action=action_B, reward=reward_B, next_obs=next_obs, done=done_B)
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
    def update_critic(mb, q_network, q_opt, q_target, target_actor, _rng):
        target_logits_BA = target_actor.get_logits(mb.next_obs)
        target_probs_BA = jax.nn.softmax(target_logits_BA)
        next_q_BA = q_target(mb.next_obs)
        v_next_B = jnp.sum(target_probs_BA * next_q_BA, axis=-1)
        target_B = mb.reward + (1 - mb.done) * config.gamma * jax.lax.stop_gradient(v_next_B)

        def critic_loss_fn(q_model):
            q_B = q_model.take(mb.obs, mb.action)
            return optax.l2_loss(q_B, target_B).mean(), q_B.mean()

        (critic_loss, mean_q), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(q_network)
        q_opt.update(q_network, grads)
        return critic_loss, mean_q

    # --- Actor + dual update ---
    def update_actor_and_dual(carry, mb, actor, actor_opt, q_target, target_actor, _rng):
        dual_params = carry.dual_params

        # E-step: compute non-parametric improved policy
        q_BA = jax.lax.stop_gradient(q_target(mb.obs))  # (B, A)
        target_logits_BA = jax.lax.stop_gradient(target_actor.get_logits(mb.obs))  # (B, A)

        temperature = jax.nn.softplus(dual_params["log_temperature"]) + 1e-8
        tempered_q_BA = q_BA / temperature

        # Non-parametric policy: softmax(Q/temp + log pi_target)
        log_target_probs_BA = jax.nn.log_softmax(target_logits_BA)
        unnorm_BA = tempered_q_BA + log_target_probs_BA
        e_step_logits_BA = jax.nn.log_softmax(unnorm_BA)  # normalized log-probs

        # M-step: fit actor to non-parametric policy
        def actor_loss_fn(actor_model):
            online_logits_BA = actor_model.get_logits(mb.obs)

            e_step_probs_BA = jnp.exp(e_step_logits_BA)
            log_online_BA = jax.nn.log_softmax(online_logits_BA)
            loss_policy = -jnp.sum(e_step_probs_BA * log_online_BA, axis=-1).mean()

            alpha = jax.nn.softplus(dual_params["log_alpha"]) + 1e-8
            target_probs_BA = jax.nn.softmax(target_logits_BA)
            log_target_BA = jax.nn.log_softmax(target_logits_BA)
            kl_B = jnp.sum(target_probs_BA * (log_target_BA - log_online_BA), axis=-1)
            loss_kl = jax.lax.stop_gradient(alpha) * jnp.mean(kl_B)

            return loss_policy + loss_kl, kl_B.mean()

        (actor_loss, mean_kl), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)
        actor_opt.update(actor, grads)

        # Dual loss
        def dual_loss_fn(dp):
            temp = jax.nn.softplus(dp["log_temperature"]) + 1e-8
            # Recompute log(Z) with differentiable temperature for correct gradient
            tempered = q_BA / temp
            unnorm = tempered + log_target_probs_BA
            log_z_B = jax.scipy.special.logsumexp(unnorm, axis=-1)
            t_loss = temp * (config.epsilon + jnp.mean(log_z_B))
            a = jax.nn.softplus(dp["log_alpha"]) + 1e-8
            a_loss = a * (config.epsilon_policy - jax.lax.stop_gradient(mean_kl))
            return t_loss + a_loss

        _dual_loss, dual_grads = jax.value_and_grad(dual_loss_fn)(dual_params)
        dual_updates, new_dual_opt_state = dual_tx.update(dual_grads, carry.dual_opt_state, dual_params)
        new_dual_params = optax.apply_updates(dual_params, dual_updates)
        new_dual_params = jax.tree.map(lambda x: jnp.maximum(x, -18.0), new_dual_params)

        carry = carry._replace(dual_params=new_dual_params, dual_opt_state=new_dual_opt_state)
        return carry, actor_loss, temperature

    def extract_step_metrics(carry, actor_loss, critic_loss, temperature, mean_q):
        alpha = jax.nn.softplus(carry.dual_params["log_alpha"]) + 1e-8
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "temperature": temperature,
            "alpha": alpha,
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
