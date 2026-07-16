from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from nexrl.types import ActFn, EnvParams, EvalMetrics, GymnaxEnv


class EvalState(NamedTuple):
    rng: jax.Array
    env_state: Any
    last_obs: jax.Array
    done: bool = False
    return_: float = 0.0
    length: int = 0


def evaluate_single(
    act: Any,  # (obs, rng) -> action, or (agent_state, obs, rng) -> action when agent_state is passed
    env: GymnaxEnv,
    env_params: EnvParams,
    rng: jax.Array,
    max_steps_in_episode: int,
    agent_state: Any = None,
) -> tuple[jax.Array, jax.Array]:
    def step(state):
        rng, rng_act, rng_step = jax.random.split(state.rng, 3)
        # `agent_state is None` is a trace-time branch: the default keeps act's (obs, rng)
        # signature; the traced-params path uses (agent_state, obs, rng) so params flow as a
        # runtime input rather than being baked in as compile-time constants.
        action = act(state.last_obs, rng_act) if agent_state is None else act(agent_state, state.last_obs, rng_act)
        obs, env_state, reward, done, _ = env.step(rng_step, state.env_state, action, env_params)
        state = EvalState(
            rng=rng,
            env_state=env_state,
            last_obs=obs,
            done=done,  # pyright: ignore[reportArgumentType]
            return_=state.return_ + reward.squeeze(),
            length=state.length + 1,
        )
        return state

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    state = EvalState(rng_eval, env_state, obs)
    state = jax.lax.while_loop(
        lambda s: jnp.logical_and(s.length < max_steps_in_episode, jnp.logical_not(s.done)),
        step,
        state,
    )
    return state.length, state.return_  # pyright: ignore[reportReturnType]


class _MultiLaneState(NamedTuple):
    rng: jax.Array
    env_state: Any
    last_obs: jax.Array
    total_steps: jax.Array  # int32, safety bound on the whole lane
    ep_idx: jax.Array  # int32, number of episodes already completed on this lane
    cur_return: jax.Array  # float32, return of the in-progress episode
    cur_length: jax.Array  # int32, length of the in-progress episode
    returns: jax.Array  # (n_episodes,) float32, completed-episode returns
    lengths: jax.Array  # (n_episodes,) float32, completed-episode lengths


def _evaluate_multi_lane(
    act: Any,  # (obs, rng) -> action, or (agent_state, obs, rng) -> action when agent_state is passed
    env: GymnaxEnv,
    env_params: EnvParams,
    rng: jax.Array,
    env_idx: jax.Array,
    n_episodes: int,
    max_steps_in_episode: int,
    agent_state: Any = None,
) -> tuple[jax.Array, jax.Array]:
    """Play ``n_episodes`` sequential episodes on the single gymnasium instance
    bound to ``env_idx`` and return their per-episode (lengths, returns).

    Multi-instance envs (Atari/DMC) hold only ``num_envs`` physical gymnasium
    instances, so more than ``num_envs`` *independent* episodes cannot run at once
    in a single vmap — extra lanes would alias an instance, the exact bug fixed
    here. Instead, vmap binds each lane to its own instance via ``reset_with_id``
    (``env_ids = arange(num_envs)``, mirroring ``create_vec_env_fns``), and each
    lane plays its episodes back-to-back: the Atari adapter auto-resets in-step, so
    the game-over step returns the fresh-episode reset obs as ``next_obs`` with
    ``done/real_done=True`` — the loop records that episode's return on the
    ``real_done`` boundary and the next step simply starts from the reset obs (no
    phantom reset step, no length inflation).

    This is a single data-dependent ``while_loop`` chain per lane, so the
    side-effecting ``pure_callback`` steps are strictly ordered — no cross-round
    reordering hazard — making it safe both inside a jit'd training scan and in a
    Python outer loop.
    """
    # Play a FIXED n_episodes to completion so every slot in `returns`/`lengths`
    # holds a real episode — no zero-filled slots to deflate the mean (the core of
    # audit finding A). Each episode is force-terminated at `max_steps_in_episode`
    # (see `step`), so a lane completes n_episodes within this loose backstop; it is
    # defence-in-depth, not the binding guard (that is `ep_idx < n_episodes`).
    max_total_steps = n_episodes * (max_steps_in_episode + 1)

    rng_reset, rng_eval = jax.random.split(rng)
    obs, env_state = env.reset_with_id(rng_reset, env_idx, env_params)
    init = _MultiLaneState(
        rng=rng_eval,
        env_state=env_state,
        last_obs=obs,
        total_steps=jnp.int32(0),
        ep_idx=jnp.int32(0),
        cur_return=jnp.float32(0.0),
        cur_length=jnp.int32(0),
        returns=jnp.zeros(n_episodes, dtype=jnp.float32),
        lengths=jnp.zeros(n_episodes, dtype=jnp.float32),
    )

    def cond(s: _MultiLaneState) -> jax.Array:
        return jnp.logical_and(s.ep_idx < n_episodes, s.total_steps < max_total_steps)

    def step(s: _MultiLaneState) -> _MultiLaneState:
        rng, rng_act, rng_step = jax.random.split(s.rng, 3)
        action = act(s.last_obs, rng_act) if agent_state is None else act(agent_state, s.last_obs, rng_act)
        obs, env_state, reward, done, info = env.step(rng_step, s.env_state, action, env_params)

        cur_return = s.cur_return + reward.squeeze()
        cur_length = s.cur_length + 1

        # Segment on the true GAME boundary, not the agent terminal. With
        # terminal_on_life_loss the env emits `done` per life but `info['real_done']`
        # only on game over; eval must report full-game returns (comparable to the
        # paper). Without the wrapper real_done == done, so this is unchanged.
        game_over = info.get("real_done", done)
        # Per-episode truncation cap (audit finding A): guarantees every episode
        # ends, so all n_episodes complete and the mean has no empty slots even for
        # a strong policy that plays very long games. Mirrors the reference's 27000
        # agent-step cap (BBF.gin:69). At Atari-100k it effectively never fires
        # (Breakout games end far sooner); the rare "capped but not game_over" case
        # (env not autoreset, next episode continues the same game) is an unhit corner.
        boundary = jnp.logical_or(game_over, cur_length >= max_steps_in_episode)

        # On episode end, record this episode into slot ep_idx (guaranteed < n_episodes
        # by `cond`), then zero the accumulators and advance the episode counter.
        returns = jnp.where(boundary, s.returns.at[s.ep_idx].set(cur_return), s.returns)
        lengths = jnp.where(boundary, s.lengths.at[s.ep_idx].set(cur_length.astype(jnp.float32)), s.lengths)
        ep_idx = s.ep_idx + boundary.astype(jnp.int32)
        cur_return = jnp.where(boundary, jnp.float32(0.0), cur_return)
        cur_length = jnp.where(boundary, jnp.int32(0), cur_length)

        return _MultiLaneState(
            rng=rng,
            env_state=env_state,
            last_obs=obs,
            total_steps=s.total_steps + 1,
            ep_idx=ep_idx,
            cur_return=cur_return,
            cur_length=cur_length,
            returns=returns,
            lengths=lengths,
        )

    final = jax.lax.while_loop(cond, step, init)
    return final.lengths, final.returns


@partial(jax.jit, static_argnames=("act", "env", "num_seeds"))
def evaluate(
    act: ActFn,
    rng: jax.Array,
    env: GymnaxEnv,
    env_params: EnvParams,
    num_seeds: int = 128,
    max_steps_in_episode: int | None = None,
    agent_state: Any = None,
) -> EvalMetrics:
    """Evaluate a policy given by `act` on `num_seeds` environments.

    Args:
        act (Callable[[chex.Array, chex.PRNGKey], chex.Array]): A policy represented as
        a function of type (obs, rng) -> action.
        rng (chex.PRNGKey): Initial seed, will be split into `num_seeds` seeds for
        parallel evaluation.
        env (environment.Environment): The environment to evaluate on.
        env_params (Any): The parameters of the environment.
        num_seeds (int): Number of initializations of the environment.

    Returns:
        Tuple[chex.Array, chex.Array]: Tuple of episode length and cumultative reward
        for each seed.
    """
    if max_steps_in_episode is None:
        max_steps_in_episode = env_params.max_steps_in_episode

    assert max_steps_in_episode is not None

    # Pure-JAX (gymnax) envs: every vmapped seed is an independent episode, so map
    # `evaluate_single` over `num_seeds` seeds. Unchanged from the original path.
    if getattr(env, "reset_with_id", None) is None:
        seeds = jax.random.split(rng, num_seeds)
        vmap_collect = jax.vmap(evaluate_single, in_axes=(None, None, None, 0, None, None))
        return vmap_collect(act, env, env_params, seeds, max_steps_in_episode, agent_state)

    # Multi-instance envs (Atari/DMC) expose only `num_envs` physical gymnasium
    # instances behind a `pure_callback`. The base `env.reset` hardcodes instance 0,
    # so vmapping `evaluate_single` over `num_seeds` seeds aliases every lane to one
    # instance and the dynamics get scrambled (the ~0 eval-return artifact). Instead
    # bind each lane to its own instance (`env_ids = arange(num_envs)`) and play
    # `ceil(num_seeds / num_envs)` episodes per lane to reach `num_seeds` total. See
    # `_evaluate_multi_lane` for the jit-safety argument.
    n_inst = env._num_envs  # pyright: ignore[reportAttributeAccessIssue]
    n_episodes = -(-num_seeds // n_inst)  # ceil division
    env_ids = jnp.arange(n_inst, dtype=jnp.int32)
    seeds = jax.random.split(rng, n_inst)
    vmap_collect = jax.vmap(_evaluate_multi_lane, in_axes=(None, None, None, 0, 0, None, None, None))
    # Both arrays are (n_inst, n_episodes).
    lengths, returns = vmap_collect(act, env, env_params, seeds, env_ids, n_episodes, max_steps_in_episode, agent_state)
    # (n_inst, n_episodes) -> (n_inst * n_episodes,), trimmed to the requested count.
    return lengths.reshape(-1)[:num_seeds], returns.reshape(-1)[:num_seeds]
