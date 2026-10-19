import warnings
from dataclasses import asdict
from functools import cached_property

import gymnasium as gym
import jax
import numpy as np
from flax import struct
from gymnasium import Env as GymnasiumEnv
from gymnasium import spaces as gymnasium_spaces
from gymnasium.wrappers import TimeLimit
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import ShapeDtypeStruct
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype


# Gymnasium currently has a functional API, but only few environments are supported
# (e.g. BlackJack, but not classic control). We want to use jax.pure_callback to call
# their methods, so we need to make their API functional. We do this by explicitly
# listing the attributes that are changed during the environment step, and overwriting
# them before stepping in our callback. We seperately handle the rng key.
# All of this nonsense means we lose all vectorization benefits from JAX, but we can
# extend the compat layer to support vectorized environments using multiprocessing
# pools in the future, or using gymnasium VectorizedEnvs with vmap_method broadcast_all


def create_gymnasium(env_name, **kwargs):
    # No gymnasium Autoreset wrapper: gymnax's base `step` already resets in the same step
    # (it calls `reset_env` and selects the reset obs/state on done), and every callback
    # loads the full env state first, so episodes end without a phantom transition.
    env = gym.make(env_name, **kwargs)
    env = Gymnasium2GymnaxEnv(env)
    return env, env.default_params


# The state attributes are the ones that are changed during the step function
# Before stepping, we overwrite them with values of the state to step from
# The random key is sometimes changed as well, we take care of it seperately
STATE_ATTRIBUTES = {
    # Classic control
    "CartPole-v1": ("state",),
    "MountainCar-v0": ("state",),
    "Pendulum-v1": ("state",),
    "Acrobot-v1": ("state",),
    "MountainCarContinuous-v0": ("state",),
}

STATE_SEED_NAME = "_nexrl_seed"
# Step counter of gymnasium's TimeLimit wrapper. It is env state too: without it, the
# `reset_env` call in every gymnax `step` would zero the counter and the time limit
# would never fire.
STATE_ELAPSED_STEPS_NAME = "_nexrl_elapsed_steps"


def _find_time_limit(env):
    """The TimeLimit wrapper in `env`'s wrapper chain, or None (no registered limit)."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, TimeLimit):
            return env
        env = env.env
    return None


def load_state(env, state):
    # not very pretty...
    # gymnasium.Env.np_random_seed is a property without a setter. Maybe it would be
    # better to add one to our env object?
    env._np_random_seed = state[STATE_SEED_NAME]

    for attr in STATE_ATTRIBUTES[env.spec.id]:
        setattr(env.unwrapped, attr, state[attr])

    time_limit = _find_time_limit(env)
    if time_limit is not None:
        time_limit._elapsed_steps = int(state[STATE_ELAPSED_STEPS_NAME])

    # CartPole counts steps past a termination and then pays reward 0. One gymnasium env
    # serves every vmap lane and the state we step from is never terminal (gymnax's
    # `step` resets on done), so clear the counter another lane may have left behind.
    if hasattr(env.unwrapped, "steps_beyond_terminated"):
        env.unwrapped.steps_beyond_terminated = None


def get_state(env):
    state = {}
    state[STATE_SEED_NAME] = env.np_random_seed

    for attr in STATE_ATTRIBUTES[env.spec.id]:
        state[attr] = np.array(getattr(env.unwrapped, attr))

    time_limit = _find_time_limit(env)
    if time_limit is not None:
        state[STATE_ELAPSED_STEPS_NAME] = np.array(time_limit._elapsed_steps, dtype=np.int32)

    return state


def step_gymnasium_env(env: GymnasiumEnv, state, action):
    action = np.array(action)

    load_state(env, state)
    obs, reward, term, trunc, info = env.step(action)
    state = get_state(env)

    done = term or trunc
    obs = obs.astype(jnp.float32)
    # gymnax's base `step` swaps in the reset obs on done, so the pre-reset obs is only
    # available via info. "truncated" = time limit without physical termination.
    info = {"truncated": bool(trunc and not term), "final_obs": obs}
    return obs, state, reward, done, info


def reset_gymnasium_env(env: GymnasiumEnv, seed: int):
    seed = np.array(seed).item()
    obs, _ = env.reset(seed=seed)
    state = get_state(env)
    return obs.astype(jnp.float32), state


def gymnasium_space_to_gymnax_space(gymnasium_space):
    if isinstance(gymnasium_space, gymnasium_spaces.Box):
        # Preserve uint8 dtype for image observations; float32 for everything else
        dtype = jnp.uint8 if gymnasium_space.dtype == np.uint8 else jnp.float32
        return gymnax_spaces.Box(
            low=gymnasium_space.low,
            high=gymnasium_space.high,
            shape=gymnasium_space.shape,
            dtype=dtype,
        )
    elif isinstance(gymnasium_space, gymnasium_spaces.Discrete):
        return gymnax_spaces.Discrete(gymnasium_space.n)
    elif isinstance(gymnasium_space, gymnasium_spaces.Dict):
        return gymnax_spaces.Dict({k: gymnasium_space_to_gymnax_space(v) for k, v in gymnasium_space.spaces.items()})


def num_entries(space):
    if isinstance(space, gymnax_spaces.Discrete):
        return space.n
    elif isinstance(space, gymnax_spaces.Box):
        return jnp.prod(jnp.array(space.shape))
    elif isinstance(space, gymnax_spaces.Dict):
        return sum(num_entries(subspace) for subspace in space.spaces.values())
    raise ValueError(f"Unsupported space {space}")


def shape_dtype_from_pytree(pytree):
    def get_shape_dtype(obj):
        if isinstance(obj, (bool, int, float)):
            dtype = canonicalize_dtype(type(obj))
            return ShapeDtypeStruct((), dtype)
        dtype = canonicalize_dtype(obj.dtype)
        return ShapeDtypeStruct(obj.shape, dtype)

    return jax.tree.map(get_shape_dtype, pytree)


class Gymnasium2GymnaxEnv(GymnaxEnv):
    def __init__(self, env: GymnasiumEnv):
        warnings.warn(
            "Wrapper for Gymnasium2GymnaxEnv is experimental, and may not correctly "
            "handle rng keys. Please report any issues by creating an issue. ",
            category=RuntimeWarning,
            stacklevel=1,
        )
        self.env = env
        self.env.reset()

    @property
    def default_params(self):
        spec = self.env.spec
        param_dict = asdict(spec)
        # `pop(key, 1000)` alone is not enough: `asdict(EnvSpec)` always emits
        # `max_episode_steps`, with value None for envs that register no limit — so the
        # pop default never fires and the params carry None, tripping `evaluate`'s
        # `assert max_steps_in_episode is not None`. Fall back on the value, not the key.
        param_dict["max_steps_in_episode"] = param_dict.pop("max_episode_steps", None) or 1000
        param_cls = type("Gymnasium2GymnaxEnvParams", (), param_dict)
        param_cls = struct.dataclass(param_cls)
        return param_cls()

    @cached_property
    def reset_shape_dtype(self):
        seed = 0
        reset_ph = reset_gymnasium_env(self.env, seed)
        return shape_dtype_from_pytree(reset_ph)

    @cached_property
    def step_shape_dtype(self):
        state = get_state(self.env)
        action = self.env.action_space.sample()
        load_state(self.env, state)

        step_ph = step_gymnasium_env(self.env, state, action)
        return shape_dtype_from_pytree(step_ph)

    def step_env(self, key, state, action, params):
        def step(state, action):
            return step_gymnasium_env(self.env, state, action)

        seed = jax.random.randint(key, (), 0, 2**16)
        state = {**state, STATE_SEED_NAME: seed}  # copy: don't mutate the caller's state
        return jax.pure_callback(
            step,
            self.step_shape_dtype,
            vmap_method="sequential",
            state=state,
            action=action,
        )

    def reset_env(self, key, params):
        def reset(seed):
            return reset_gymnasium_env(self.env, seed)

        seed = jax.random.randint(key, (), 0, 2**16)
        return jax.pure_callback(reset, self.reset_shape_dtype, vmap_method="sequential", seed=seed)

    def get_obs(self, state):
        raise NotImplementedError

    def is_terminal(self, state):
        return NotImplementedError

    @property
    def name(self):
        return self.env.unwrapped.__class__.__name__

    def action_space(self, params):
        return gymnasium_space_to_gymnax_space(self.env.action_space)

    def observation_space(self, params):
        return gymnasium_space_to_gymnax_space(self.env.observation_space)

    @property
    def num_actions(self) -> int:
        return num_entries(self.action_space(self.default_params))


# ---------------------------------------------------------------------------
# Multi-instance adapter for complex envs (Atari, DMC)
# ---------------------------------------------------------------------------


def _step_multi_instance(envs, env_idx, action, *, autoreset_in_step: bool = False):
    """Step a single env instance identified by env_idx.

    ``autoreset_in_step`` selects the auto-reset convention:

    * ``True`` (Atari, DMC): the env has NO ``Autoreset`` wrapper; instead we reset this
      instance IN THE SAME STEP as a real episode end and return the reset obs as
      ``next_obs`` (with ``done`` still True). This matches every pure-JAX gymnax env and
      Dopamine/BBF, and avoids the phantom ``(terminal_obs, discarded_action, reward=0,
      reset_obs, done=False)`` transition that gymnasium's ``Autoreset`` (a two-step reset
      that discards the post-terminal action) otherwise stores at max priority every game
      over (it corrupts n-step/SPR windows anchored there).
    * ``False`` (default, no factory uses it): no reset here. The env must reset itself,
      e.g. with a gymnasium ``Autoreset`` wrapper, and then shows the phantom above.

    Truncation info (same convention as every nexrl adapter): ``info['truncated']`` is
    ``trunc and not term`` — the time limit ended the episode without a physical
    termination — and ``info['final_obs']`` is the obs before the in-step reset (equal to
    ``obs`` when no reset happened). Both refer to the GAME-level end: EpisodicLifeEnv
    passes the inner ``terminated``/``truncated`` through unchanged.

    With EpisodicLifeEnv (terminal_on_life_loss), ``done`` is the GAME boundary (game over)
    and ``info['life_terminal']`` marks a life loss that continues the game (NO reset);
    ``info['real_done']`` also marks game over. Without the wrapper, life_terminal is absent
    (False) and real_done coincides with done. Downstream, the buffer/n-step terminal is
    ``done OR life_terminal`` (per-life cut) and ``real_done`` drives full-game return
    reporting and eval segmentation. Because the in-step reset is gated on ``real_done``, a
    life loss never triggers it — the game continues with the agent's real next action.
    """
    idx = int(env_idx)
    action = np.array(action)
    if action.ndim == 0:
        action = action.item()
    obs, reward, term, trunc, info = envs[idx].step(action)
    done = term or trunc
    real_done = bool(info.get("real_done", done))
    life_terminal = bool(info.get("life_terminal", False))
    truncated = bool(trunc and not term)
    final_obs = np.asarray(obs)
    # gymnax-convention auto-reset: on a real episode end, reset this instance now and
    # return the fresh-episode reset frame as next_obs (instead of the terminal frame).
    # `done` stays True, so downstream n-step/SPR windows still cut here and mask next_obs.
    # A terminated gymnasium env must be reset before it can be stepped again anyway, so
    # this both removes the phantom and keeps the instance steppable.
    if autoreset_in_step and real_done:
        obs, _ = envs[idx].reset()
    obs = np.asarray(obs)
    return (
        obs,
        np.array(idx, dtype=np.int32),
        float(reward),
        done,
        {"real_done": real_done, "life_terminal": life_terminal, "truncated": truncated, "final_obs": final_obs},
    )


def _reset_multi_instance(envs, env_idx, seed):
    """Reset a single env instance identified by env_idx."""
    idx = int(env_idx)
    seed = np.array(seed).item()
    obs, _ = envs[idx].reset(seed=seed)
    obs = np.asarray(obs)
    return obs, np.array(idx, dtype=np.int32)


class MultiInstanceGymnasium2GymnaxEnv(GymnaxEnv):
    """Multi-instance adapter: N independent gymnasium env instances.

    Instead of serializing complex env state into JAX pytrees, maintains
    num_envs independent env instances. State is just an _env_idx integer.
    """

    def __init__(self, make_fn, num_envs: int = 1, max_steps: int | None = None, autoreset_in_step: bool = False):
        self.make_fn = make_fn
        self._num_envs = num_envs
        # gymnax-convention auto-reset inside step_env (env has no gymnasium Autoreset
        # wrapper). Atari and DMC set this True to avoid the phantom post-terminal
        # transition. See _step_multi_instance for the two conventions.
        self._autoreset_in_step = autoreset_in_step
        # Explicit per-episode step cap. ALE-v5 registers no ``max_episode_steps``
        # on its spec, so without this ``default_params`` falls back to 1000 — far
        # below a real Atari episode. Callers (e.g. ``create_atari``) pass the true
        # frame cap (108000 frames / frame_skip). ``None`` keeps the spec fallback.
        self._max_steps = max_steps
        self.envs = [make_fn() for _ in range(num_envs)]
        # Do an initial reset to determine shapes
        obs, _ = self.envs[0].reset(seed=0)
        self._obs_array = np.asarray(obs)

    def clone(self) -> "MultiInstanceGymnasium2GymnaxEnv":
        """Build a second set of instances with identical configuration.

        ``self.envs`` holds live emulator handles mutated in place inside
        ``pure_callback``, so anything that drives this env (notably evaluation)
        destroys whatever episode the training loop has in flight. Callers that
        need to step an env without disturbing training clone it first; see
        ``nexrl.algos.utils.make_eval_env``.

        Every constructor argument must be propagated, ``autoreset_in_step``
        above all: the ``make_fn`` of ``create_atari`` and ``create_dmc`` builds NO
        gymnasium ``Autoreset`` wrapper (the adapter resets in-step instead), so a clone
        left at the ``False`` default would never reset — every lane would stick
        in a terminated env after its first game over and returns would collapse
        to ~0. ``_max_steps`` is forwarded raw (possibly ``None``) rather than as
        ``default_params.max_steps_in_episode``, which would bake in the spec
        fallback and change DMC's semantics.
        """
        return type(self)(
            self.make_fn,
            self._num_envs,
            max_steps=self._max_steps,
            autoreset_in_step=self._autoreset_in_step,
        )

    @property
    def default_params(self):
        from nexrl.compat._env_params import make_env_params

        # `is not None` rather than truthiness: an explicit max_steps=0 must not
        # silently fall through to the spec/1000 fallback. The remaining terms keep
        # truthiness on purpose — ALE-v5 *has* a `max_episode_steps` attribute whose
        # value is None, which must fall through to 1000.
        if self._max_steps is not None:
            max_steps = self._max_steps
        else:
            max_steps = getattr(self.envs[0].spec, "max_episode_steps", None) or 1000
        return make_env_params(max_steps)

    @cached_property
    def _obs_dtype(self):
        return canonicalize_dtype(self._obs_array.dtype)

    @cached_property
    def reset_shape_dtype(self):
        obs_sd = ShapeDtypeStruct(self._obs_array.shape, self._obs_dtype)
        idx_sd = ShapeDtypeStruct((), jnp.int32)
        return obs_sd, idx_sd

    @cached_property
    def step_shape_dtype(self):
        obs_sd = ShapeDtypeStruct(self._obs_array.shape, self._obs_dtype)
        idx_sd = ShapeDtypeStruct((), jnp.int32)
        reward_sd = ShapeDtypeStruct((), jnp.float32)
        done_sd = ShapeDtypeStruct((), jnp.bool_)
        info_sd = {
            "real_done": ShapeDtypeStruct((), jnp.bool_),
            "life_terminal": ShapeDtypeStruct((), jnp.bool_),
            "truncated": ShapeDtypeStruct((), jnp.bool_),
            "final_obs": obs_sd,
        }
        return obs_sd, idx_sd, reward_sd, done_sd, info_sd

    def step(self, key, state, action, params=None):
        """Override gymnax base step to avoid calling reset_env every step.

        The base class computes both step_env and reset_env on every call
        (safe for pure-JAX envs), but with pure_callback the reset is a
        destructive side-effect that wipes the gymnasium env state. Episode
        boundaries are handled inside the step: either by a gymnasium ``Autoreset``
        wrapper (``autoreset_in_step=False``) or by an in-step reset
        (``autoreset_in_step=True``); see ``_step_multi_instance``.
        """
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)

    def step_env(self, key, state, action, params):
        autoreset_in_step = self._autoreset_in_step

        def step(env_idx, action):
            return _step_multi_instance(self.envs, env_idx, action, autoreset_in_step=autoreset_in_step)

        return jax.pure_callback(
            step,
            self.step_shape_dtype,
            vmap_method="sequential",
            env_idx=state,
            action=action,
        )

    def reset_env(self, key, params):
        # WARNING: a pure_callback cannot recover its own vmap-lane index, so this
        # base reset hardcodes instance 0. Under `jax.vmap(env.reset)` EVERY lane
        # therefore aliases instance 0 and `step_env` (which routes via env_idx)
        # steps a single instance for all lanes — corrupting transitions whenever
        # num_envs>1. Correct multi-env use goes through `reset_with_id` (wired in
        # `create_vec_env_fns`), which binds each lane to its own instance. This
        # path stays correct only for num_envs==1.
        def reset(env_idx, seed):
            return _reset_multi_instance(self.envs, env_idx, seed)

        seed = jax.random.randint(key, (), 0, 2**16)
        env_idx = jnp.int32(0)
        return jax.pure_callback(
            reset,
            self.reset_shape_dtype,
            vmap_method="sequential",
            env_idx=env_idx,
            seed=seed,
        )

    def reset_with_id(self, key, env_idx, params=None):
        """Reset a SPECIFIC instance ``env_idx`` and bind this lane to it.

        Vmap this over ``(keys, env_ids=arange(num_envs))`` so each lane resets and
        is permanently bound to its own gymnasium instance. This is the fix for the
        aliasing bug described in ``reset_env``: instead of a hardcoded 0, the
        per-lane ``env_idx`` flows into the callback (it IS a vmapped argument here),
        so lane ``i`` resets instance ``i`` and returns ``i`` as its state.
        ``step_env`` already routes via ``env_idx=state``, keeping the binding.
        ``params`` is accepted (and ignored) only to match the vmap signature.
        """
        del params

        def reset(env_idx, seed):
            return _reset_multi_instance(self.envs, env_idx, seed)

        seed = jax.random.randint(key, (), 0, 2**16)
        return jax.pure_callback(
            reset,
            self.reset_shape_dtype,
            vmap_method="sequential",
            env_idx=env_idx,
            seed=seed,
        )

    def get_obs(self, state):
        raise NotImplementedError

    def is_terminal(self, state):
        raise NotImplementedError

    @property
    def name(self):
        return self.envs[0].unwrapped.__class__.__name__

    def action_space(self, params):
        return gymnasium_space_to_gymnax_space(self.envs[0].action_space)

    def observation_space(self, params):
        return gymnasium_space_to_gymnax_space(self.envs[0].observation_space)

    @property
    def num_actions(self) -> int:
        return num_entries(self.action_space(self.default_params))


# ---------------------------------------------------------------------------
# Atari environment creation
# ---------------------------------------------------------------------------


class EpisodicLifeEnv(gym.Wrapper):
    """End-of-life == end-of-episode for the AGENT, while the GAME continues.

    Dopamine/SB3-style ``terminal_on_life_loss`` (used by Schwarzer's BBF runner):
    the agent receives ``terminated=True`` when a life is lost so it does not
    bootstrap value across the life boundary, but the ALE game is NOT reset on a life
    loss — only on a true game over — so the agent keeps experiencing the later,
    deeper states (essential for Breakout, where broken bricks persist across lives).

    Reference parity (Schwarzer's ``DataEfficientAtariRunner._run_parallel``,
    ``bbf/eval_run_experiment.py``): a life loss does NOT reset the env. On a life
    loss the reference stores the per-life terminal in the buffer but only calls
    ``agent.reset_one`` (its ``elif d:`` branch) — it does NOT re-initialise the env,
    so the agent's REAL next action continues the same game. Only a true game over
    triggers a real reset. This wrapper therefore does NOT raise a gym terminal on a
    life loss: it passes the underlying ``terminated`` (game over only) through as
    ``real_done`` so the adapter's in-step auto-reset fires ONLY at game over, and
    carries the per-life terminal out-of-band in ``info['life_terminal']``. Downstream
    (BBF ``jit_collect``) the buffer/n-step terminal is ``done OR life_terminal``
    (bootstrap cut per life), while ``info['real_done']`` (game over) drives the
    reported full-game return and eval episode segmentation.

    Why keep the per-life terminal out of the gym ``done`` channel: if a life loss
    raised ``real_done``, the adapter would reset the ALE game on every life (wiping
    Breakout's persistent bricks). Passing only the game-over terminal keeps resets to
    one per game (game over), on par with non-TLL Atari. (Historically, under the old
    gymnasium ``Autoreset`` wrapper, raising a per-life terminal also injected ~5
    corrupted discarded-action / reward-0 transitions per game, which halved Breakout.)

    Placement: wrap the frame-stacked env. The Atari factory adds NO outer ``Autoreset``
    wrapper — the adapter resets in-step (``autoreset_in_step=True``). Sitting above
    FrameStack keeps the stacked-frame history continuous across a life boundary,
    matching the reference.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        real_done = bool(terminated or truncated)
        self.was_real_done = real_done
        # ale.lives() via the wrapped chain (attribute-forwarded down to the ALE env).
        new_lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        # Life lost but game not truly over -> agent-only terminal (game continues).
        life_terminal = 0 < new_lives < self.lives
        self.lives = new_lives
        # Pass through the GAME-OVER terminal only (so the adapter's in-step reset fires
        # only at game over); carry the per-life terminal out-of-band. No env reset on a
        # life loss.
        info = {**info, "real_done": real_done, "life_terminal": bool(life_terminal)}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # A life loss no longer raises a gym terminal, so the adapter's in-step reset only
        # fires at a true game over: this is always a real ALE reset starting a fresh
        # game (matching the reference's _initialize_episode at game over).
        obs, info = self.env.reset(seed=seed, options=options)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        self.was_real_done = True
        info = {**info, "real_done": False, "life_terminal": False}
        return obs, info


def create_atari(env_name, *, num_envs: int = 1, terminal_on_life_loss: bool = False, noop_max: int = 30, **kwargs):
    """Create Atari environment with standard preprocessing.

    Observation: (4, 84, 84) uint8 (grayscale, frame-stacked).

    Args:
        terminal_on_life_loss: if True, the agent sees a terminal on each lost life
            (via ``EpisodicLifeEnv``) while the game continues across lives and the
            full-game return is preserved through ``info['real_done']``. Matches
            Schwarzer's BBF runner; off by default to keep other Atari configs intact.
        noop_max: random no-op starts per reset (gymnasium AtariPreprocessing default
            30). Dopamine's DrQ path applies none — pass 0 for reference fidelity.
    """
    import ale_py  # noqa: F401

    # Atari-100k convention (and Schwarzer's BBF gin: sticky_actions=False): disable
    # sticky actions. ALE v5 defaults repeat_action_probability=0.25; override to 0.
    # Caller-supplied kwargs win.
    kwargs.setdefault("repeat_action_probability", 0.0)

    frame_skip = 4

    def make_fn():
        env = gym.make(f"ALE/{env_name}", frameskip=1, **kwargs)
        env = gym.wrappers.AtariPreprocessing(
            env, noop_max=noop_max, frame_skip=frame_skip, grayscale_obs=True, scale_obs=False
        )
        env = gym.wrappers.FrameStackObservation(env, 4)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)  # above FrameStack (see EpisodicLifeEnv docstring)
        # NO gymnasium Autoreset wrapper: the adapter auto-resets IN-STEP
        # (autoreset_in_step=True below), matching the gymnax convention and Dopamine/BBF.
        # gymnasium's Autoreset does a two-step reset that discards the post-terminal action
        # and injects a phantom (reset_obs, reward=0, done=False) transition at max priority
        # every game over -> corrupts n-step/SPR anchors. In-step reset removes it.
        return env

    # Per-episode agent-step cap = ALE's 108000-frame limit / frame_skip = 27000
    # (Dopamine/Schwarzer's BBF max_steps_per_episode; BBF.gin:69). ALE-v5 leaves
    # spec.max_episode_steps=None, so pass it explicitly — otherwise eval falls back
    # to a 1000-step cap that truncates real games and deflates eval returns.
    max_steps = 108_000 // frame_skip
    wrapped = MultiInstanceGymnasium2GymnaxEnv(make_fn, num_envs, max_steps=max_steps, autoreset_in_step=True)
    return wrapped, wrapped.default_params


# ---------------------------------------------------------------------------
# DeepMind Control environment creation
# ---------------------------------------------------------------------------


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat action N times, summing rewards and breaking on termination or truncation.

    Returns the ``terminated``/``truncated`` flags of the last inner step, so an inner
    ``TimeLimit`` truncation reaches the adapter's ``info['truncated']`` unchanged.
    """

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                break
        return obs, total_reward, term, trunc, info


def create_dmc(
    env_name,
    *,
    num_envs: int = 1,
    pixels: bool = False,
    rgb: bool = False,
    action_repeat: int = 1,
    frame_stack: int = 3,
    **kwargs,
):
    """Create DMC environment via shimmy.

    State-based: flat obs, MLP-compatible.
    Pixel-based (grayscale): (frame_stack, 84, 84) uint8.
    Pixel-based (rgb): (frame_stack*3, 84, 84) uint8.

    Args:
        env_name: DMC task name (e.g. "cartpole-balance").
        num_envs: Number of independent env instances.
        pixels: Enable pixel observations.
        rgb: Use RGB instead of grayscale (requires pixels=True).
        action_repeat: Repeat each action N times.
        frame_stack: Number of frames to stack.
    """
    import shimmy  # noqa: F401

    def make_fn():
        env = gym.make(f"dm_control/{env_name}", render_mode="rgb_array" if pixels else None, **kwargs)
        if action_repeat > 1:
            env = ActionRepeatWrapper(env, action_repeat)
        if pixels:
            # Convert dict obs to rendered pixel obs
            env = gym.wrappers.AddRenderObservation(env, render_only=True)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            if rgb:
                # HWC (84,84,3) → CHW (3,84,84)
                env = gym.wrappers.TransformObservation(
                    env,
                    lambda obs: np.transpose(obs, (2, 0, 1)),
                    gymnasium_spaces.Box(0, 255, shape=(3, 84, 84), dtype=np.uint8),
                )
                # FrameStack → (frame_stack, 3, 84, 84), flatten to (frame_stack*3, 84, 84)
                env = gym.wrappers.FrameStackObservation(env, frame_stack)
                stacked_shape = (frame_stack * 3, 84, 84)
                env = gym.wrappers.ReshapeObservation(env, stacked_shape)
            else:
                env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
                # (84, 84) → FrameStack → (frame_stack, 84, 84) — already CHW
                env = gym.wrappers.FrameStackObservation(env, frame_stack)
        # NO gymnasium Autoreset wrapper: the adapter auto-resets IN-STEP, as for Atari
        # (see create_atari and _step_multi_instance). Autoreset's two-step reset would
        # add a phantom (reset_obs, reward=0, done=False) transition after every episode.
        return env

    wrapped = MultiInstanceGymnasium2GymnaxEnv(make_fn, num_envs, autoreset_in_step=True)
    # Adjust max_steps for action repeat
    if action_repeat > 1:
        from nexrl.compat._env_params import make_env_params

        base_max_steps = getattr(wrapped.envs[0].spec, "max_episode_steps", None) or 1000
        adjusted_max_steps = base_max_steps // action_repeat
        env_params = make_env_params(adjusted_max_steps)
        return wrapped, env_params
    return wrapped, wrapped.default_params
