import warnings
from dataclasses import asdict
from functools import cached_property

import gymnasium as gym
import jax
import numpy as np
from flax import struct
from gymnasium import Env as GymnasiumEnv
from gymnasium import spaces as gymnasium_spaces
from gymnasium.wrappers import Autoreset
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
    env = gym.make(env_name, **kwargs)
    env = Autoreset(env)
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


def load_state(env, state):
    # not very pretty...
    # gymnasium.Env.np_random_seed is a property without a setter. Maybe it would be
    # better to add one to our env object?
    env._np_random_seed = state[STATE_SEED_NAME]

    for attr in STATE_ATTRIBUTES[env.spec.id]:
        setattr(env.unwrapped, attr, state[attr])


def get_state(env):
    state = {}
    state[STATE_SEED_NAME] = env.np_random_seed

    for attr in STATE_ATTRIBUTES[env.spec.id]:
        state[attr] = np.array(getattr(env.unwrapped, attr))

    return state


def step_gymnasium_env(env: GymnasiumEnv, state, action):
    action = np.array(action)

    load_state(env, state)
    obs, reward, term, trunc, info = env.step(action)
    state = get_state(env)

    done = term or trunc
    info = {}
    return obs.astype(jnp.float32), state, reward, done, info


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
        param_dict["max_steps_in_episode"] = param_dict.pop("max_episode_steps", 1000)
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
        state[STATE_SEED_NAME] = seed
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
        return jax.pure_callback(reset, self.reset_shape_dtype, seed=seed)

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


def _step_multi_instance(envs, env_idx, action):
    """Step a single env instance identified by env_idx."""
    idx = int(env_idx)
    action = np.array(action)
    if action.ndim == 0:
        action = action.item()
    obs, reward, term, trunc, _ = envs[idx].step(action)
    done = term or trunc
    obs = np.asarray(obs)
    return obs, np.array(idx, dtype=np.int32), float(reward), done, {}


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

    def __init__(self, make_fn, num_envs: int = 1):
        self.make_fn = make_fn
        self._num_envs = num_envs
        self.envs = [make_fn() for _ in range(num_envs)]
        # Do an initial reset to determine shapes
        obs, _ = self.envs[0].reset(seed=0)
        self._obs_array = np.asarray(obs)

    @property
    def default_params(self):
        from nexrl.compat._env_params import make_env_params

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
        info_sd = {}
        return obs_sd, idx_sd, reward_sd, done_sd, info_sd

    def step(self, key, state, action, params=None):
        """Override gymnax base step to avoid calling reset_env every step.

        The base class computes both step_env and reset_env on every call
        (safe for pure-JAX envs), but with pure_callback the reset is a
        destructive side-effect that wipes the gymnasium env state.
        Autoreset already handles episode boundaries at the gymnasium level.
        """
        if params is None:
            params = self.default_params
        return self.step_env(key, state, action, params)

    def step_env(self, key, state, action, params):
        def step(env_idx, action):
            return _step_multi_instance(self.envs, env_idx, action)

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


def create_atari(env_name, *, num_envs: int = 1, **kwargs):
    """Create Atari environment with standard preprocessing.

    Observation: (4, 84, 84) uint8 (grayscale, frame-stacked).
    """
    import ale_py  # noqa: F401

    # Atari-100k convention (and Schwarzer's BBF gin: sticky_actions=False): disable
    # sticky actions. ALE v5 defaults repeat_action_probability=0.25; override to 0.
    # Caller-supplied kwargs win. terminal_on_life_loss is intentionally NOT enabled
    # here yet — it needs the reported return decoupled from the per-life terminal
    # (tracked as a follow-up).
    kwargs.setdefault("repeat_action_probability", 0.0)

    def make_fn():
        env = gym.make(f"ALE/{env_name}", frameskip=1, **kwargs)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = Autoreset(env)
        return env

    wrapped = MultiInstanceGymnasium2GymnaxEnv(make_fn, num_envs)
    return wrapped, wrapped.default_params


# ---------------------------------------------------------------------------
# DeepMind Control environment creation
# ---------------------------------------------------------------------------


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat action N times, summing rewards and breaking on termination."""

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
        env = Autoreset(env)
        return env

    wrapped = MultiInstanceGymnasium2GymnaxEnv(make_fn, num_envs)
    # Adjust max_steps for action repeat
    if action_repeat > 1:
        from nexrl.compat._env_params import make_env_params

        base_max_steps = getattr(wrapped.envs[0].spec, "max_episode_steps", None) or 1000
        adjusted_max_steps = base_max_steps // action_repeat
        env_params = make_env_params(adjusted_max_steps)
        return wrapped, env_params
    return wrapped, wrapped.default_params
