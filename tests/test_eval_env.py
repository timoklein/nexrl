"""Tests for the dedicated-evaluation-env machinery.

Evaluation must never drive the live training emulator instances: for stateful
multi-instance adapters (Atari, DMC) `evaluate` calls `reset_with_id`/`step` on the
very instances the training loop has episodes in flight on, destroying one training
episode per env per eval point. `make_eval_env` + `MultiInstanceGymnasium2GymnaxEnv.clone`
give evaluation its own instances.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from nexrl.algos.utils import FloatObsWrapper, make_eval_env
from nexrl.compat.gymnasium2gymnax import MultiInstanceGymnasium2GymnaxEnv


class _CloneableEnv:
    """Stand-in for a stateful adapter: `clone` returns a fresh object."""

    def __init__(self):
        self.envs = [object()]

    def clone(self):
        return _CloneableEnv()


class _PureJaxEnv:
    """Stand-in for a functional env: no `clone`, safe to share."""


class _FakeGymEnv:
    """Minimal gymnasium-like env, enough to construct the multi-instance adapter."""

    def __init__(self, max_episode_steps):
        self.spec = SimpleNamespace(max_episode_steps=max_episode_steps)

    def reset(self, seed=None):
        return np.zeros((2,), dtype=np.uint8), {}


def _make_multi(num_envs=2, max_steps=None, autoreset_in_step=False, spec_max=None):
    return MultiInstanceGymnasium2GymnaxEnv(
        lambda: _FakeGymEnv(spec_max),
        num_envs,
        max_steps=max_steps,
        autoreset_in_step=autoreset_in_step,
    )


# --- make_eval_env dispatch ------------------------------------------------


def test_cloneable_env_is_cloned():
    env = _CloneableEnv()
    assert make_eval_env(env) is not env


def test_disabled_returns_training_env():
    """run_evaluation=False must not pay for a second set of emulator instances."""
    env = _CloneableEnv()
    assert make_eval_env(env, False) is env


def test_pure_jax_env_is_shared():
    """Functional envs carry their state in the caller's pytree — sharing is harmless."""
    env = _PureJaxEnv()
    assert make_eval_env(env) is env


def test_float_obs_wrapper_is_preserved_and_inner_cloned():
    """Regression test: FloatObsWrapper delegates `clone` via __getattr__.

    A naive `getattr(env, "clone")()` would return the *inner* env, silently dropping
    the float cast, and an `isinstance(env, MultiInstance...)` check would not fire at
    all. Both must keep the wrapper and clone through it.
    """
    inner = _CloneableEnv()
    eval_env = make_eval_env(FloatObsWrapper(inner))
    assert isinstance(eval_env, FloatObsWrapper)
    assert eval_env.env is not inner
    assert isinstance(eval_env.env, _CloneableEnv)


def test_float_obs_wrapper_around_pure_jax_env():
    inner = _PureJaxEnv()
    eval_env = make_eval_env(FloatObsWrapper(inner))
    assert isinstance(eval_env, FloatObsWrapper)
    assert eval_env.env is inner


# --- MultiInstanceGymnasium2GymnaxEnv.clone --------------------------------


def test_clone_propagates_every_ctor_arg():
    """`autoreset_in_step` above all — a clone left at the False default would never
    reset, so each eval lane would stick in a terminated env after its first game over
    and returns would collapse to ~0."""
    env = _make_multi(num_envs=3, max_steps=27_000, autoreset_in_step=True)
    clone = env.clone()

    assert clone is not env
    assert clone.envs is not env.envs
    assert len(clone.envs) == 3
    assert not set(map(id, clone.envs)) & set(map(id, env.envs))
    assert clone._num_envs == env._num_envs
    assert clone._max_steps == env._max_steps
    assert clone._autoreset_in_step is True
    assert clone.make_fn is env.make_fn


def test_clone_keeps_max_steps_none_for_spec_fallback():
    """DMC passes no max_steps; the clone must not bake in the resolved spec value."""
    env = _make_multi(max_steps=None, spec_max=500)
    clone = env.clone()
    assert clone._max_steps is None
    assert clone.default_params.max_steps_in_episode == 500


# --- default_params fallbacks ----------------------------------------------


def test_explicit_max_steps_wins_over_spec():
    env = _make_multi(max_steps=27_000, spec_max=1_000)
    assert env.default_params.max_steps_in_episode == 27_000


def test_max_steps_zero_is_honoured():
    """`self._max_steps or ...` silently swallowed 0 and fell through to 1000."""
    env = _make_multi(max_steps=0, spec_max=None)
    assert env.default_params.max_steps_in_episode == 0


def test_spec_none_falls_back_to_1000():
    """ALE-v5 has the attribute present with value None — must reach the 1000 default."""
    env = _make_multi(max_steps=None, spec_max=None)
    assert env.default_params.max_steps_in_episode == 1_000


# --- real Atari env --------------------------------------------------------


@pytest.mark.slow
def test_atari_clone_is_independent():
    """End-to-end guard on the real adapter: the clone must be steppable in its own
    right (autoreset_in_step propagated) and hold different ALE instances."""
    pytest.importorskip("ale_py")
    from nexrl.compat.gymnasium2gymnax import create_atari

    env, _ = create_atari("Breakout-v5", num_envs=2)
    clone = env.clone()

    assert clone.envs is not env.envs
    assert clone._num_envs == 2
    assert clone._max_steps == 27_000  # 108000 ALE frames / frame_skip 4
    assert clone._autoreset_in_step is True
    assert clone.default_params.max_steps_in_episode == 27_000


@pytest.mark.slow
def test_atari_clone_survives_game_over():
    """The `autoreset_in_step` trap, checked functionally rather than by attribute.

    `create_atari` builds no gymnasium `Autoreset` wrapper, so a clone constructed at
    the `False` default would never reset: after the first game over its instance stays
    terminated, every subsequent step is invalid, and eval returns collapse to ~0. Play
    one full game on a *cloned* env, then keep stepping past the game over — the adapter
    must have reset in-step and keep producing live transitions.

    Driven through `_step_multi_instance` directly (plain Python, no JAX) so the check
    is fast and targets exactly the flag that must be propagated.
    """
    pytest.importorskip("ale_py")
    from nexrl.compat.gymnasium2gymnax import _reset_multi_instance, _step_multi_instance, create_atari

    env, _ = create_atari("Breakout-v5", num_envs=1)
    clone = env.clone()
    rng = np.random.default_rng(0)
    n_actions = clone.envs[0].action_space.n

    def step():
        return _step_multi_instance(clone.envs, 0, rng.integers(n_actions), autoreset_in_step=clone._autoreset_in_step)

    _reset_multi_instance(clone.envs, 0, 0)

    for _ in range(20_000):
        _, _, _, _, info = step()
        if info["real_done"]:
            break
    else:
        pytest.fail("no game over within 20000 steps — cannot exercise the reset path")

    # Post-game-over: the instance must be live again. A never-reset ALE env returns a
    # frozen terminal frame and immediately re-signals done on every step.
    dones = []
    for _ in range(50):
        _, _, _, done, _ = step()
        dones.append(bool(done))
    assert not any(dones), "clone did not auto-reset after game over — autoreset_in_step was lost"
