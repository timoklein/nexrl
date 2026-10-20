import jax
import pytest
from jax import numpy as jnp

from nexrl import REPPOConfig, train_reppo
from nexrl.algos.utils import normalize_obs

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


# Reference algorithm defaults (learning rate, KL bound, coefficients, gamma, lambda, 151 bins on
# [-100, 100]), scaled down to the analytic test envs: small nets, a small rollout, few updates.
ARGS = {
    "env": "Pendulum-v1",  # overridden by the env= argument
    "obs_key": None,
    "policy_obs_dim": None,
    "num_envs": 32,
    "num_steps": 32,
    "num_epochs": 4,
    "num_minibatches": 8,
    "actor_hidden_dim": 32,
    "critic_hidden_dim": 32,
    "total_timesteps": 65_536,
    "eval_freq": 65_536,
    "skip_initial_evaluation": True,
}


def _train(env):
    config = REPPOConfig(**ARGS)
    return train_reppo(config, jax.random.PRNGKey(0), env=env)


# The critic and the policy see observations normalized by the running RMS of the training run, so
# the tests normalize their probe observations the same way. Actions given to the critic are in
# policy space [-1, 1]; the env sees low + 0.5 * (a + 1) * (high - low).


def _normalize(state, obs):
    return normalize_obs(state["obs_rms_state"], obs)


def _q_fn(state):
    def q_fn(obs, actions):
        return state["critic"].value(_normalize(state, obs), actions)

    return q_fn


def _act_fn(state):
    # Deterministic policy-space action tanh(loc) for a batch of raw observations
    def act(obs):
        return state["actor"].act(_normalize(state, obs), None)

    return act


def test_env1():
    # The reward is +1 and the episode ends after one step, so Q is 1 (plus a small entropy bonus)
    state, _ = _train(TestEnv1Continuous())

    obs = jnp.zeros((10, 1))
    for v in _q_fn(state)(obs, _act_fn(state)(obs)):
        assert v == pytest.approx(1.0, abs=0.1)


def test_env2():
    # The reward is the observation whatever the action is, so Q is the observation
    state, _ = _train(TestEnv2Continuous())

    rng = jax.random.PRNGKey(0)
    obs_B1 = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    q_B = _q_fn(state)(obs_B1, _act_fn(state)(obs_B1))

    obs_B = obs_B1.squeeze(-1)
    assert jnp.corrcoef(q_B, obs_B)[0, 1] > 0.95
    clear = jnp.abs(obs_B) > 0.2
    assert jnp.all(jnp.sign(q_B[clear]) == jnp.sign(obs_B[clear]))
    for q, o in zip(q_B, obs_B):
        assert float(q) == pytest.approx(float(o), abs=0.1)


def test_env3():
    # The reward is 0 on the first step and 1 on the second, which ends the episode
    state, _ = _train(TestEnv3Continuous())
    gamma = REPPOConfig(**ARGS).gamma

    q_fn, act = _q_fn(state), _act_fn(state)
    for obs, r in zip((-1.0, 1.0), (gamma, 1.0)):
        obs_B1 = jnp.full((10, 1), obs)
        for v in q_fn(obs_B1, act(obs_B1)):
            assert v == pytest.approx(r, abs=0.1)


def test_env4():
    # Env action range is [-2, 2] and reward = env action, so the best policy action is +1 (env
    # action 2) and Q is increasing in the policy action.
    # REPPO is on-policy: the policy collapses onto actions near +1 early (about 0.9-1.0 at the end)
    # and the critic only sees those. On a grid over the whole [-1, 1] the critic is extrapolating
    # (probe seeds 0-4: corr(Q, a) 0.08 to 0.79, Q(-1) 0.02 to 1.09, true value -2), so Q is checked
    # on actions sampled from the trained policy (corr 0.997-0.999 on seeds 0-4).
    state, _ = _train(TestEnv4Continuous())
    policy = state["actor"]

    num_samples = 1000
    obs_B1 = jnp.zeros((num_samples, 1))
    obs_norm_B1 = _normalize(state, obs_B1)

    best_env_action = 2.0
    det_env_action = -2.0 + 0.5 * (_act_fn(state)(obs_B1[:1]) + 1.0) * 4.0
    assert float(det_env_action.squeeze()) == pytest.approx(best_env_action, abs=0.1)

    loc_B1, log_std_B1 = policy(obs_norm_B1)
    action_B1, _ = policy.sample_and_log_prob(loc_B1, log_std_B1, jax.random.PRNGKey(1))
    assert jnp.mean(action_B1 > 0.0) >= 0.9

    q_B = _q_fn(state)(obs_B1, action_B1)
    assert jnp.corrcoef(q_B, action_B1.squeeze(-1))[0, 1] > 0.95


def test_env5():
    # Reward is -|action - obs|: Q is higher at the matching action than far from it, and the
    # policy outputs the observation.
    state, _ = _train(TestEnv5Continuous())

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    q_fn = _q_fn(state)
    assert jnp.all(q_fn(obs, obs) > q_fn(obs, -jnp.sign(obs)))

    actions = _act_fn(state)(obs)
    for o, a in zip(obs.squeeze(-1), actions.squeeze(-1)):
        assert float(a) == pytest.approx(float(o), abs=0.2)
