import jax
import pytest
from flax import nnx
from jax import numpy as jnp

from nexrl import FlashSACConfig, train_flash_sac
from nexrl.networks.flash_sac import FlashSACCritic, FlashSACPolicy, project_params

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


# Reference algorithm defaults, scaled down to the analytic test envs
ARGS = {
    "env": "Pendulum-v1",  # overridden by the env= argument
    "obs_key": None,
    "policy_obs_dim": None,
    "num_envs": 4,
    "buffer_size": 8192,
    "learning_starts": 256,
    "batch_size": 128,
    "actor_hidden_dim": 32,
    "actor_num_blocks": 1,
    "critic_hidden_dim": 32,
    "critic_num_blocks": 1,
    "total_timesteps": 8192,
    "eval_freq": 8192,
    "skip_initial_evaluation": True,
}

# FlashSAC rescales rewards at sample time (by max(std of returns, G_max / 5)), so Q values are
# in scaled units. Tests whose assertion is an absolute Q value (env1, env3) turn the scaling
# off; the others assert on actions and on Q relative to other states/actions.
RAW_REWARDS = {"normalize_rewards": False}


def _train(env, **overrides):
    config = FlashSACConfig(**{**ARGS, **overrides})
    return train_flash_sac(config, jax.random.PRNGKey(0), env=env)


def _q_fn(state):
    # Policy-space actions in [-1, 1]; min over the ensemble, as in the FlashSAC target
    def q_fn(obs, actions):
        return state["critic"].q_values(obs, actions, train=False).min(axis=0)

    return q_fn


def _make_act(state):
    from nexrl.algos.utils import make_eval_act

    model = state["actor"]
    return make_eval_act(lambda obs, rng: model.act(obs, rng), FlashSACConfig(**ARGS))


def test_env1():
    state, _ = _train(TestEnv1Continuous(), **RAW_REWARDS)
    act = _make_act(state)

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    obs = jnp.zeros((10, 1))
    actions = jnp.expand_dims(jax.vmap(act)(obs, rngs), 1)

    for v in _q_fn(state)(obs, actions):
        assert v == pytest.approx(1.0, abs=0.1)


def test_env2():
    # The reward is the observation whatever the action is, so scaled Q is proportional to obs
    state, _ = _train(TestEnv2Continuous())
    act = _make_act(state)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs_B1 = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    actions = jnp.expand_dims(jax.vmap(act)(obs_B1, rngs), 1)

    obs_B = obs_B1.squeeze(-1)
    q_B = _q_fn(state)(obs_B1, actions)
    assert jnp.corrcoef(q_B, obs_B)[0, 1] > 0.95
    clear = jnp.abs(obs_B) > 0.2
    assert jnp.all(jnp.sign(q_B[clear]) == jnp.sign(obs_B[clear]))


# The critic's nnx.vmap'd ensemble cannot run inside a jax.vmap that captures the module, so
# the tests below evaluate it on a batch (the eval action is deterministic, rng unused).


def test_env3():
    state, _ = _train(TestEnv3Continuous(), **RAW_REWARDS)
    act = _make_act(state)
    q_fn = _q_fn(state)
    gamma = FlashSACConfig(**ARGS).gamma

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    for obs in jnp.array([[-1], [1]]):
        r = 1 * gamma if obs == -1 else 1
        obs_B1 = jnp.broadcast_to(obs, (10, 1))
        actions = jnp.expand_dims(jax.vmap(act)(obs_B1, rngs), 1)
        for v in q_fn(obs_B1, actions):
            assert v == pytest.approx(r, abs=0.1)


def test_env4():
    # Env action range is [-2, 2] and reward = env action, so scaled Q is increasing in the
    # policy action and the policy should move to the positive side.
    # The observation is constant, so every BatchNorm sees zero variance: its train-mode output is
    # exactly its bias, while its eval-mode output divides the running-mean lag by sqrt(1e-5) and
    # is noise (probe: eval action anywhere in [-0.9, 0.9] across seeds, train-mode action ~+0.9).
    # The assertion therefore uses train-mode samples, the actions the policy loss optimizes.
    # Full fp32 matmuls: with the GPU default (TF32 on A100) the Q/action correlation lands at
    # 0.935-0.938 instead of above 0.95; CPU and "highest" both pass.
    with jax.default_matmul_precision("highest"):
        state, _ = _train(TestEnv4Continuous())

        num_rngs = 100
        obs_B1 = jnp.zeros((num_rngs, 1))
        aa, _ = state["actor"](obs_B1, jax.random.PRNGKey(1), train=True)
        assert sum(aa.squeeze(-1) > 0.0) >= 0.9 * num_rngs

        grid_B1 = jnp.linspace(-1.0, 1.0, num_rngs)[:, None]
        q_B = _q_fn(state)(obs_B1, grid_B1)
        assert jnp.corrcoef(q_B, grid_B1.squeeze(-1))[0, 1] > 0.95


def test_env5():
    # Reward is -|action - obs|: Q is higher at the matching action than far from it, and the
    # policy outputs the observation. At 8192 steps the action gap sits at the 0.15 tolerance
    # (seeds 0-4 give 0.10-0.20 on main); 16384 steps give 0.07-0.10.
    state, _ = _train(TestEnv5Continuous(), total_timesteps=16_384, eval_freq=16_384)

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    q_fn = _q_fn(state)
    assert jnp.all(q_fn(obs, obs) > q_fn(obs, -jnp.sign(obs)))

    act = _make_act(state)
    vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
    num_rngs = 100
    actions = vmap_act(obs, jax.random.split(rng, num_rngs))

    for i in range(obs.shape[0]):
        assert float(actions[:, i].mean()) == pytest.approx(float(obs[i].squeeze()), abs=0.15)
        assert jnp.isclose(actions[:, i], obs[i], atol=0.5).sum() >= 0.9 * num_rngs


def test_project_params():
    obs_dim, action_dim, hidden, blocks = 5, 2, 16, 2
    rngs = nnx.Rngs(0)
    critic = FlashSACCritic(obs_dim, action_dim, 11, -5.0, 5.0, hidden, blocks, rngs)
    policy = FlashSACPolicy(obs_dim, action_dim, hidden, blocks, rngs)

    def perturb(model, key):
        # Random positive rescale of every parameter, so no norm constraint holds before projecting
        params = nnx.state(model, nnx.Param)
        leaves, treedef = jax.tree.flatten(params)
        keys = jax.random.split(key, len(leaves))
        leaves = [x * jax.random.uniform(k, x.shape, minval=0.5, maxval=3.0) + 0.1 for x, k in zip(leaves, keys)]
        nnx.update(model, jax.tree.unflatten(treedef, leaves))

    for i, model in enumerate((critic, policy)):
        perturb(model, jax.random.PRNGKey(i))
        project_params(model)

        num_linear = num_rms = num_bn = 0
        for _, module in nnx.iter_modules(model):
            if isinstance(module, nnx.Linear):
                norms = jnp.linalg.norm(module.kernel[...], axis=-2)
                assert jnp.allclose(norms, 1.0, atol=1e-5)
                num_linear += 1
            elif isinstance(module, nnx.RMSNorm):
                d = module.scale[...].shape[-1]
                assert jnp.allclose(jnp.linalg.norm(module.scale[...], axis=-1), d**0.5, rtol=1e-5)
                num_rms += 1
            elif isinstance(module, nnx.BatchNorm):
                both = jnp.concatenate([module.scale[...], module.bias[...]], axis=-1)
                d = module.scale[...].shape[-1]
                assert jnp.allclose(jnp.linalg.norm(both, axis=-1), d**0.5, rtol=1e-5)
                num_bn += 1
        assert num_linear > 0 and num_rms > 0 and num_bn > 0
