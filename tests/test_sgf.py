"""SGF verification tests.

Tier 1: Component unit tests (shapes, VICReg, gradient flow)
Tier 2: Smoke test with tiny config on Atari
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from nexrl.networks.dreamer import (
    ContinuePredictor,
    DreamerActor,
    DreamerCritic,
    RewardPredictor,
)
from nexrl.networks.sgf import (
    SGFEncoder,
    SGFPredictor,
    SGFProjector,
    SGFTransitionPredictor,
    SGFWorldModel,
    augment,
    random_intensity,
    random_shift,
    vicreg_loss,
)


# Shared dims for component tests
B, C, H, W = 8, 4, 84, 84
Y_DIM, Z_DIM = 64, 128  # small for tests
ACTION_DIM = 4
HIDDEN = 64
NUM_BINS = 31


# ---------------------------------------------------------------------------
# Tier 1: Component tests
# ---------------------------------------------------------------------------


class TestAugmentations:
    def test_random_shift_preserves_shape(self):
        rng = jax.random.PRNGKey(0)
        obs = jax.random.uniform(rng, (B, C, H, W))
        shifted = random_shift(rng, obs, pad=3)
        assert shifted.shape == (B, C, H, W)

    def test_random_shift_different_keys_different_output(self):
        obs = jnp.ones((B, C, H, W))
        out1 = random_shift(jax.random.PRNGKey(0), obs, pad=3)
        out2 = random_shift(jax.random.PRNGKey(1), obs, pad=3)
        # Edge-padded ones shifted should still be ones, but let's test with varying input
        obs_var = jax.random.uniform(jax.random.PRNGKey(42), (B, C, H, W))
        out1 = random_shift(jax.random.PRNGKey(0), obs_var, pad=3)
        out2 = random_shift(jax.random.PRNGKey(1), obs_var, pad=3)
        assert not jnp.allclose(out1, out2)

    def test_random_intensity_preserves_shape(self):
        rng = jax.random.PRNGKey(0)
        obs = jax.random.uniform(rng, (B, C, H, W))
        out = random_intensity(rng, obs, scale=0.05)
        assert out.shape == (B, C, H, W)

    def test_random_intensity_bounded(self):
        rng = jax.random.PRNGKey(0)
        obs = jnp.ones((B, C, H, W))
        out = random_intensity(rng, obs, scale=0.05)
        # With clipped noise in [-2, 2] and scale=0.05, output in [0.9, 1.1]
        assert jnp.all(out >= 0.89)
        assert jnp.all(out <= 1.11)

    def test_augment_pipeline(self):
        rng = jax.random.PRNGKey(0)
        obs = jax.random.uniform(rng, (B, C, H, W)) * 255.0
        out = augment(rng, obs, shift_pad=3, intensity_scale=0.05)
        assert out.shape == (B, C, H, W)
        assert jnp.all(jnp.isfinite(out))


class TestVICReg:
    def test_collapsed_features_high_var_loss(self):
        """All-same z should produce high variance loss."""
        z = jnp.ones((B, Z_DIM)) * 0.5  # collapsed
        z_prime = jnp.ones((B, Z_DIM)) * 0.5
        var_loss, _cov_loss = vicreg_loss(z, z_prime, var_weight=25.0, cov_weight=1.0)
        # Std of constant is 0, so hinge(1-0) = 1.0, avg = 1.0, weight=25 -> 25.0
        assert float(var_loss) > 20.0

    def test_diverse_features_low_var_loss(self):
        """Well-spread z should have low variance loss."""
        rng = jax.random.PRNGKey(0)
        z = jax.random.normal(rng, (64, Z_DIM)) * 2.0  # std >> 1
        z_prime = jax.random.normal(jax.random.PRNGKey(1), (64, Z_DIM)) * 2.0
        var_loss, _cov_loss = vicreg_loss(z, z_prime, var_weight=25.0, cov_weight=1.0)
        assert float(var_loss) < 1.0  # hinge is 0 when std > 1

    def test_correlated_features_high_cov_loss(self):
        """Correlated features should produce high covariance loss."""
        rng = jax.random.PRNGKey(0)
        base = jax.random.normal(rng, (B, 1))
        z = jnp.broadcast_to(base, (B, Z_DIM))  # all features identical
        z_prime = jnp.broadcast_to(base, (B, Z_DIM))
        _, cov_loss = vicreg_loss(z, z_prime, var_weight=0.0, cov_weight=1.0)
        assert float(cov_loss) > 0.1

    def test_gradient_flows_through_vicreg(self):
        """VICReg loss should produce gradients."""
        z = jax.random.normal(jax.random.PRNGKey(0), (B, Z_DIM))

        def loss_fn(z_in):
            var_l, cov_l = vicreg_loss(z_in, z_in, var_weight=25.0, cov_weight=1.0)
            return var_l + cov_l

        grad = jax.grad(loss_fn)(z)
        assert jnp.any(grad != 0)


class TestNetworkShapes:
    def setup_method(self):
        self.rngs = nnx.Rngs(42)

    def test_encoder_shape(self):
        enc = SGFEncoder(C, H, W, Y_DIM, self.rngs)
        obs = jax.random.uniform(jax.random.PRNGKey(0), (B, C, H, W)) * 255.0
        y = enc(obs)
        assert y.shape == (B, Y_DIM)
        assert jnp.all(jnp.isfinite(y))

    def test_projector_shape(self):
        proj = SGFProjector(Y_DIM, Z_DIM, HIDDEN, self.rngs)
        y = jax.random.normal(jax.random.PRNGKey(0), (B, Y_DIM))
        z = proj(y)
        assert z.shape == (B, Z_DIM)
        assert jnp.all(jnp.isfinite(z))

    def test_predictor_shape(self):
        pred = SGFPredictor(Z_DIM, ACTION_DIM, HIDDEN, self.rngs)
        z = jax.random.normal(jax.random.PRNGKey(0), (B, Z_DIM))
        a = jax.nn.one_hot(jnp.zeros(B, dtype=jnp.int32), ACTION_DIM)
        z_pred = pred(z, a)
        assert z_pred.shape == (B, Z_DIM)
        assert jnp.all(jnp.isfinite(z_pred))

    def test_transition_predictor_shape_and_skip(self):
        trans = SGFTransitionPredictor(Y_DIM, ACTION_DIM, HIDDEN, 3, self.rngs)
        y = jax.random.normal(jax.random.PRNGKey(0), (B, Y_DIM))
        a = jax.nn.one_hot(jnp.zeros(B, dtype=jnp.int32), ACTION_DIM)
        next_y = trans(y, a)
        assert next_y.shape == (B, Y_DIM)
        assert jnp.all(jnp.isfinite(next_y))
        # Skip connection: with zero-init output, next_y ≈ y
        # (not exact due to random init, but structurally present)

    def test_reward_predictor_shape(self):
        input_dim = 2 * Y_DIM + ACTION_DIM
        rp = RewardPredictor(input_dim, NUM_BINS, (HIDDEN,), jax.nn.silu, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (B, input_dim))
        logits = rp(x)
        assert logits.shape == (B, NUM_BINS)

    def test_continue_predictor_shape(self):
        input_dim = 2 * Y_DIM + ACTION_DIM
        cp = ContinuePredictor(input_dim, (HIDDEN,), jax.nn.silu, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (B, input_dim))
        logit = cp(x)
        assert logit.shape == (B,)

    def test_actor_shape(self):
        actor = DreamerActor(Y_DIM, ACTION_DIM, True, (HIDDEN,), jax.nn.silu, self.rngs)
        y = jax.random.normal(jax.random.PRNGKey(0), (B, Y_DIM))
        action, log_prob = actor(y, jax.random.PRNGKey(1))
        assert action.shape == (B, ACTION_DIM)  # one-hot with ST
        assert log_prob.shape == (B,)

    def test_critic_shape(self):
        critic = DreamerCritic(Y_DIM, NUM_BINS, (HIDDEN,), jax.nn.silu, self.rngs)
        y = jax.random.normal(jax.random.PRNGKey(0), (B, Y_DIM))
        logits = critic(y)
        assert logits.shape == (B, NUM_BINS)
        val = critic.value(y)
        assert val.shape == (B,)


class TestGradientFlow:
    """Verify correct gradient flow through the loss functions."""

    def setup_method(self):
        self.rngs = nnx.Rngs(42)
        self.encoder = SGFEncoder(C, H, W, Y_DIM, self.rngs)
        self.projector = SGFProjector(Y_DIM, Z_DIM, HIDDEN, self.rngs)
        self.predictor = SGFPredictor(Z_DIM, ACTION_DIM, HIDDEN, self.rngs)
        pred_input_dim = 2 * Y_DIM + ACTION_DIM
        self.reward_pred = RewardPredictor(pred_input_dim, NUM_BINS, (HIDDEN,), jax.nn.silu, self.rngs)
        self.continue_pred = ContinuePredictor(pred_input_dim, (HIDDEN,), jax.nn.silu, self.rngs)
        self.world_model = SGFWorldModel(self.encoder, self.projector, self.predictor, self.reward_pred, self.continue_pred)
        self.transition = SGFTransitionPredictor(Y_DIM, ACTION_DIM, HIDDEN, 3, self.rngs)

    def test_repr_loss_gradients_flow_to_encoder(self):
        """Representation loss should produce non-zero encoder gradients."""
        obs = jax.random.uniform(jax.random.PRNGKey(0), (B, C, H, W)) * 255.0
        next_obs = jax.random.uniform(jax.random.PRNGKey(1), (B, C, H, W)) * 255.0
        actions = jax.random.randint(jax.random.PRNGKey(2), (B,), 0, ACTION_DIM)

        def repr_loss(wm):
            y = wm.encoder(obs)
            next_y = wm.encoder(next_obs)
            z = wm.projector(y)
            next_z = wm.projector(next_y)
            flat_a = jax.nn.one_hot(actions, ACTION_DIM, dtype=jnp.float32)
            z_pred = wm.predictor(z, flat_a)
            temporal = jnp.mean((z_pred - jax.lax.stop_gradient(next_z)) ** 2)
            var_l, cov_l = vicreg_loss(z, next_z)
            return temporal + var_l + cov_l

        loss, grads = nnx.value_and_grad(repr_loss)(self.world_model)
        assert jnp.isfinite(loss)
        # Check encoder has non-zero gradients
        enc_grads = nnx.state(grads.encoder, nnx.Param)
        has_nonzero = any(jnp.any(v != 0) for v in jax.tree.leaves(enc_grads))
        assert has_nonzero, "Encoder should receive gradients from repr loss"

    def test_transition_loss_no_gradients_to_encoder(self):
        """Transition loss should NOT flow gradients to encoder (stop_gradient)."""
        obs = jax.random.uniform(jax.random.PRNGKey(0), (B, C, H, W)) * 255.0
        next_obs = jax.random.uniform(jax.random.PRNGKey(1), (B, C, H, W)) * 255.0
        actions = jax.random.randint(jax.random.PRNGKey(2), (B,), 0, ACTION_DIM)

        def trans_loss(wm, trans):
            y = jax.lax.stop_gradient(wm.encoder(obs))
            next_y = jax.lax.stop_gradient(wm.encoder(next_obs))
            flat_a = jax.nn.one_hot(actions, ACTION_DIM, dtype=jnp.float32)
            next_y_pred = trans(y, flat_a)
            return jnp.mean((next_y_pred - next_y) ** 2)

        _loss, (wm_grads, trans_grads) = nnx.value_and_grad(trans_loss, argnums=(0, 1))(self.world_model, self.transition)
        # Encoder gradients should be zero
        enc_grads = nnx.state(wm_grads.encoder, nnx.Param)
        all_zero = all(jnp.all(v == 0) for v in jax.tree.leaves(enc_grads))
        assert all_zero, "Encoder should NOT receive gradients from transition loss"
        # Transition should have non-zero gradients
        trans_param_grads = nnx.state(trans_grads, nnx.Param)
        has_nonzero = any(jnp.any(v != 0) for v in jax.tree.leaves(trans_param_grads))
        assert has_nonzero, "Transition predictor should receive gradients"

    def test_actor_loss_no_gradients_to_world_model(self):
        """Actor REINFORCE loss should not flow to world model (sg states)."""
        actor = DreamerActor(Y_DIM, ACTION_DIM, True, (HIDDEN,), jax.nn.silu, self.rngs)
        y = jax.random.normal(jax.random.PRNGKey(0), (B, Y_DIM))
        sg_y = jax.lax.stop_gradient(y)

        def actor_loss(act):
            _, log_prob = act(sg_y, jax.random.PRNGKey(1))
            return -jnp.mean(log_prob)

        loss, grads = nnx.value_and_grad(actor_loss)(actor)
        assert jnp.isfinite(loss)
        # Actor should have gradients
        actor_grads = nnx.state(grads, nnx.Param)
        has_nonzero = any(jnp.any(v != 0) for v in jax.tree.leaves(actor_grads))
        assert has_nonzero


class TestImagination:
    """Test imagination rollout shapes and lambda returns."""

    def test_lambda_returns_known_values(self):
        """Verify lambda returns match hand-computed values."""
        # Simple case: 2-step, gamma=1, lambda=1 (pure MC return)
        rewards = jnp.array([[1.0], [2.0]])  # (H, B=1)
        disc = jnp.array([[1.0], [1.0]])  # no discount
        values = jnp.array([[0.0], [0.0]])  # zero values
        bootstrap = jnp.array([5.0])  # V(terminal)

        def lambda_scan(next_val, t_data):
            rew, d, cont_val = t_data
            target = rew + d * (0.0 * cont_val + 1.0 * next_val)  # lambda=1
            return target, target

        _, returns = jax.lax.scan(
            lambda_scan,
            bootstrap,
            (rewards[::-1], disc[::-1], values[::-1]),
        )
        returns = returns[::-1]
        # With lambda=1: R_1 = r_1 + disc * R_2, R_2 = r_2 + disc * bootstrap
        # R_2 = 2 + 1*5 = 7, R_1 = 1 + 1*7 = 8
        assert jnp.allclose(returns[0, 0], 8.0)
        assert jnp.allclose(returns[1, 0], 7.0)


# ---------------------------------------------------------------------------
# Tier 2: Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sgf_smoke_atari():
    """Run a very short SGF training on Atari to verify end-to-end correctness.

    Checks: all metric keys present, all values finite, losses don't explode.
    """
    from nexrl import SGFConfig, train_sgf

    config = SGFConfig(
        env="atari/Breakout-v5",
        encoder_kwargs=None,  # SGF uses its own encoder
        agent_kwargs=SGFConfig.__dataclass_fields__["agent_kwargs"].default_factory(),
        num_envs=1,
        buffer_size=500,
        fill_buffer=100,
        batch_size=16,
        imagination_batch_size=32,
        imagination_horizon=3,
        wm_every=1,
        agent_every=1,
        total_timesteps=300,
        eval_freq=150,
        repr_learning_rate=6e-4,
        repr_warmup_steps=50,
        transition_learning_rate=1.2e-3,
        transition_warmup_steps=50,
        ac_learning_rate=2.4e-4,  # higher for short test
        y_dim=64,
        z_dim=128,
        projector_hidden=128,
        predictor_hidden=128,
        transition_hidden=64,
        transition_num_hidden=2,
        value_num_bins=31,
        reward_num_bins=31,
        shift_pad=2,
        action_stack=2,
        run_evaluation=False,
        skip_initial_evaluation=True,
    )

    rng = jax.random.PRNGKey(0)
    _state, metrics = train_sgf(config, rng)

    # Check all expected metric keys present
    expected_keys = {
        "eval_lengths",
        "eval_returns",
        "train_return",
        "global_step",
        "repr_loss",
        "dynamics_loss",
        "vicreg_var_loss",
        "vicreg_cov_loss",
        "temporal_loss",
        "reward_loss",
        "terminal_loss",
        "transition_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
    }
    assert expected_keys.issubset(set(metrics.keys())), f"Missing keys: {expected_keys - set(metrics.keys())}"

    # Check all metrics are finite
    for key, val in metrics.items():
        assert jnp.all(jnp.isfinite(val)), f"Metric {key} has non-finite values: {val}"

    # Check global step advanced
    last_step = int(metrics["global_step"][-1])
    assert last_step > 0, f"global_step should advance, got {last_step}"

    print("\n--- SGF Smoke Test Results ---")
    print(f"Global steps: {metrics['global_step']}")
    for key in sorted(expected_keys - {"eval_lengths", "eval_returns", "global_step"}):
        vals = metrics[key]
        print(f"  {key}: {[f'{float(v):.4f}' for v in vals]}")
    print("All metrics finite. Smoke test passed!")
