# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install (editable)
uv sync                           # Core only
uv sync --extra compat            # With environment adapters (incl. Atari)
uv sync --extra compat --extra dmc # Also with DeepMind Control (requires labmaze/Bazel)

# Lint, format, type check
uv run ruff check .               # Check for issues
uv run ruff format .              # Auto-format
uv run pyright src/nexrl/         # Type check (0 errors expected)

# Test
uv run pytest tests/              # Run all tests
uv run pytest tests/test_ppo.py   # Single test file

# Train
uv run python train.py ppo --config configs/ppo/gymnax/cartpole.yaml
uv run python train.py ppo --config configs/ppo/gymnax/cartpole.yaml --learning-rate 0.001
uv run python train.py ppo --env Pendulum-v1 --agent-kwargs.activation tanh
uv run python train.py sac_simba --config configs/sac_simba/gymnax/pendulum.yaml
uv run python train.py fast_sac --config configs/fast_sac/playground/g1_joystick_flat_terrain.yaml
uv run python train.py dqn --config configs/dqn/atari/breakout.yaml
uv run python train.py drq --config configs/drq/atari/breakout.yaml
uv run python train.py ppo --config configs/ppo/atari/breakout.yaml
uv run python train.py ppo_rnn --config configs/ppo_rnn/gymnax/cartpole.yaml
uv run python train.py sac --config configs/sac/dm_control/cheetah_run.yaml
uv run python train.py ppo --config configs/ppo/playground/cartpole_balance.yaml
uv run python train.py sac --config configs/sac/playground/reacher_easy.yaml

# Weights & Biases logging (post-training upload, zero overhead during training)
uv run python train.py ppo --config configs/ppo/gymnax/cartpole.yaml --track
uv run python train.py ppo --config configs/ppo/gymnax/cartpole.yaml --track --wandb-project my-project --wandb-entity my-team

# GPU selection — set CUDA_VISIBLE_DEVICES before the command, no --gpu flag
CUDA_VISIBLE_DEVICES=2 uv run python train.py ppo --config configs/ppo/gymnax/cartpole.yaml
```

## Throughput ledger

| machine (GPU) | benchmark | settings | SPS | peak VRAM | packing | date |
|---|---|---|---|---|---|---|
| this box (A100-PCIE-40GB) | DrQ Atari-100k Breakout | RR=1, num_envs=1, eval_freq 10k (10x128-ep evals) | 52 eff. | 8.8 GB | 3/GPU, near-linear (CPU-bound on ALE) | 2026-08-20 |
| this box (A100-PCIE-40GB) | DrQ Atari-100k Breakout | reference net (SAME/7744), eval_freq 100k (1x400-ep final eval) | ~97 eff. (17.3 min/run) | — | 3/GPU | 2026-08-21 |

Notes: effective SPS includes eval overhead — training alone is ~350 SPS (10k steps + compile in 58s);
the ten 128-episode eval points are ~85% of wall-clock and lengthen as the agent improves. `nvidia-smi`
shows JAX's preallocated ~30 GB, not the 8.8 GB working set — measure with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Atari runs must be one seed per process: the multi-instance
adapter aliases vmapped seed lanes (`--num-seeds > 1` corrupts data).

## Algorithm Verification

When implementing or modifying algorithms, verify correctness using benchmark environments:

- **Discrete**: `CartPole-v1`, threshold >480 (DQN, IQN, PQN, Rainbow, PPO, PPO-RNN, PPG, PPO-ICM, PPO-E3B, PPO-RND, V-MPO, MPO, AlphaZero, MuZero, Dreamer, MrQ, MRS.Q, SGF)
- **Continuous**: `Pendulum-v1`, threshold >-200 (SAC, SAC-SimBa, FastSAC, TD3, PPO, PPO-RND, V-MPO-cont, MPO-cont, Dreamer, MrQ, MRS.Q)
- **Visual**: `dm_control/cartpole-balance-v0` (pixels), >900 within 200k steps (DrQ-v2)
- **Visual Atari-100k**: `atari/Breakout-v5`, 100k env steps (DrQ, BBF); BBF target 370.6 (paper, 50 seeds); DrQ target 16.1 (paper Table 5, 5 seeds) — reproduced at 16.75 ± 0.66 (n=6, commit c4cc0e7; 19.38 with --noop-max 0)

## Architecture

Pure JAX RL library — `jax.jit`/`jax.vmap`/`jax.pmap` on entire training loops.

**Standalone train functions**: Each algorithm is a single `train_algo(config, rng)` function. No classes, no mixins, no inheritance.

```python
from nexrl import get_train_fn, PPOConfig
train_fn = get_train_fn("ppo")
config = PPOConfig.from_yaml("configs/ppo/gymnax/cartpole.yaml")
state, metrics = jax.jit(jax.vmap(train_fn, in_axes=(None, 0)), static_argnums=(0,))(config, keys)
```

**Key modules**: `algos/` (standalone `train_*` functions), `algos/utils.py` (env creation, GAE, normalization, target updates, eval), `networks/` (Flax NNX networks, re-exported from `__init__.py`), `configs.py` (typed dataclasses + `ALGO_CONFIG_MAP`), `types.py` (shared type aliases), `buffers.py` (JAX-native `ReplayBuffer`/`CircularBuffer`), `compat/` (environment adapters).

**NamedTuple carry**: Each algo defines a `*Carry` NamedTuple for non-module state (rng, env_state, replay buffer, normalization stats, training metrics accumulator).

**Shared GAE**: `calculate_gae(rewards_TB, values_TB, dones_TB, last_val_B, gamma, gae_lambda, episodic=True)` — single source of truth. `episodic=False` zeroes done masks for non-episodic intrinsic reward streams.

**Config system**: Typed dataclasses in `configs.py`, layout `configs/{algo}/{suite}/{env}.yaml`. CLI (`train.py`): two-pass — argparse extracts algorithm/config/flags, tyro provides typed per-algo config with YAML defaults and CLI overrides.

**Distrax distribution rule**: All policy classes create distrax distributions internally via `_action_dist()` (private) and return only plain arrays from public methods. Never return a distribution object — distributions must be created and consumed in the same scope.

### NNX Patterns and Enforcement Rules (MANDATORY)

Algorithms use Flax NNX (`flax.nnx`): direct calls `model(obs)`, `nnx.Optimizer(model, tx)`, `nnx.value_and_grad(loss_fn)(model)`.

**Before writing ANY loop/branch that touches an `nnx.Module` or `nnx.Optimizer`:**

1. STOP and check: Am I using the NNX lifted transform (`nnx.fori_loop` / `nnx.scan` / `nnx.cond` / `nnx.while_loop`)?
2. If I wrote `jax.lax.{scan,fori_loop,cond,while_loop}` with a module argument: **WRONG.** Replace with `nnx.*` equivalent.
3. If I wrote `nnx.split` / `nnx.merge` anywhere except the final return boundary: **WRONG.** Use an NNX lifted transform instead.

**Allowed `jax.lax.*` usage** (read-only captured modules):
- `jax.lax.scan` for trajectory collection where modules are only called, never mutated
- `jax.lax.cond` where no modules are involved

**Self-review anti-patterns**: `jax.lax.fori_loop`/`scan`/`cond` with module args; `nnx.split`/`nnx.merge` anywhere except final return; `jax.jit` instead of `nnx.jit` wrapping module calls.

### Training Metrics

All `train_*` return `(TrainState, Metrics)` where `Metrics = dict[str, jax.Array]`. Common keys: `eval_lengths`, `eval_returns`, `global_step`. Per-algo keys are in each algorithm's source.

**Real-time logging**: Pass `log_fn` via `functools.partial` before `jax.jit`/`jax.vmap`. Called via `jax.debug.callback` at each eval point.

**Implementation notes**:
- Metrics accumulate as running sums + count in the carry's `train_metrics` dict
- Off-policy algos only accumulate when `start_training=True` (after buffer fill)
- **Carry reset inside `nnx.scan` must use traced ops (`x * 0`) to preserve carry references — never Python literals**

### Shape Suffix Convention

Tensor variable names use uppercase dimension suffixes (e.g. `obs_BD`, `action_BA`, `q_target_CB`). Each file has a dimension key comment.

**Dimension codebook:** `B` batch, `D` obs dim, `A` action dim, `H` hidden, `T` trajectory len, `C` num critics, `K` quantile samples (IQN), `E` cosine embed (IQN), `L` ICM latent, `N` atoms (distributional), `P` PPG aux buffer, `S` latent state (MuZero/MrQ zs), `X` zsa (MrQ/MRS.Q), `Q` Q-ensemble (MRS.Q), `G` RSSM deter, `Z` RSSM stoch, `I` imagination horizon, `F` CNN features, `U` critic updates per env step (FastSAC).

**Rules:**
- Rename **local variables only** — never NamedTuple fields, function parameters, dict keys, or network method signatures
- Extract from structs with suffix: `obs_BD = mb.obs` when used in tensor ops
- Scalars after reduction get no suffix: `loss = loss_B.mean()`
- PPO dual-mode actions: use comment `# (B,) discrete or (B, A) continuous` instead of suffix
- Ruff N806/N803 suppressed for `networks/*.py` and `algos/*.py` in `pyproject.toml`

### Environment Compatibility

Namespaced strings: `CartPole-v1` (gymnax), `gymnasium/CartPole-v1`, `brax/halfcheetah`, `atari/Breakout-v5`, `dm_control/cheetah-run`, `playground/CheetahRun`. Adapters in `compat/` convert to gymnax interface.

**Key gotchas:**
- **`jax.ensure_compile_time_eval()`**: `compat/` wraps env creation in this. Third-party packages (Craftax) run `jnp` ops at import time — if first triggered inside JIT, those ops see tracers and crash.
- **Atari/DMC** use `MultiInstanceGymnasium2GymnaxEnv` — sequential `jax.pure_callback`, no vectorization. Atari obs: `(4, 84, 84)` uint8. DMC pixels: `(3, 84, 84)` uint8.
- **MuJoCo Playground** (`playground/`): JAX-native, fully jittable/vmappable. Uses `impl="jax"` (warp has CUDA issues on some GPUs). Episode truncation handled by wrapper. Vision limited (only CartpoleBalance as of 0.2.0) — use `dm_control/` for pixel training.
- **Craftax** uses optimistic reset via `create_vec_env_fns()` (pre-computes `num_envs // 16` resets per step).

**CNN encoders**: Set `encoder_kwargs=CNNConfig(encoder="nature"|"impala")`. Both handle NCHW→NHWC and uint8→float32/255 internally. Obs stay uint8 throughout storage — converted at forward-pass time. `FloatObsWrapper` skipped when `encoder_kwargs` is set. Each network gets its own encoder instance; Polyak updates include encoder params.

### Type Checking

Pyright `basic` mode on `src/nexrl/` (excluding `compat/` and tests). Use `jax.Array` over `chex.Array`. Third-party stub issues (`reportAttributeAccessIssue`, `reportReturnType`, `reportArgumentType`) set to `"warning"` in `pyproject.toml`.

## Algorithm-Specific Gotchas

These are non-obvious design decisions that differ from naive implementations. Read the relevant section before modifying an algorithm.

### Strided buffer sampling (MuZero, Dreamer, MrQ, MRS.Q)

`CircularBuffer.extend` interleaves transitions as `[env0_t0, env1_t0, ..., envN_t0, env0_t1, ...]`. Algorithms needing temporal sequences from the **same env** stride by `num_envs`: `index = env_id + (start_timestep + offset) * num_envs`.

### Dreamer

- **Bounded normal (continuous)**: Tanh on **mean only**, plain Gaussian sample (`mean + std * noise`). Unlike SAC's tanh-squashed Gaussian, this avoids tanh-Jacobian log-det correction which causes extreme log_prob variance near action boundaries and makes REINFORCE unstable.
- **REINFORCE stop-gradient**: Log_probs and entropy must be computed on `stop_gradient(states)` — otherwise indirect gradients flow through the entire imagination trajectory and corrupt the policy gradient. Pattern: collect trajectory, then recompute via `actor.log_prob_of(sg(states), sg(actions))`.
- **RSSM**: GRU update gate biased toward -1 (keeps old state). Uniform mixing (`unimix=0.01`) prevents categorical collapse.

### MrQ

- **Burst updates**: Every `target_update_freq` steps: hard-copy targets, recompute reward scale, run `target_update_freq` encoder updates. Value+policy update every step. Keeps encoder/target fixed during RL optimization.
- **Reward scaling**: `target = (1/r̄) * (G + γ^H * r̄' * min Q'_j)` where `r̄'` is reward scale at last target copy.
- **Gradient flow in policy loss**: Flows through `encoder.zsa()`: `policy(zs) → action → encoder.zsa(zs, action) → value(zsa) → loss`. Encoder is NOT stop-gradient'd (only `zs` is). Separate optimizers ensure only policy params update.
- **Action scaling**: Policy outputs [-1, 1]. Continuous scaled by `max_action`. Buffer stores normalized. Discrete noise scaled 0.5x.

### MRS.Q (extends MrQ)

- **MPPI planning** replaces policy+noise for collection. CEM loop with policy trajectories + random trajectories, Gumbel-softmax elite selection, warm-starting via `prev_mean` in carry.
- **Q-ensemble** (default 10): target uses `min`, value loss broadcasts `huber_loss`, policy loss averages over ensemble.
- **SEM**: `sem(x, groups=8)` — softmax within groups, concatenated. Requires `zs_dim % sem_groups == 0`.
- **JIT correctness**: MPPI uses `jax.lax.*` (modules read-only). Encoder burst uses `nnx.fori_loop`. Burst/RL branching uses `nnx.cond`.

### PPO-RND (Dual Optimizer)

**Separate optimizers** for policy and RND predictor. The RND predictor overfits and collapses exploration signal when trained for multiple epochs with the policy — single epoch with own optimizer (`rnd_learning_rate=3e-4`, `rnd_update_epochs=1`) preserves useful intrinsic reward.

### PPO-RNN

- **Hidden state reset**: Done at timestep t means obs at t is from new episode — reset BEFORE GRU step: `hstate = hstate * (1 - done[:, None])`.
- **Minibatch shuffling**: Shuffles along env axis (axis=1) preserving temporal structure, unlike standard PPO's flatten-and-shuffle.
- **Sequence recomputation**: Must use `nnx.scan` (not `jax.lax.scan`) because model is the gradient target. Model and hstate packed as single carry tuple.

### DrQ (discrete, Atari-100k)

- **Reference is Dopamine labs/atari_100k DrQ.gin**, not the PyTorch SAC release: Double + Dueling DQN, n-step 10, ONE shift+intensity augmentation each for obs/next_obs (no K/M target averaging — that's the DMC variant). Loss is **Huber**, not MSE.
- **Augmentation dtype order**: `random_shift` on uint8 (precision-free crop), `random_intensity` promotes to float32, encoder's `/255` after — commutes with Dopamine's float-first order. `max_offset=2*shift_pad` (full-range offsets, exclusive) matches `_per_image_random_crop`.
- **`polyak=0.0` + `target_update_freq=1` encodes a HARD target copy every env step** (Dopamine `target_update_period=1` at RR=1) via `maybe_update_targets`'s freq==1 polyak branch. Tuning polyak away from 0.0 silently breaks fidelity.
- **Sample-time n-step** (`ReplayBuffer.sample_nstep`): buffer stores 1-step rows; truncated n-step returns are built when a batch is sampled, matching Dopamine. This serves the last n−1 pre-terminal transitions with done=True targets and stores no masked filler rows — the at-insert ring (still used by DrQ-v2/Rainbow) drops those windows, giving 10× sparser terminal signal.
- Epsilon anneal starts AFTER buffer fill (`_anneal_epsilon`), not from step 0.
- **Reference network layout**: conv (padding **SAME** — Flax `nn.Conv` default in Dopamine's `FullRainbowNetwork`, flatten 11×11×64=7744) → Dense(512) → ReLU → dueling heads, 4.05M params. The single 512 hidden layer is the ENCODER's dense, so the yaml sets `agent_kwargs.hidden_layer_sizes: []` and `encoder_kwargs.padding: SAME`. Adding `[512]` back creates a second hidden layer the reference doesn't have; `padding: VALID` (Nature-DQN canon, other algos' default) shrinks the trunk to 3136/1.95M.
- `noop_max` (train+eval no-op starts) defaults to gymnasium's 30; the Dopamine reference applies NONE — `--noop-max 0` is reference-faithful (A/B pending).

### FastSAC

- **Reference is RL-X `fastsac/flax_full_jit`** (default path only: n_steps=1, no clipped double-Q, no grad clipping, no LR anneal). Defaults in `FastSACConfig` = reference algorithm defaults + RL-X's G1 env config (4096 envs, `obs_key: privileged_state`).
- **Truncations bootstrap**: the Playground adapter puts `truncated` (time limit without physical termination) and `final_obs` (pre-reset obs, which gymnax's autoreset overwrites) in `info`. FastSAC reads them with `.get` defaults, so on envs without these keys every done is terminal, like nexrl SAC.
- **Each critic learns from its own target critic's projection** (no min over critics); the policy loss uses the mean of the two expected Qs.
- **Alpha loss uses the next-state log-probs of the critic step** and is in the same gradient call as the critic loss. log_alpha has its own AdamW, weight decay included — the reference's choice, which pulls α toward 1.
- **Polyak (τ=0.125) runs after every critic update** (8 per env step), not once per env step.
- **Obs RMS is updated from the sampled batch** (obs + next_obs of all U batches), not from fresh env obs; warm-up acts with the identity-initialized normalizer.
- Buffer is a `(buffer_size_per_env, num_envs)` ring (`CircularBuffer.append` of one row per vectorized step); sampling draws a row and an env uniformly.

### DrQ-v2

- **No target encoder/actor** — only `critic_target` with Polyak averaging.
- **Gradient flow**: Encoder gets gradients only through critic loss. Actor loss uses `stop_gradient` on encoder features.
- **N-step returns**: Computed at collection time via `NStepState` ring buffer (n=3).
- Bypasses `create_env()` — calls `create()` directly with `pixels=True, rgb=True, action_repeat=2, frame_stack=3`.

### SGF

- **VICReg + augmentations** instead of RSSM + KL + reconstruction. Deterministic encoder, stochasticity via image augmentations.
- **Encoder LayerNorm on output** — critical for representation stability.
- **Transition predictor** trains on un-augmented encoder outputs with stop_gradient. Skip connection: `next_y = y + transition_mlp(cat(y, action))`.
- Atari-100k only (image observations required).

### MuZero

- **Distributional value/reward**: rlax `signed_hyperbolic`/`signed_parabolic` with 2-hot categorical targets.
- Single joint optimizer over `MuZeroModel`. K-step unrolling loss with gradient scaling on latent states.

## Design Decisions

Choices intentionally rejected. Do not attempt without discussion.

| Decision | Rationale |
|----------|-----------|
| **Don't unify algorithm pairs** (SAC/SimBa, SAC/FastSAC, MPO/MPO-cont, AlphaZero/MuZero, Dreamer/MuZero, MrQ/TD3, MrQ/MRS.Q, SGF/Dreamer, DrQ-v2/TD3, PPO/PPO-RNN) | Each algo is standalone — architecture, loss, optimizer structure, and training flow differ fundamentally. Shared code would obscure algorithmic differences behind conditionals |
| **Don't extract Trajectory/AdvantageMinibatch to utils** | V-MPO's variant differs; ~30 LOC per file preserves self-containedness |
| **Don't use config inheritance** | Explicit field duplication keeps each config independently readable; inheritance changes dataclass field ordering |
| **Don't extract metrics accumulation helpers** | Interleaved with carry updates; `x * 0` tracing semantics would be obscured |
| **Keep `algos/` flat** (no subfolders) | ~20 files, manageable; subfolders obscure "every algo is a standalone peer" |
| **Keep dataclass configs over Hydra** | tyro two-pass CLI handles typed overrides; Hydra adds OmegaConf complexity incompatible with frozen dataclasses and JIT |
| **Keep `utils.py` as one file** | ~355 LOC; natural split points exist at ~500 LOC but aren't warranted yet |
