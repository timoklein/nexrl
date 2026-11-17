# Changelog

## [1.0.0] - 2026-11-17

### Algorithms

- On-policy: REPPO
- Off-policy: BBF, DrQ (discrete Atari-100k variant), FastSAC, FlashSAC, FastTD3, XQC

### Features

- Evaluation runs on dedicated env instances (`make_eval_env`) in all algorithms
- `epsilon_eval` config field for DQN, IQN, PQN and Rainbow
- Sample-time n-step returns (`ReplayBuffer.sample_nstep`)
- MuJoCo Playground adapter: dict-observation selection (`obs_key`) and `action_repeat` (XQC config field, also passed to `dm_control/`)
- `policy_obs_dim` config field (REPPO, FastSAC, FlashSAC, FastTD3): the policy sees only the first `policy_obs_dim` obs dims, the critic the full obs
- All env adapters report truncation info (`info["truncated"]`, `info["final_obs"]`) and reset inside the step; FastSAC, FlashSAC, FastTD3 and XQC bootstrap through truncations on every suite

### Fixes

- FastSAC, FlashSAC and FastTD3: the policy sees only the first `policy_obs_dim` obs dims (G1: the 103-dim `state` prefix of `privileged_state`), as in RL-X; the critic keeps the full obs. Before, the policy saw all 216 dims
- `gymnasium/` adapter: time limits never fired (the per-step reset zeroed the step counter); vmapped reset now works
- DMC and Navix adapters: removed the phantom transition after each episode end (two-step / lazy reset)
- CI test job split into 4 parallel shards
- `--track` now stops before training when `wandb` is missing (it used to fail after training); `wandb` is the new `track` extra
- FlashSAC `test_env4` pins full float32 matmul precision; it failed on A100 GPUs at the default TF32 precision

### Other

- `THIRD_PARTY_NOTICES.md` lists code derived from rejax, purejaxql, Bigger Better Faster, Dopamine, RL-X, craftax_baselines and Meta's MR.Q / MRS.Q with their licenses
- MrQ and MRS.Q (`algos/mrq.py`, `algos/mrsq.py`, `networks/mrq.py`, `networks/mrsq.py`) derive from Meta's CC BY-NC 4.0 code and are licensed CC BY-NC 4.0 (non-commercial use only); the rest of nexrl stays Apache-2.0
- Removed unused rejax images from `img/` and the unused rejax `configs/configs.zip`

## [0.1.0] - 2026-04-14

Initial release of nexrl, a pure-JAX RL library.

### Algorithms

- On-policy: PPO, PPO-RNN, PPG, PPO-RND, PPO-ICM, PPO-E3B, V-MPO, PQN
- Off-policy: SAC, SAC-SimBa, TD3, DQN, IQN, Rainbow, MPO, DrQ-v2
- Model-based: Dreamer, MuZero, AlphaZero, MrQ, MRS.Q, SGF

### Features

- All algorithms are fully `jax.jit` and `jax.vmap` compatible
- Typed dataclass configs with YAML support and CLI overrides via tyro
- Flax NNX for all neural network components
- W&B integration for post-training upload
- Environment adapters for Gymnax, Brax, Craftax, Atari, DM Control, MuJoCo Playground, and more
