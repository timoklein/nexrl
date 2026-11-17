# Third-party notices

nexrl is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). Parts of it are derived
from the projects below. nexrl has modified all of these files: the code was ported to Flax NNX
and restructured into nexrl's standalone `train_*` functions and shared utilities.

## rejax

- Source: https://github.com/keraJLi/rejax
- License: Apache License 2.0 (same text as [LICENSE](LICENSE))
- Authors: Jarek Liesen, Chris Lu, Robert Lange
- Used in: nexrl is based on rejax. Files derived from it:
  - environment adapters: `src/nexrl/compat/brax2gymnax.py`, `craftax2gymnax.py`,
    `gymnasium2gymnax.py`, `jumanji2gymnax.py`, `kinetix2gymnax.py`, `navix2gymnax.py`,
    `playground2gymnax.py`, `_env_params.py`
  - `src/nexrl/buffers.py` (`CircularBuffer`, `Minibatch`, `ReplayBuffer`) and
    `src/nexrl/evaluate.py`
  - algorithms: `src/nexrl/algos/ppo.py`, `sac.py`, `td3.py`, `dqn.py`, `iqn.py`, `pqn.py`
  - networks: `src/nexrl/networks/policies.py`, `src/nexrl/networks/value.py`
  - hyperparameters of the `configs/*/gymnax/*.yaml` files for the algorithms above

## purejaxql

- Source: https://github.com/mttga/purejaxql
- License: Apache License 2.0 (same text as [LICENSE](LICENSE))
- Copyright 2024 Matteo Gallici
- Used in: `src/nexrl/algos/pqn.py`

## Bigger, Better, Faster (google-research)

- Source: https://github.com/google-research/google-research/tree/master/bigger_better_faster
- License: Apache License 2.0 (same text as [LICENSE](LICENSE))
- Copyright The Google Research Authors
- Used in: `src/nexrl/algos/bbf.py`, `src/nexrl/networks/bbf.py` and the BBF replay buffer in
  `src/nexrl/buffers.py` follow the reference agent, network and replay buffer
  (`spr_agent.py`, `spr_networks.py`, `subsequence_replay_buffer.py`)

## Dopamine

- Source: https://github.com/google/dopamine (`dopamine/labs/atari_100k`)
- License: Apache License 2.0 (same text as [LICENSE](LICENSE))
- Copyright 2021 The Dopamine Authors
- Used in: `src/nexrl/algos/drq.py` follows the Atari-100k DrQ agent and its `DrQ.gin` settings

## RL-X

- Source: https://github.com/nico-bohlinger/RL-X
- License: MIT (below)
- Used in: `src/nexrl/algos/fast_sac.py`, `src/nexrl/algos/fast_td3.py`,
  `src/nexrl/algos/flash_sac.py`, `src/nexrl/algos/xqc.py`, `src/nexrl/algos/reppo.py`,
  `src/nexrl/networks/fast_sac.py`, `src/nexrl/networks/fast_td3.py`,
  `src/nexrl/networks/flash_sac.py`, `src/nexrl/networks/xqc.py`, `src/nexrl/networks/reppo.py`

```
MIT License

Copyright (c) 2022 Nico Bohlinger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## craftax_baselines

- Source: https://github.com/MichaelTMatthews/craftax_baselines
- License: MIT (below)
- Used in: `src/nexrl/algos/ppo_rnn.py`, `src/nexrl/networks/rnn.py`

```
MIT License

Copyright (c) 2024 Michael Matthews

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
