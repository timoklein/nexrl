# Third-party notices

nexrl is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). Parts of it are derived
from the projects below. nexrl has modified all of these files: the code was ported to Flax NNX
and restructured into nexrl's standalone `train_*` functions and shared utilities.

## rejax

- Source: https://github.com/keraJLi/rejax
- License: Apache License 2.0 (same text as [LICENSE](LICENSE))
- Authors: Jarek Liesen, Chris Lu, Robert Lange
- Used in: nexrl is based on rejax; parts of its library layout and algorithm implementations
  derive from it.

## purejaxql

- Source: https://github.com/mttga/purejaxql
- License: Apache License 2.0 (same text as [LICENSE](LICENSE))
- Copyright 2024 Matteo Gallici
- Used in: `src/nexrl/algos/pqn.py`

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
