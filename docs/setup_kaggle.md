# Kaggle setup (Futoshiki, 7B models)

What this note is for: the steps to run the 7B models on Kaggle's free GPUs, written down once so every teammate does it the same way.

What to include:

- Account steps: phone verification (needed for internet and GPUs), the weekly GPU quota (about 30 hours; check the current number).
- Notebook settings: accelerator "GPU T4 x2", internet on, persistence settings.
- Installing the harness: how the harness folder gets into the notebook (Kaggle dataset upload or a git clone) and which vllm version installs cleanly.
- Model settings that matter on T4: fp16 (no bf16 on T4), tensor-parallel size 2, the max model length that fits in memory.
- Keeping results: write outputs to the working folder, save them as a Kaggle dataset version after each session, and how to resume after the 12-hour session limit.
- Order of runs: smoke test on Qwen2.5-7B-Instruct, then SynLogic-7B, its parent, Qwen2.5-7B-Instruct, and R1-Distill last because its traces are longest.
- Measured numbers from the smoke test: tokens per second, average output length per model, so the pilot can be budgeted.
