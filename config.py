# config.py
#
# What it is for:
#   The single source of truth for every setting. No other file should hard-code a path,
#   a seed, a model name or a grid value. Changing an experiment means changing this file only.
#
# What to include:
#
# 1. Paths
#    - Root folders for items/, responses/ and results/.
#    - Where the SynLogic clone lives, so its commit hash can be read at run time.
#
# 2. Seed ranges, one per purpose, never overlapping
#    - eval (Exp 1 and gold items): 0 to 999,999
#    - train (Arul's exposure draws, Exp 3): from 1,000,000
#    - calib (calibration pool, Exp 2): from 2,000,000
#
# 3. Difficulty grids per task
#    - Futoshiki: board sizes 4 to 7, crossed with 2 difficulty settings.
#      Record both the shared knobs (board size, difficulty level) and each generator's native knobs.
#    - Countdown: 3, 4, 5 and 6 numbers, with a target range per count.
#    - For each task, the training range B ("inside the range"):
#      Futoshiki = the settings SynLogic-7B trained on (to fill in from docs/lookups.md);
#      Countdown = 3 to 4 numbers. Everything outside B counts as "beyond the range" (gold rung G2).
#
# 4. Item counts
#    - Smoke test: 20 items in total.
#    - Pilot: 50 items per cell, 1 sample each.
#    - Final: 100 to 150 items per cell, 4 samples each.
#
# 5. Model registry, one entry per model
#    - Short name, Hugging Face ID (check every ID on its model card), role (exposed, parent, broad),
#      task it is tested on, home generator (for exposed models), whether it uses a chat template or plain text (base models),
#      machine (Kaggle T4x2 or RTX 4060), dtype (half on T4), tensor-parallel size, max model length, max new tokens.
#    - Futoshiki on Kaggle: SynLogic-7B (exposed), Qwen2.5-7B base (parent), Qwen2.5-7B-Instruct (broad),
#      R1-Distill-Qwen-7B (broad).
#    - Countdown on the 4060: a 3B Countdown RL model (exposed), its 3B base parent, Qwen2.5-3B-Instruct (broad).
#
# 6. Sampling settings
#    - Temperature 0.6, top-p 0.95, a fixed seed; max new tokens 8k (16k for R1-Distill, about 1k for Countdown).
#
# 7. Prompt versions to run: "native" and "canonical" (see prompts.py).
