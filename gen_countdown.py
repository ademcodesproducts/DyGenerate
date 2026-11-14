# gen_countdown.py
#
# What it is for:
#   Adapters for Countdown, the task tested on the RTX 4060. The home generator is the one the public
#   3B Countdown RL models trained on (TinyZero-style Countdown data); the away generator is Reasoning Gym's countdown.
#
# What to include:
#
# 1. Home adapter
#    - Fresh items made the same way as the 3B model's training data: same generation script and settings
#      (record which script and commit). Do not reuse rows from the training dataset itself.
#
# 2. Away adapter
#    - Reasoning Gym's countdown, with its native settings (how many numbers, value range, target range) recorded.
#
# 3. Shared knobs for both: number count (3, 4, 5, 6) and target range.
#    3 to 4 numbers is inside the training range; 5 to 6 is beyond it.
#
# 4. Conversion to the canonical form in items.py (numbers, target) and one reference solution.
#
# 5. Each generator's own puzzle text, kept for the "native" prompt, and each generator's code version.
#
# Before relying on it: confirm which data the chosen 3B models actually trained on (model cards) and that
# Reasoning Gym's countdown does not reuse the TinyZero code (see docs/lookups.md).
