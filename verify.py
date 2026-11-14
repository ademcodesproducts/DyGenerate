# verify.py
#
# What it is for:
#   Our own checkers, used to score every response from every generator, so scoring cannot differ by generator.
#
# What to include:
#
# 1. Futoshiki checker (input: canonical puzzle and a parsed grid; output: 1 or 0)
#    - Grid is n by n.
#    - Every row and every column is a permutation of 1 to n.
#    - Every given cell matches.
#    - Every inequality holds.
#
# 2. Countdown checker (input: numbers, target and a parsed expression; output: 1 or 0)
#    - Only the given numbers are used, each at most once.
#    - Only + - * / and parentheses.
#    - Evaluate safely with a small expression parser, never Python's eval.
#    - Result equals the target exactly (watch division: use exact fractions).
#
# 3. A cross-check routine
#    - On about 100 items per generator, compare our checker's verdict with the generator suite's own verifier
#      (Reasoning Gym's score_answer, SynLogic's checker) on both correct and deliberately broken answers.
#    - Any disagreement must be explained before the pilot runs.
