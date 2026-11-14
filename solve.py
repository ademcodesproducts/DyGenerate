# solve.py
#
# What it is for:
#   Small exact solvers used while making items, never while scoring.
#
# What to include:
#
# 1. A Futoshiki solution counter
#    - Backtracking over the canonical puzzle, stopping as soon as it finds a second solution.
#    - Used to keep only puzzles with exactly one solution. Boards up to 7 are fast enough.
#
# 2. A Countdown solvability check
#    - Searches for any expression that reaches the target with the given numbers.
#    - Used to drop unsolvable items (Countdown can have several valid answers; that is fine).
#
# 3. A check that each generator's reference solution passes our own checker in verify.py.
#    An item whose reference fails is a conversion bug; count and report these.
