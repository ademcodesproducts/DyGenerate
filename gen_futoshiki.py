# gen_futoshiki.py
#
# What it is for:
#   Thin adapters around the two existing Futoshiki generators. We do not write our own generator.
#   SynLogic's generator is SynLogic-7B's home; Reasoning Gym's is the away generator.
#
# What to include, once per generator (SynLogic, Reasoning Gym):
#
# 1. A function that takes the shared knobs (board size, difficulty level) and a seed,
#    maps them onto that generator's native settings, and calls the generator.
#    Record the native settings actually used.
#
# 2. A converter from the generator's native output to the canonical form in items.py:
#    board size, given cells, inequality pairs, reference solution.
#    Watch the conventions: 0- or 1-indexed cells, which way an inequality points,
#    horizontal versus vertical inequalities.
#
# 3. The generator's own puzzle text, kept unchanged for the "native" prompt.
#
# 4. The generator's code version, read at run time (SynLogic: git hash of the clone; Reasoning Gym: package version).
#
# 5. A note of anything the two generators do differently in how they build puzzles
#    (how many givens, how dense the inequalities are). That difference is what a home advantage
#    can come from, so it should be written down, not hidden.
#
# Before relying on it: confirm the two implementations share no code (see docs/lookups.md).
