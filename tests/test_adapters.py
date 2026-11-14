# tests/test_adapters.py
#
# What it is for:
#   Proves that converting each generator's native puzzle into canonical form loses nothing.
#
# What to include, for every adapter:
# - On a sample of items, the generator's own reference solution passes our checker after conversion.
# - Inequality direction and row/column indexing match the native puzzle (test one hand-checked example per generator).
# - The same seed and settings give the same item twice (generation is reproducible).
# - Items from the eval, train and calib seed ranges never share a canonical hash.
# - The canonical prompt for the same canonical puzzle is identical whichever generator produced it.
