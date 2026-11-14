# make_items.py
#
# What it is for:
#   The command-line step that creates item files. Run once per task, generator and purpose.
#
# What to include:
#
# 1. Arguments: task, generator, purpose (eval, train, calib), size (smoke, pilot, final).
#
# 2. The loop, for every cell of the task's grid in config.py:
#    - draw seeds from the purpose's seed range
#    - call the adapter, convert to canonical form
#    - check the reference solution with verify.py
#    - Futoshiki: keep only puzzles with exactly one solution (solve.py); Countdown: keep only solvable ones
#    - drop duplicates by canonical hash, across generators and across purposes
#    - label the region (inside or beyond the training range)
#    - build both prompt versions (prompts.py)
#    - keep going until the cell has the requested number of items
#
# 3. Output: items/<task>_<generator>_<purpose>_<size>.jsonl
#
# 4. A summary printed and saved next to the file: items per cell, how many were rejected and why
#    (duplicate, not unique, reference failed check), and the generator version used.
