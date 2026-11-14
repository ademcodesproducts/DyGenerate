# items.py
#
# What it is for:
#   Defines what one evaluation item looks like and the canonical form of each puzzle type,
#   so items from different generators can be compared, deduplicated and scored the same way.
#
# What to include:
#
# 1. The item record, with these fields
#    - id: a stable ID derived from task, generator and the hash of the canonical puzzle
#    - task: "futoshiki" or "countdown"
#    - generator: "synlogic", "reasoning_gym", "tinyzero", ...
#    - commit: the generator's code version (git hash or package version)
#    - params: the shared difficulty knobs, plus the generator's native knobs under a separate key
#    - seed and purpose (eval, train, calib)
#    - region: "in_range" or "beyond_range", computed from config.py's range B
#    - puzzle: the canonical form (below)
#    - answer: one reference solution
#    - prompt_native and prompt_canonical: filled in by prompts.py
#
# 2. Canonical forms
#    - Futoshiki: board size n; given cells as (row, column, value); inequalities as ordered cell pairs meaning
#      "first cell is less than second cell". Fix one convention: 0-indexed rows and columns.
#    - Countdown: the list of numbers and the target.
#
# 3. Canonical hash
#    - A hash of the canonical puzzle only (not the wording), used for IDs and for removing duplicates
#      within and across generators and purposes.
#
# 4. Reading and writing items as JSON Lines, one item per line.
