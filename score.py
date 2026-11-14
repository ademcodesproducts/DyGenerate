# score.py
#
# What it is for:
#   Joins raw responses with their items, parses each answer and marks it correct or not with our own checker.
#
# What to include:
#
# 1. Load every response file and the matching item files; join on item ID.
#
# 2. For each sample: parse (parse.py), then check (verify.py). Keep the parse status.
#
# 3. Output: results/scores.csv in long format, one row per sample, with columns for:
#    model, model role, task, generator, home or away for that model, region (inside or beyond the range),
#    board size or number count, difficulty level, prompt version, item ID, sample index,
#    correct (1 or 0), parse status, generated tokens.
#
# 4. A sanity report: rows per model and generator, share of each parse status, any items with no responses.
