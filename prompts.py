# prompts.py
#
# What it is for:
#   Builds the exact text each model sees, so a home advantage cannot come from one generator
#   having friendlier instructions or a different answer format.
#
# What to include:
#
# 1. One shared instruction wrapper per task
#    - Futoshiki: solve the puzzle, show reasoning, give the final grid as n lines of n space-separated numbers
#      inside answer tags.
#    - Countdown: give one arithmetic expression using the given numbers, each at most once, inside answer tags.
#    - The same wrapper for every generator and every model.
#
# 2. Two prompt versions per item
#    - native: the generator's own puzzle text inside the shared wrapper.
#    - canonical: the puzzle re-rendered from its canonical form, so both generators look identical in text.
#    The difference between the two splits a home advantage into "surface wording" and "how puzzles are built".
#
# 3. Model formatting
#    - Chat models: the user message only; the model's own chat template is applied at inference time.
#    - Base models (the parents): the same text as a plain prompt, ending where the answer should start.
#    - No system prompts that differ between models.
#
# 4. A version string for the wrapper text, saved with every response, so a later wording change is traceable.
