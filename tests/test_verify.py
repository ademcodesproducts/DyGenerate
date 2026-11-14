# tests/test_verify.py
#
# What it is for:
#   Proves the checkers are right before any model is scored with them. A wrong checker silently corrupts every result.
#
# What to include:
#
# Futoshiki
# - A known solved puzzle passes.
# - Each single violation fails: a repeated number in a row, in a column, a wrong given cell, one broken inequality,
#   a grid of the wrong size.
#
# Countdown
# - A correct expression passes; one using a number twice fails; one using a number not given fails;
#   one that misses the target fails; division that only works with rounding fails.
# - Unsafe input (anything other than numbers, operators and parentheses) is rejected, not evaluated.
#
# Parser (parse.py)
# - Takes the last answer block when there are several; reports truncated, no_answer and malformed correctly.
