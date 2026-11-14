# parse.py
#
# What it is for:
#   Turns a model's raw output into a final answer, with one rule for every model and generator.
#
# What to include:
#
# 1. Answer extraction: take the last answer-tag block in the output. Strip any reasoning that comes before it.
#
# 2. Futoshiki: read n lines of n integers from that block. Tolerate extra spaces and commas; reject anything else.
#
# 3. Countdown: read one expression from that block; normalise symbols such as × and ÷.
#
# 4. A status for every output, saved alongside the answer
#    - ok: an answer was parsed
#    - truncated: the model hit the token limit (from the response's finish reason)
#    - no_answer: finished but gave no answer tags
#    - malformed: tags present but the content could not be parsed
#    These statuses feed the truncation and null-rate tables; never silently count them as wrong only.
#
# 5. Optional fallback, off by default: if there are no tags, try the last grid or expression in the text.
#    Report results with and without the fallback if it is ever switched on.
