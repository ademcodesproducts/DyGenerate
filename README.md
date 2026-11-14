# Evaluation harness (Exp 0 and Exp 1)

What this folder is for: one pipeline that turns puzzle generators into scored model answers, so every model is judged on the same items, the same prompt wrapper and the same checker. Arul pulls training draws from it (Exp 3), Rishbha builds the gold rungs on it, and Exp 1 runs on it first.

What to include in this README once the code exists:

- One paragraph on the goal of Exp 1: does SynLogic-7B (Futoshiki) and a 3B Countdown RL model show a home advantage over an independent generator, beyond what their parents and broad models show?
- The workflow, in order: make items, run inference, score, analyze.
- The exact command for each step, for Kaggle and for the RTX 4060 (WSL2).
- Which generator versions were used (commit hashes, copied from config.py at run time).
- The definition of "inside the training range" (B) per task, with the source it came from (see docs/lookups.md).
- Known limitations: fp16 on T4, truncation rates, any generator whose code is not independent.

Layout:

- config.py: every setting in one place
- items.py: item schema and canonical puzzle formats
- gen_futoshiki.py, gen_countdown.py: adapters around existing generators
- prompts.py: the shared instruction wrapper and the two prompt versions
- solve.py: solution counting (uniqueness, solvability)
- verify.py: our own checkers, used for all scoring
- parse.py: extracting the final answer from a model's output
- make_items.py: writes item files
- run_vllm.py: runs a model over item files
- score.py: joins answers with items and marks each correct or not
- analyze.py: accuracy tables, home advantage, difference in differences, plots
- tests/: checks for the checkers and adapters
- docs/: setup notes and the lookups still to do
- items/, responses/, results/: data folders (see the README in each)

Done when: item files exist, one accuracy-vs-difficulty plot per task, the first home-advantage and difference-in-differences numbers with intervals, and truncation rates. That result is the Gate A decision on Nov 23.
