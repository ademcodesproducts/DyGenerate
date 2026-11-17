# Lookups to do before the pilot

What this note is for: facts the plan depends on that nobody has confirmed yet. Fill in each answer with its source link and the date checked.

What to include, one entry each:

- SynLogic-7B: exact Hugging Face ID, its parent model, and which Futoshiki settings it trained on (the dataset card's easier subset). This defines the training range B for Futoshiki.
- SynLogic Futoshiki generator: where it lives in the repo, its settings and its verifier; the commit used.
- Reasoning Gym Futoshiki: the dataset name, its settings (board size, difficulty) and its score function; the package version used.
- Code independence: do SynLogic's and Reasoning Gym's Futoshiki share any code or derive from the same source?
- 3B Countdown RL models: exact IDs, parent model, the training data and settings (number count, target range), from each model card.
- Countdown generators: which script made that training data, and whether Reasoning Gym's countdown shares code with it.
- 3B Knights-and-Knaves models (curriculum and non-curriculum): IDs and training details, kept for later.
- Knights and Knaves independence: Reasoning Gym's version is adapted from Xie et al.'s; is there any implementation not derived from it, and do Enigmata's fixed items carry recoverable difficulty settings?
- Broad models: exact IDs for Qwen2.5-7B base, Qwen2.5-7B-Instruct, R1-Distill-Qwen-7B, Qwen2.5-3B base and Qwen2.5-3B-Instruct, and each model's recommended sampling settings.

## Answers (Nov 15-17, 2026)

**SynLogic-7B / Futoshiki**
- Model: MiniMaxAI/SynLogic-7B. Parent Qwen/Qwen2.5-7B (base). GRPO on 16k SynLogic-Easy rows (model card).
- Training range B, from the SynLogic-Easy train parquet itself: 600 Futoshiki rows (3.8% of 15,837), ALL 4x4: 308 at (2 inequalities, 2 givens), 292 at (4, 3). So B = our cells (4,0) and (4,1); 5x5 and up is beyond B. Source: huggingface.co/datasets/MiniMaxAI/SynLogic, synlogic_easy/train.parquet.
- Of those 600 training puzzles: 241 (40%) have no solution, 459 have 2+ solutions, 40 are unique; the reference answer violates the puzzle's own constraints in 500 of 600. Futoshiki exposure is small and noisy, so expect a small home advantage.
- Training prompt: a "<think> ... </think> <answer> ... </answer>" preamble, then SynLogic's text; the answer is "[[r1,r2,...]]" inside the answer tags. parse.py accepts that format.
- Generator: games/tasks/futoshiki/scripts/ at SynLogic d8c527f. Signs are random, not read off the solution.
- Reasoning Gym Futoshiki (away): dataset "futoshiki", board size 4-9, difficulty 0-3; reasoning-gym 0.1.25.
- Code independence: the two generator files share no code (different grid builder, sign rule, clue removal, text).

**Countdown**
- Exposed model: philschmid/qwen-2.5-3b-r1-countdown, final weights at revision 099c0f8. Parent Qwen/Qwen2.5-3B-Instruct (not base). TRL GRPO, 450 steps, on a 45k-row shuffle(seed=42) subset of Countdown-Tasks-3to4; reward requires all numbers exactly once. Training reward for a correct equation: ~6% at step 2, ~40% at the end. Prompt: system "You are a helpful assistant. You first thinks...", user "Using the numbers [...], create an equation that equals T ...", assistant prefill "Let me solve this step by step.
<think>". Sources: model card, trainer_state.json, philschmid/deep-learning-pytorch-huggingface training/mini-deepseek-r1-aha-grpo.ipynb.
- 16 intermediate checkpoints (steps 25-450) are commits in the model repo: config.COUNTDOWN_RL_CHECKPOINTS. A free exposure dose curve.
- Backup exposed models (no model card, provenance unknown): rayliuray/TinyZero-CountDown-Qwen2.5-3b-GRPO-Step10..400, guanning/TinyZero_Countdown_Qwen2.5-3B_PPO_token1024_step300.
- Training data: Jiayi-Pan/Countdown-Tasks-3to4, 490,364 rows (449,570 distinct puzzles), 3 or 4 numbers in 1-99, targets uniform on 10-100, every one of 400 sampled rows solvable with all numbers once. The card does not say how it was made, but uniform draws + a solvability filter reproduce the distribution, which is what gen_countdown.py does.
- Fresh draws collide with training rows: 39% of solvable 3-number home draws and 28% of Reasoning Gym's 3-number items were verbatim training puzzles. make_items now drops every training puzzle (training_sets.py) and reports the count as "in_training_set".
- Reasoning Gym countdown vs TinyZero: no shared code. Both require every number exactly once.

**Still open**
- Broad model for Countdown: meta-llama/Llama-3.2-3B-Instruct is gated, so request access on Hugging Face.
- Recommended sampling settings per model (the harness uses temperature 0.6, top-p 0.95 for all).
- Knights and Knaves items (deferred).
