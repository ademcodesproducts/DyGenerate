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

## Answered while building the harness (Nov 15, 2026)

- SynLogic-7B ID: MiniMaxAI/SynLogic-7B (SynLogic README). Training range still open: check the dataset card for which Futoshiki settings were used.
- SynLogic Futoshiki generator: games/tasks/futoshiki/scripts/ at commit d8c527f. Settings: grid_size, num_inequality_signs, num_prefilled_coords; run.sh ships 4x4 (2,2), 4x4 (4,3), 5x5 (7,4), 5x5 (10,5). Inequality signs are random, not read off the solution.
- Reasoning Gym Futoshiki: dataset "futoshiki", settings board size 4-9 and difficulty 0-3, score_answer compares to its unique solution; reasoning-gym 0.1.25.
- Futoshiki code independence: the two generator files share no code (different grid builder, sign rule, clue removal, text). Still worth one line on whether either cites the other.
- Countdown home data: TinyZero trains on Jiayi-Pan/Countdown-Tasks-3to4 (490,364 rows; 3 or 4 numbers in 1-99; targets 10-100). The script that made it is not in the TinyZero repo. Open: find it (Stream-of-Search?) and its target distribution.
- Reasoning Gym countdown vs TinyZero: no shared code (RG builds a random expression and uses its value as target). Both require every number exactly once.
