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
