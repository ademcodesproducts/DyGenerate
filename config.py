"""Every setting in one place.

No other file hard-codes a path, a seed, a model name or a grid value. Changing an experiment means
changing this file only. Values marked LOOKUP come from docs/lookups.md and must be confirmed there.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- 1. Paths

ROOT = Path(__file__).resolve().parent
ITEMS_DIR = ROOT / "items"
RESPONSES_DIR = ROOT / "responses"
RESULTS_DIR = ROOT / "results"

# Generator code that is not on PyPI is cloned here (git clone, then record the commit).
# Override with environment variables when the clones live elsewhere (e.g. a Kaggle dataset).
SYNLOGIC_DIR = Path(os.environ.get("SYNLOGIC_DIR", ROOT / "external" / "SynLogic"))
TINYZERO_DIR = Path(os.environ.get("TINYZERO_DIR", ROOT / "external" / "TinyZero"))  # only for the cross-check

# The exposed models' actual training data. make_items drops any eval item that matches a row here.
_HF = "https://huggingface.co/datasets"
TRAINING_SETS = {
    "futoshiki": {"url": f"{_HF}/MiniMaxAI/SynLogic/resolve/main/synlogic_easy/train.parquet",
                  "path": ROOT / "external" / "training_sets" / "synlogic_easy_train.parquet"},
    "countdown": {"url": f"{_HF}/Jiayi-Pan/Countdown-Tasks-3to4/resolve/main/data/train-00000-of-00001.parquet",
                  "path": ROOT / "external" / "training_sets" / "countdown_tasks_3to4_train.parquet"},
}

# ---------------------------------------------------------------- 2. Seed ranges (half-open, never overlapping)

SEED_RANGES = {
    "eval": (0, 1_000_000),           # Exp 1 and gold items
    "train": (1_000_000, 2_000_000),  # Arul's exposure draws (Exp 3)
    "calib": (2_000_000, 3_000_000),  # calibration pool (Exp 2)
}
# Each cell of a grid gets its own block of seeds inside the purpose's range, so adding items to one
# cell never shifts the seeds of another. 8 cells x 100,000 fits inside 1,000,000.
SEEDS_PER_CELL = 100_000

# ---------------------------------------------------------------- 3. Difficulty grids
#
# Shared knobs for every task: "size" (Futoshiki board size, Countdown number count) and "level"
# (0 = easier, 1 = harder). A cell is one (size, level) pair. Each generator maps a cell onto its own
# native settings below; the native settings actually used are stored with every item.

TASKS = {
    "futoshiki": {
        "generators": ["synlogic", "reasoning_gym"],
        "home": "synlogic",            # SynLogic-7B's training generator
        "sizes": [4, 5, 6, 7],
        "levels": [0, 1],
        # SynLogic-7B trained on SynLogic-Easy (model card), whose 600 Futoshiki rows are all 4x4:
        # 308 at (2 inequalities, 2 givens) and 292 at (4, 3), i.e. our cells (4, 0) and (4, 1).
        "range_b": {(4, 0), (4, 1)},
        "range_b_confirmed": True,
    },
    "countdown": {
        "generators": ["tinyzero", "reasoning_gym"],
        "home": "tinyzero",            # the 3B Countdown RL models' training data
        "sizes": [3, 4, 5, 6],
        "levels": [0],                 # Countdown has one level; difficulty is the number count
        "range_b": {(3, 0), (4, 0)},   # Countdown-Tasks-3to4 has 3 or 4 numbers
        "range_b_confirmed": True,
    },
}

# Futoshiki native settings per cell.
# SynLogic: (number of inequality signs, number of prefilled cells). The 4x4 and 5x5 rows are SynLogic's
# own run.sh datasets; 6x6 and 7x7 are our extrapolation (keep the ratios roughly constant).
FUTOSHIKI_NATIVE = {
    "synlogic": {
        (4, 0): {"num_inequality_signs": 2, "num_prefilled_coords": 2},
        (4, 1): {"num_inequality_signs": 4, "num_prefilled_coords": 3},
        (5, 0): {"num_inequality_signs": 7, "num_prefilled_coords": 4},
        (5, 1): {"num_inequality_signs": 10, "num_prefilled_coords": 5},
        (6, 0): {"num_inequality_signs": 10, "num_prefilled_coords": 6},
        (6, 1): {"num_inequality_signs": 14, "num_prefilled_coords": 7},
        (7, 0): {"num_inequality_signs": 14, "num_prefilled_coords": 8},
        (7, 1): {"num_inequality_signs": 19, "num_prefilled_coords": 9},
    },
    # Reasoning Gym: difficulty 0-3 sets the chance that each adjacent pair gets an inequality.
    "reasoning_gym": {level: {"difficulty": d} for level, d in {0: 1, 1: 3}.items()},
}

# Countdown native settings. Both generators get the same value and target ranges, taken from the
# home training data (Countdown-Tasks-3to4, all 490,364 rows: numbers 1-99, targets uniform on 10-100,
# every sampled row solvable with all numbers once), so uniform draws + a solvability filter match it.
COUNTDOWN_VALUE_RANGE = (1, 99)
COUNTDOWN_TARGET_RANGE = {3: (10, 100), 4: (10, 100), 5: (10, 100), 6: (10, 100)}
# Both TinyZero's reward and Reasoning Gym's scorer require every number to be used exactly once.
COUNTDOWN_USE_ALL_NUMBERS = True

# SynLogic picks inequality signs at random, so most of its small puzzles have many solutions.
# Requiring uniqueness would reject nearly all of them, and our checker accepts any valid grid,
# so uniqueness is recorded on every item (n_solutions, capped at 2) instead of filtered.
FUTOSHIKI_REQUIRE_UNIQUE = False

# ---------------------------------------------------------------- 4. Item counts

ITEM_COUNTS = {
    "smoke": {"total": 20, "samples": 1},     # spread evenly over cells
    "pilot": {"per_cell": 50, "samples": 1},
    "final": {"per_cell": 150, "samples": 4},
}

# ---------------------------------------------------------------- 5. Model registry
#
# role: exposed (trained on the home generator), parent (the exposed model's starting point), broad.
# format: "chat" applies the model's own chat template; "plain" sends the text as one prompt.
# Check every hf_id on its model card before a run (docs/lookups.md).

KAGGLE = {"machine": "kaggle_t4x2", "dtype": "half", "tensor_parallel_size": 2, "gpu_memory_utilization": 0.9}
RTX4060 = {"machine": "rtx4060", "dtype": "half", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}

MODELS = {
    # Futoshiki, Kaggle T4 x2
    "synlogic-7b": {"hf_id": "MiniMaxAI/SynLogic-7B", "role": "exposed", "task": "futoshiki",
                    "home": "synlogic", "format": "chat", "max_model_len": 10240, "max_new_tokens": 8192, **KAGGLE},
    "qwen2.5-7b": {"hf_id": "Qwen/Qwen2.5-7B", "role": "parent", "task": "futoshiki",
                   "format": "plain", "max_model_len": 10240, "max_new_tokens": 8192, **KAGGLE},
    "qwen2.5-7b-instruct": {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "role": "broad", "task": "futoshiki",
                            "format": "chat", "max_model_len": 10240, "max_new_tokens": 8192, **KAGGLE},
    "r1-distill-qwen-7b": {"hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "role": "broad", "task": "futoshiki",
                           "format": "chat", "max_model_len": 18432, "max_new_tokens": 16384, **KAGGLE},
    # Countdown, RTX 4060 (WSL2)
    # Exposed: philschmid's mini-R1, GRPO from Qwen2.5-3B-Instruct on 45k rows of Countdown-Tasks-3to4,
    # 450 steps, training reward for a correct equation rose from ~6% to ~40%. "revision" pins the final
    # weights; without it vLLM would also download every checkpoint folder.
    "countdown-rl-3b": {"hf_id": "philschmid/qwen-2.5-3b-r1-countdown", "revision": "099c0f8cbfc522e7c3a476edfb749f576b164539",
                        "role": "exposed", "task": "countdown", "home": "tinyzero", "format": "chat",
                        "max_model_len": 4096, "max_new_tokens": 1024, **RTX4060},
    "qwen2.5-3b-instruct": {"hf_id": "Qwen/Qwen2.5-3B-Instruct", "role": "parent", "task": "countdown",
                            "format": "chat", "max_model_len": 4096, "max_new_tokens": 1024, **RTX4060},
    "qwen2.5-3b": {"hf_id": "Qwen/Qwen2.5-3B", "role": "base", "task": "countdown",
                   "format": "plain", "max_model_len": 4096, "max_new_tokens": 1024, **RTX4060},
    # Broad model from another family; gated on Hugging Face (request access first).
    "llama3.2-3b-instruct": {"hf_id": "meta-llama/Llama-3.2-3B-Instruct", "role": "broad", "task": "countdown",
                             "format": "chat", "max_model_len": 4096, "max_new_tokens": 1024, **RTX4060},
}

# Intermediate checkpoints of the Countdown model (commit per training step): a free exposure dose curve.
COUNTDOWN_RL_CHECKPOINTS = {
    25: "b821786444e56bbb5d30a099d81aee818107f422", 50: "7e5167b7bad16dfd218e8acba74f48331f767b36",
    75: "9ab4dbfc85f8a1ed6bfdf09738c3960da2081da7", 100: "2d5757d211bbe485750f16a6ef37b5ec698fc9d7",
    125: "ae5d28bc12c973cd7a44a94b510658cdaf924e46", 150: "02b892a27b9c0f75f475fd2e396c81d491aedcc8",
    175: "04b9f34ce27c54ced4fc091a28848e899a09f694", 200: "7bc26ddbc5d39f9eb99e2cc6cfed4008ce5410ea",
    250: "ac27ceb8ea3bb4dd6aed2eb9cbe2d771a84d5778", 300: "4bab7d2a371da0dcd681853beb2f9316314c71d9",
    325: "587237376e0055dc3ced0e079397d103bc4cfc31", 350: "eaf0bde9e8aa862b7d3d2f207349b9bb6d25ef6d",
    375: "1e22d39f3ef664ab1277b55602bb01efba6fe77b", 400: "1039b386053ec6a24fa29c122aac4415cef94fcd",
    425: "3b18948b10d422cf3212cee9bf526fcbffa8e4a5", 450: "0288df2bb763b0131072549214b3854360360fbc",
}
for _step, _rev in COUNTDOWN_RL_CHECKPOINTS.items():
    MODELS[f"countdown-rl-3b-step{_step}"] = {**MODELS["countdown-rl-3b"], "revision": _rev,
                                              "role": "exposed_checkpoint", "step": _step}

# ---------------------------------------------------------------- 6. Sampling

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "seed": 1234}

# ---------------------------------------------------------------- 7. Prompts and analysis

PROMPT_VERSIONS = ["native", "canonical"]

BOOTSTRAP_REPS = 2000
EQUIVALENCE_MARGIN = 0.03  # "no difference" means the 90% interval lies inside +/- 3 points
