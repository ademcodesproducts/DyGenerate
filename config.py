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
        # LOOKUP: the settings SynLogic-7B trained on. SynLogic's run.sh only ships 4x4 and 5x5 data,
        # so this is the working guess until the dataset card confirms it.
        "range_b": {(4, 0), (4, 1), (5, 0), (5, 1)},
        "range_b_confirmed": False,
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
# home training data (Countdown-Tasks-3to4: numbers 1-99, targets 10-100, checked on 300 rows).
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
    "countdown-rl-3b": {"hf_id": "LOOKUP/countdown-rl-3b", "role": "exposed", "task": "countdown",
                        "home": "tinyzero", "format": "plain", "max_model_len": 4096, "max_new_tokens": 1024, **RTX4060},
    "qwen2.5-3b": {"hf_id": "Qwen/Qwen2.5-3B", "role": "parent", "task": "countdown",
                   "format": "plain", "max_model_len": 4096, "max_new_tokens": 1024, **RTX4060},
    "qwen2.5-3b-instruct": {"hf_id": "Qwen/Qwen2.5-3B-Instruct", "role": "broad", "task": "countdown",
                            "format": "chat", "max_model_len": 4096, "max_new_tokens": 1024, **RTX4060},
}

# ---------------------------------------------------------------- 6. Sampling

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "seed": 1234}

# ---------------------------------------------------------------- 7. Prompts and analysis

PROMPT_VERSIONS = ["native", "canonical"]

BOOTSTRAP_REPS = 2000
EQUIVALENCE_MARGIN = 0.03  # "no difference" means the 90% interval lies inside +/- 3 points
