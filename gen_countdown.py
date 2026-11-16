"""Adapters for Countdown, the task tested on the RTX 4060.

Home: the data the public 3B Countdown RL models trained on. TinyZero trains on the Hugging Face dataset
  Jiayi-Pan/Countdown-Tasks-3to4 (3 or 4 numbers in 1-99, targets 10-100), but the script that made that
  dataset is not in the TinyZero repo. So the home adapter makes fresh items with the same ranges and the
  same rule (all numbers, each once, + - * /), keeps only solvable ones, and uses TinyZero's own question
  sentence. LOOKUP: find the real generation script; if its target distribution differs from uniform,
  match it here. Never reuse rows from the training dataset itself.
Away: Reasoning Gym's countdown. It builds a random expression over the numbers and takes its value as
  the target, so every item is solvable by construction and its reference is that expression.
  Note: one of its three question templates says "at most once", but its scorer requires all numbers.
Confirm Reasoning Gym's countdown does not reuse TinyZero's code (docs/lookups.md). Reading both found
no shared code.
"""

import importlib.metadata
import random

import config
import items

TINYZERO_COMMIT = "tinyzero@95df88f2dcb05f33bd18da546531b52d0954c18b"  # prompt text and reward rule
TINYZERO_QUESTION = ("Using the numbers {numbers}, create an equation that equals {target}. You can use basic "
                     "arithmetic operations (+, -, *, /) and each number can only be used once.")


def native_params(size):
    lo, hi = config.COUNTDOWN_VALUE_RANGE
    tlo, thi = config.COUNTDOWN_TARGET_RANGE[size]
    return {"num_count": size, "min_value": lo, "max_value": hi, "min_target": tlo, "max_target": thi}


def generate_tinyzero(size, level, seed):
    """Fresh numbers and a target drawn uniformly from the training ranges. May be unsolvable;
    make_items keeps only solvable ones and fills the reference answer from solve.py."""
    p = native_params(size)
    rng = random.Random(seed)
    numbers = [rng.randint(p["min_value"], p["max_value"]) for _ in range(size)]
    target = rng.randint(p["min_target"], p["max_target"])
    return {
        "puzzle": items.canonical_countdown(numbers, target),
        "answer": None,
        "native_params": p,
        "native_text": TINYZERO_QUESTION.format(numbers=numbers, target=target),
        "native": {"target": target, "nums": numbers},
        "commit": TINYZERO_COMMIT,
    }


def rg_native_text(question):
    """Reasoning Gym's own question without its answer-format block."""
    return question.split("\n\nFinal answer format instructions:")[0]


def generate_reasoning_gym(size, level, seed):
    import reasoning_gym
    p = native_params(size)
    dataset = reasoning_gym.create_dataset("countdown", min_numbers=size, max_numbers=size,
                                           min_value=p["min_value"], max_value=p["max_value"],
                                           min_target=p["min_target"], max_target=p["max_target"],
                                           seed=seed, size=1)
    raw = dataset[0]
    meta = raw["metadata"]
    return {
        "puzzle": items.canonical_countdown(meta["numbers"], meta["target"]),
        "answer": raw["answer"],
        "native_params": p,
        "native_text": rg_native_text(raw["question"]),
        "native": raw,
        "commit": "reasoning-gym==" + importlib.metadata.version("reasoning-gym"),
    }


GENERATORS = {"tinyzero": generate_tinyzero, "reasoning_gym": generate_reasoning_gym}


def generate(generator, size, level, seed):
    return GENERATORS[generator](size, level, seed)
