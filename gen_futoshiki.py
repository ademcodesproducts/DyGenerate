"""Thin adapters around the two existing Futoshiki generators. We do not write our own generator.

SynLogic's generator is SynLogic-7B's home; Reasoning Gym's is the away generator.

How the two build puzzles (the source of any home advantage, so it is written down, not hidden):
  SynLogic: a random Latin square, then a FIXED number of random adjacent pairs, each given a sign
    chosen at random (not read off the solution), then a FIXED number of random givens. So its
    reference grid often violates its own inequalities, puzzles can be unsolvable, and small puzzles
    usually have many solutions. Its verifier accepts any valid grid, so training rewarded that.
    Text: 1-indexed "(r,c) > (r,c)" list under an X-grid.
  Reasoning Gym: a random Latin square, then each adjacent pair gets an inequality CONSISTENT with
    the solution with probability 0.03 + 0.07 * difficulty, then givens are removed down to ~10% of
    cells while the solution stays unique. Text: one board with <, >, and the symbols for vertical
    inequalities drawn between rows.
Confirm the two share no code before relying on them (docs/lookups.md). Reading both generator
files (SynLogic d8c527f, Reasoning Gym 0.1.25) found no shared code: different grid builders,
sign rules, clue removal and text.
"""

import importlib.metadata
import random
import subprocess
import sys

import config
import items


def _jsonable(obj):
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "item"):   # numpy scalars
        return obj.item()
    return obj


# ---------------------------------------------------------------- SynLogic


def synlogic_commit():
    try:
        out = subprocess.run(["git", "-C", str(config.SYNLOGIC_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return "synlogic@" + out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "synlogic@unknown"


_synlogic_gen = None


def _synlogic():
    global _synlogic_gen
    if _synlogic_gen is None:
        if not config.SYNLOGIC_DIR.exists():
            raise FileNotFoundError(f"SynLogic clone not found at {config.SYNLOGIC_DIR}; see README")
        sys.path.insert(0, str(config.SYNLOGIC_DIR))
        from games.tasks.futoshiki.scripts.futoshiki_generator import FutoshikiGenerator
        _synlogic_gen = FutoshikiGenerator()
    return _synlogic_gen


def synlogic_native_text(question):
    """SynLogic's own text without its answer-format block (we use the shared one in prompts.py)."""
    start = question.index("\n\nPlease provide each element")
    end = question.index("\n\nSolving tips")
    return question[:start] + question[end:]


def generate_synlogic(size, level, seed):
    native_params = dict(config.FUTOSHIKI_NATIVE["synlogic"][(size, level)])
    state = random.getstate()
    random.seed(seed)   # SynLogic draws everything from the global random module
    try:
        raw = _synlogic().generate_sample(grid_size=size, is_chinese=False, **native_params)
    finally:
        random.setstate(state)
    raw = _jsonable(raw)
    answer = raw["answer"]
    meta = raw["metadata"]
    givens = [(r, c, answer[r][c]) for r, c in meta["prefilled_coords"]]
    lt = []
    for a, b, sign in meta["constraints"]:
        lt.append((a, b) if sign == "<" else (b, a))
    return {
        "puzzle": items.canonical_futoshiki(size, givens, lt),
        "answer": answer,
        "native_params": native_params,
        "native_text": synlogic_native_text(raw["question"]),
        "native": raw,
        "commit": synlogic_commit(),
    }


# ---------------------------------------------------------------- Reasoning Gym


def reasoning_gym_commit():
    return "reasoning-gym==" + importlib.metadata.version("reasoning-gym")


def rg_native_text(question):
    """Reasoning Gym's own text without its answer-format sentence (the notation lines stay)."""
    drop = "Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.\n"
    return question.replace(drop, "")


def generate_reasoning_gym(size, level, seed):
    import reasoning_gym
    native_params = dict(config.FUTOSHIKI_NATIVE["reasoning_gym"][level])
    d = native_params["difficulty"]
    dataset = reasoning_gym.create_dataset("futoshiki", min_board_size=size, max_board_size=size,
                                           min_difficulty=d, max_difficulty=d, seed=seed, size=1)
    raw = _jsonable(dataset[0])
    meta = raw["metadata"]
    givens = [(r, c, v) for r, row in enumerate(meta["puzzle"]) for c, v in enumerate(row) if v]
    lt = []
    for r1, c1, r2, c2, sign in meta["constraints"]:
        lt.append(((r1, c1), (r2, c2)) if sign == "<" else ((r2, c2), (r1, c1)))
    return {
        "puzzle": items.canonical_futoshiki(size, givens, lt),
        "answer": meta["solution"],
        "native_params": native_params,
        "native_text": rg_native_text(raw["question"]),
        "native": raw,
        "commit": reasoning_gym_commit(),
    }


GENERATORS = {"synlogic": generate_synlogic, "reasoning_gym": generate_reasoning_gym}


def generate(generator, size, level, seed):
    return GENERATORS[generator](size, level, seed)
