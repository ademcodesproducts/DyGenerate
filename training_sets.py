"""Canonical hashes of the exposed models' actual training puzzles, so eval items never repeat one.

Without this, fresh draws collide with training rows: 21 of 50 fresh 3-number Countdown items were
verbatim rows of Countdown-Tasks-3to4. Those would measure memorised items, not generator exposure.
The files are downloaded once into external/training_sets/ (git-ignored).
"""

import json
import urllib.request

import pandas as pd

import config
import items


def _download(spec):
    path = spec["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec['url']} -> {path}")
        urllib.request.urlretrieve(spec["url"], path)
    return path


def _countdown_puzzles(df):
    for nums, target in zip(df["nums"], df["target"]):
        yield items.canonical_countdown(list(nums), target)


def _synlogic_futoshiki_puzzles(df):
    for source, extra in zip(df["data_source"], df["extra_info"]):
        if source != "futoshiki":
            continue
        game = json.loads(extra["game_data_str"])
        meta, answer = game["metadata"], game["answer"]
        if isinstance(answer, str):
            answer = json.loads(answer)
        givens = [(r, c, answer[r][c]) for r, c in meta["prefilled_coords"]]
        lt = [(a, b) if sign == "<" else (b, a) for a, b, sign in meta["constraints"]]
        yield items.canonical_futoshiki(meta["grid_size"], givens, lt)


READERS = {"countdown": _countdown_puzzles, "futoshiki": _synlogic_futoshiki_puzzles}


def training_hashes(task):
    """Set of canonical hashes for every training puzzle of the task's exposed model."""
    spec = config.TRAINING_SETS.get(task)
    if spec is None:
        return set()
    df = pd.read_parquet(_download(spec))
    return {items.canonical_hash(task, p) for p in READERS[task](df)}
