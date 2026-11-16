"""The item record and the canonical form of each puzzle type.

Canonical forms (0-indexed rows and columns):
  futoshiki: {"n": n, "givens": [[r, c, v], ...], "lt": [[[r1, c1], [r2, c2]], ...]}
             each lt pair means "first cell is less than second cell"; both lists are sorted.
  countdown: {"numbers": [sorted ints], "target": int}

An item is a plain dict with these fields:
  id, task, generator, commit, params {"size", "level", "native"}, seed, purpose, region,
  puzzle (canonical), answer (one reference solution), answer_source ("generator" or "solver"),
  n_solutions (Futoshiki only, capped at 2), native_text (the generator's own puzzle text),
  native (the generator's raw output, kept for the cross-check), prompt_native, prompt_canonical.
"""

import hashlib
import json
from pathlib import Path

import config


def canonical_futoshiki(n, givens, lt_pairs):
    givens = sorted([int(r), int(c), int(v)] for r, c, v in givens)
    lt = sorted([[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]] for a, b in lt_pairs)
    return {"n": int(n), "givens": givens, "lt": lt}


def canonical_countdown(numbers, target):
    return {"numbers": sorted(int(x) for x in numbers), "target": int(target)}


def canonical_hash(task, puzzle):
    """Hash of the canonical puzzle only (not the wording), shared across generators and purposes."""
    blob = json.dumps({"task": task, "puzzle": puzzle}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def item_id(task, generator, puzzle):
    return f"{task}-{generator}-{canonical_hash(task, puzzle)[:16]}"


def region(task, size, level):
    return "in_range" if (size, level) in config.TASKS[task]["range_b"] else "beyond_range"


def make_item(task, generator, commit, size, level, native_params, seed, purpose, puzzle, answer,
              answer_source, native_text, native, n_solutions=None):
    return {
        "id": item_id(task, generator, puzzle),
        "task": task,
        "generator": generator,
        "commit": commit,
        "params": {"size": size, "level": level, "native": native_params},
        "seed": seed,
        "purpose": purpose,
        "region": region(task, size, level),
        "puzzle": puzzle,
        "answer": answer,
        "answer_source": answer_source,
        "n_solutions": n_solutions,
        "native_text": native_text,
        "native": native,
        "prompt_native": None,
        "prompt_canonical": None,
    }


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
