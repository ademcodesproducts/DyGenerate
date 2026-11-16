"""Creates one item file per task, generator, purpose and size.

    python make_items.py --task futoshiki --generator synlogic --purpose eval --size pilot

Output: items/<task>_<generator>_<purpose>_<size>.jsonl and a .summary.json next to it.
Seeds: each cell gets its own block inside the purpose's seed range, so smoke, pilot and final files
for the same cell start from the same seeds (the pilot items are the first items of the final file).
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import config
import gen_countdown
import gen_futoshiki
import items
import prompts
import solve

ADAPTERS = {"futoshiki": gen_futoshiki, "countdown": gen_countdown}


def out_path(task, generator, purpose, size, out_dir=None):
    return Path(out_dir or config.ITEMS_DIR) / f"{task}_{generator}_{purpose}_{size}.jsonl"


def known_hashes(task, generator, purpose, out_dir=None):
    """Canonical hashes in other item files of the same task (other generators or other purposes).

    Files of the same task, generator and purpose (other sizes) are skipped: they reuse the same seeds.
    """
    seen = set()
    for path in Path(out_dir or config.ITEMS_DIR).glob(f"{task}_*.jsonl"):
        for item in items.read_jsonl(path):
            if item["generator"] == generator and item["purpose"] == purpose:
                continue
            seen.add(items.canonical_hash(task, item["puzzle"]))
    return seen


def cells(task):
    spec = config.TASKS[task]
    return [(s, lv) for s in spec["sizes"] for lv in spec["levels"]]


def per_cell_target(task, size_name):
    counts = config.ITEM_COUNTS[size_name]
    if "per_cell" in counts:
        return counts["per_cell"]
    return math.ceil(counts["total"] / len(cells(task)))


def make_one(task, generator, size, level, seed, purpose):
    """One candidate item, or (None, reason) if it is rejected."""
    try:
        raw = ADAPTERS[task].generate(generator, size, level, seed)
    except Exception as exc:   # a generator that gives up (e.g. no target in range) is a rejection
        return None, f"generator_error:{type(exc).__name__}"
    answer, source, n_solutions = raw["answer"], "generator", None
    if task == "futoshiki":
        n_solutions, solution = solve.solve_futoshiki(raw["puzzle"], limit=2)
        if n_solutions == 0:
            return None, "unsolvable"
        if config.FUTOSHIKI_REQUIRE_UNIQUE and n_solutions > 1:
            return None, "not_unique"
    item = items.make_item(task, generator, raw["commit"], size, level, raw["native_params"], seed, purpose,
                           raw["puzzle"], answer, source, raw["native_text"], raw["native"], n_solutions)
    if answer is None or not solve.reference_passes(item):
        reason = "reference_missing" if answer is None else "reference_failed"
        if task == "futoshiki":
            item["answer"] = solution
        else:
            item["answer"] = solve.solve_countdown(raw["puzzle"]["numbers"], raw["puzzle"]["target"])
            if item["answer"] is None:
                return None, "unsolvable"
        item["answer_source"] = "solver"
        item["reference_note"] = reason
        if not solve.reference_passes(item):
            return None, "solver_failed_check"   # should never happen; a bug in solve.py or verify.py
    return prompts.build_prompts(item), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(config.TASKS))
    ap.add_argument("--generator", required=True)
    ap.add_argument("--purpose", required=True, choices=list(config.SEED_RANGES))
    ap.add_argument("--size", required=True, choices=list(config.ITEM_COUNTS))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.generator not in config.TASKS[args.task]["generators"]:
        ap.error(f"{args.generator} is not a generator for {args.task}")
    if not config.TASKS[args.task]["range_b_confirmed"]:
        print(f"WARNING: training range B for {args.task} is not confirmed yet (docs/lookups.md).")

    start_seed, end_seed = config.SEED_RANGES[args.purpose]
    target = per_cell_target(args.task, args.size)
    seen = known_hashes(args.task, args.generator, args.purpose, args.out_dir)
    t0 = time.time()
    out, summary_cells, rejected = [], {}, Counter()
    commit = None
    for idx, (size, level) in enumerate(cells(args.task)):
        cell_start = start_seed + idx * config.SEEDS_PER_CELL
        assert cell_start + config.SEEDS_PER_CELL <= end_seed, "seed block leaves the purpose's range"
        kept, cell_rejects = 0, Counter()
        for seed in range(cell_start, cell_start + config.SEEDS_PER_CELL):
            if kept >= target:
                break
            item, reason = make_one(args.task, args.generator, size, level, seed, args.purpose)
            if item is not None:
                h = items.canonical_hash(args.task, item["puzzle"])
                if h in seen:
                    item, reason = None, "duplicate"
                else:
                    seen.add(h)
            if item is None:
                cell_rejects[reason] += 1
                continue
            if item["answer_source"] == "solver":
                cell_rejects[f"kept_{item['reference_note']}"] += 1
            commit = item["commit"]
            out.append(item)
            kept += 1
        if kept < target:
            print(f"WARNING: cell {(size, level)} has only {kept}/{target} items after its whole seed block")
        summary_cells[f"{size},{level}"] = {"kept": kept, "region": items.region(args.task, size, level),
                                            "notes": dict(cell_rejects)}
        rejected.update(cell_rejects)
        print(f"cell size={size} level={level}: kept {kept}, {dict(cell_rejects)}")

    path = out_path(args.task, args.generator, args.purpose, args.size, args.out_dir)
    items.write_jsonl(path, out)
    summary = {
        "file": path.name, "task": args.task, "generator": args.generator, "purpose": args.purpose,
        "size": args.size, "items": len(out), "per_cell_target": target, "commit": commit,
        "seed_range": [start_seed, end_seed], "cells": summary_cells, "totals": dict(rejected),
        "unique_share": (sum(it["n_solutions"] == 1 for it in out) / len(out)
                         if out and args.task == "futoshiki" else None),
        "seconds": round(time.time() - t0, 1),
    }
    with open(path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {len(out)} items to {path} ({summary['seconds']} s)")


if __name__ == "__main__":
    main()
