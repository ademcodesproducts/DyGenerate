"""Joins raw responses with their items, parses each answer and marks it correct with our own checker.

    python score.py            # all of responses/ against all of items/
    python score.py --fallback # also try untagged answers (report both if this is ever used)

Output: results/scores.csv, one row per sample, and a sanity report printed to the screen.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import config
import items
import parse
import verify


def load_items(items_dir):
    """Items by ID, and the set of item IDs in each item file."""
    by_id, ids_by_file = {}, {}
    for path in Path(items_dir).glob("*.jsonl"):
        file_items = items.read_jsonl(path)
        ids_by_file[path.name] = {it["id"] for it in file_items}
        for it in file_items:
            by_id[it["id"]] = it
    return by_id, ids_by_file


def score_rows(responses_dir, item_by_id, fallback=False):
    rows, missing = [], Counter()
    for path in sorted(Path(responses_dir).glob("*/*.jsonl")):
        if path.name == "run_log.jsonl":
            continue
        for rec in items.read_jsonl(path):
            it = item_by_id.get(rec["item_id"])
            if it is None:
                missing[path.name] += 1
                continue
            model = config.MODELS.get(rec["model"], {})
            n = it["puzzle"].get("n")
            answer, status = parse.parse(it["task"], rec["text"], rec["finish_reason"], n=n, fallback=fallback)
            home = config.TASKS[it["task"]]["home"]
            rows.append({
                "model": rec["model"], "role": model.get("role"), "task": it["task"],
                "generator": it["generator"], "home_away": "home" if it["generator"] == home else "away",
                "region": it["region"], "size": it["params"]["size"], "level": it["params"]["level"],
                "prompt_version": rec["prompt_version"], "wrapper_version": rec.get("wrapper_version"),
                "item_file": rec.get("item_file"), "item_id": it["id"], "sample": rec["sample"],
                "correct": verify.check_item(it, answer), "parse_status": status,
                "gen_tokens": rec["gen_tokens"], "n_solutions": it.get("n_solutions"),
            })
    return pd.DataFrame(rows), missing


def sanity_report(df, ids_by_file, missing):
    print(f"{len(df)} scored samples")
    if missing:
        print(f"WARNING: responses whose item is not in items/: {dict(missing)}")
    if df.empty:
        return
    print("\nRows per model and generator:")
    print(df.groupby(["model", "generator", "prompt_version"]).size().unstack(fill_value=0))
    print("\nParse status shares:")
    print(df.groupby("model")["parse_status"].value_counts(normalize=True).unstack(fill_value=0).round(3))
    for (model, version, item_file), group in df.groupby(["model", "prompt_version", "item_file"]):
        without = len(ids_by_file.get(item_file, set()) - set(group["item_id"]))
        if without:
            print(f"note: {model} / {version} / {item_file}: {without} items have no responses")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items-dir", default=config.ITEMS_DIR)
    ap.add_argument("--responses-dir", default=config.RESPONSES_DIR)
    ap.add_argument("--out", default=config.RESULTS_DIR / "scores.csv")
    ap.add_argument("--fallback", action="store_true")
    args = ap.parse_args()

    item_by_id, ids_by_file = load_items(args.items_dir)
    df, missing = score_rows(args.responses_dir, item_by_id, fallback=args.fallback)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    sanity_report(df, ids_by_file, missing)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
