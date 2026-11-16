"""Turns results/scores.csv into the Exp 1 numbers and plots that decide Gate A.

    python analyze.py

Definitions (all per task and prompt version):
  item accuracy   mean over an item's samples
  cell accuracy   mean over a cell's items (cell = generator x size x level)
  G1              home generator, inside range B        (what dynamic benchmarks report today)
  G2              home generator, beyond range B
  G4              away generator(s), every cell
  gold            G2 and G4 cells pooled
  Set accuracies weight every cell equally.
  home advantage  mean over in-range cells of (home cell - away cell), same size and level
  DiD             home advantage of the exposed model minus that of its parent / each broad model
Uncertainty: bootstrap over items within each cell (2000 reps); an item's samples move together, and
the same resampled items are used for every model, so differences between models are paired.
"No difference" needs the 90% interval inside +/- EQUIVALENCE_MARGIN (an equivalence test), never
just a 95% interval that crosses zero.
"""

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean over empty slices


def item_level(df):
    keys = ["model", "role", "task", "generator", "home_away", "region", "size", "level", "prompt_version", "item_id"]
    return df.groupby(keys, dropna=False)["correct"].mean().rename("acc").reset_index()


def cell_table(items_df):
    keys = ["model", "role", "task", "generator", "home_away", "region", "size", "level", "prompt_version"]
    return items_df.groupby(keys, dropna=False)["acc"].agg(acc="mean", n_items="count").reset_index()


# ---------------------------------------------------------------- bootstrap engine


def cell_matrices(sub, models):
    """(generator, size, level) -> items x models matrix of item accuracies (NaN where a model has no answer)."""
    mats = {}
    for key, g in sub.groupby(["generator", "size", "level"]):
        wide = g.pivot_table(index="item_id", columns="model", values="acc", aggfunc="mean")
        mats[key] = wide.reindex(columns=models).to_numpy(dtype=float)
    return mats


def bootstrap_cell_means(mats, reps, rng):
    point = {k: np.nanmean(m, axis=0) for k, m in mats.items()}
    boot = {}
    for k, m in mats.items():
        idx = rng.integers(0, len(m), size=(reps, len(m)))
        boot[k] = np.nanmean(m[idx], axis=1)   # reps x models
    return point, boot


def set_mean(means, keys):
    arrs = [means[k] for k in keys if k in means]
    return np.nanmean(np.stack(arrs), axis=0) if arrs else None


def item_sets(task, cells):
    spec = config.TASKS[task]
    home, b = spec["home"], spec["range_b"]
    sets = {
        "G1": [k for k in cells if k[0] == home and (k[1], k[2]) in b],
        "G2": [k for k in cells if k[0] == home and (k[1], k[2]) not in b],
        "G4": [k for k in cells if k[0] != home],
    }
    sets["gold"] = sets["G2"] + sets["G4"]
    return sets


def home_advantage(means, task):
    spec = config.TASKS[task]
    diffs = []
    for size, level in sorted(spec["range_b"]):
        hk = (spec["home"], size, level)
        for away in spec["generators"]:
            ak = (away, size, level)
            if away != spec["home"] and hk in means and ak in means:
                diffs.append(means[hk] - means[ak])
    return np.nanmean(np.stack(diffs), axis=0) if diffs else None


def interval(boot, level=95):
    lo = (100 - level) / 2
    return np.nanpercentile(boot, [lo, 100 - lo], axis=0)


def verdict(point, ci95, ci90, margin):
    if ci95[0] > 0:
        return "positive"
    if ci95[1] < 0:
        return "negative"
    if -margin <= ci90[0] and ci90[1] <= margin:
        return f"equivalent to zero (within +/-{margin * 100:.0f} pts)"
    return "inconclusive"


# ---------------------------------------------------------------- analysis per task and prompt version


def analyze_subset(sub, task, reps, rng, margin):
    models = sorted(sub["model"].unique())
    roles = {m: config.MODELS.get(m, {}).get("role", "?") for m in models}
    mats = cell_matrices(sub, models)
    point, boot = bootstrap_cell_means(mats, reps, rng)
    sets = item_sets(task, list(mats))
    out = {"sets": [], "ha": [], "did": [], "flips": []}

    set_point, set_boot = {}, {}
    for name, keys in sets.items():
        p, bt = set_mean(point, keys), set_mean(boot, keys)
        if p is None:
            continue
        set_point[name], set_boot[name] = p, bt
        ci = interval(bt)
        for j, m in enumerate(models):
            out["sets"].append({"model": m, "role": roles[m], "set": name, "cells": len(keys),
                                "acc": p[j], "ci_low": ci[0, j], "ci_high": ci[1, j]})

    ha_p, ha_b = home_advantage(point, task), home_advantage(boot, task)
    if ha_p is not None:
        ci = interval(ha_b)
        for j, m in enumerate(models):
            out["ha"].append({"model": m, "role": roles[m], "home_advantage": ha_p[j],
                              "ci_low": ci[0, j], "ci_high": ci[1, j]})
        for e in [j for j, m in enumerate(models) if roles[m] == "exposed"]:
            for o, m in enumerate(models):
                if o == e:
                    continue
                d_p, d_b = ha_p[e] - ha_p[o], ha_b[:, e] - ha_b[:, o]
                c95, c90 = interval(d_b, 95), interval(d_b, 90)
                out["did"].append({"exposed": models[e], "versus": m, "versus_role": roles[m], "did": d_p,
                                   "ci95_low": c95[0], "ci95_high": c95[1], "ci90_low": c90[0], "ci90_high": c90[1],
                                   "verdict": verdict(d_p, c95, c90, margin)})

    if "G1" in set_point and "gold" in set_point:
        g1p, gdp, g1b, gdb = set_point["G1"], set_point["gold"], set_boot["G1"], set_boot["gold"]
        for a, b in itertools.combinations(range(len(models)), 2):
            swapped = np.sign(g1p[a] - g1p[b]) * np.sign(gdp[a] - gdp[b]) < 0
            boot_swap = np.nanmean(np.sign(g1b[:, a] - g1b[:, b]) * np.sign(gdb[:, a] - gdb[:, b]) < 0)
            out["flips"].append({"model_a": models[a], "model_b": models[b], "G1_a": g1p[a], "G1_b": g1p[b],
                                 "gold_a": gdp[a], "gold_b": gdp[b], "swapped": bool(swapped),
                                 "swap_share_in_bootstrap": boot_swap})
    return out


def plot_task(cells, task, version, out_dir):
    sub = cells[(cells["task"] == task) & (cells["prompt_version"] == version)]
    if sub.empty:
        return None
    spec = config.TASKS[task]
    in_sizes = sorted({s for s, _ in spec["range_b"]})
    by_size = sub.groupby(["model", "home_away", "size"])["acc"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, side in zip(axes, ["home", "away"]):
        part = by_size[by_size["home_away"] == side]
        for model, g in part.groupby("model"):
            ax.plot(g["size"], g["acc"], marker="o", label=f"{model} ({config.MODELS.get(model, {}).get('role', '?')})")
        if in_sizes:
            ax.axvspan(min(in_sizes) - 0.5, max(in_sizes) + 0.5, color="grey", alpha=0.15, label="training range B")
        gens = spec["home"] if side == "home" else ", ".join(g for g in spec["generators"] if g != spec["home"])
        ax.set_title(f"{side}: {gens}")
        ax.set_xlabel("board size" if task == "futoshiki" else "number count")
        ax.set_xticks(spec["sizes"])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{task} - {version} prompt")
    fig.tight_layout()
    path = Path(out_dir) / f"plot_{task}_{version}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def pct(x):
    return f"{x * 100:+.1f}" if pd.notna(x) else "n/a"


def write_summary(results, path):
    lines = ["# Exp 1 summary (generated by analyze.py)", ""]
    for (task, version), res in results.items():
        lines.append(f"## {task}, {version} prompt")
        if not config.TASKS[task]["range_b_confirmed"]:
            lines.append("_Training range B is not confirmed yet (docs/lookups.md); G1/G2 split is provisional._")
        lines.append("")
        if res["ha"]:
            lines.append("Home advantage (home minus away accuracy inside range B, points, 95% CI):")
            for r in res["ha"]:
                lines.append(f"- {r['model']} ({r['role']}): {pct(r['home_advantage'])} [{pct(r['ci_low'])}, {pct(r['ci_high'])}]")
            lines.append("")
        if res["did"]:
            lines.append("Difference in differences (exposed minus comparison):")
            for r in res["did"]:
                lines.append(f"- {r['exposed']} vs {r['versus']} ({r['versus_role']}): {pct(r['did'])} "
                             f"[{pct(r['ci95_low'])}, {pct(r['ci95_high'])}] -> {r['verdict']}")
            lines.append("")
        swaps = [r for r in res["flips"] if r["swapped"]]
        lines.append(f"Ranking swaps between G1 and gold: {len(swaps)} of {len(res['flips'])} pairs")
        for r in swaps:
            lines.append(f"- {r['model_a']} vs {r['model_b']} (swapped in {r['swap_share_in_bootstrap'] * 100:.0f}% of bootstrap reps)")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=config.RESULTS_DIR / "scores.csv")
    ap.add_argument("--out-dir", default=config.RESULTS_DIR)
    ap.add_argument("--reps", type=int, default=config.BOOTSTRAP_REPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    items_df = item_level(df)
    cells = cell_table(items_df)
    cells.to_csv(out_dir / "accuracy_by_cell.csv", index=False)

    status = (df.groupby(["model", "task", "generator", "size", "level", "prompt_version"])["parse_status"]
              .value_counts(normalize=True).unstack(fill_value=0).reset_index())
    status.to_csv(out_dir / "parse_status.csv", index=False)

    rng = np.random.default_rng(args.seed)
    results, tables = {}, {"sets": [], "ha": [], "did": [], "flips": []}
    for (task, version), sub in items_df.groupby(["task", "prompt_version"]):
        res = analyze_subset(sub, task, args.reps, rng, config.EQUIVALENCE_MARGIN)
        results[(task, version)] = res
        for name, rows in res.items():
            tables[name] += [{"task": task, "prompt_version": version, **r} for r in rows]
        plot_task(cells, task, version, out_dir)

    for name, file in [("sets", "item_sets.csv"), ("ha", "home_advantage.csv"), ("did", "did.csv"),
                       ("flips", "ranking_flips.csv")]:
        pd.DataFrame(tables[name]).to_csv(out_dir / file, index=False)
    write_summary(results, out_dir / "exp1_summary.md")
    print((out_dir / "exp1_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
