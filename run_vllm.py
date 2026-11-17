"""Runs one model over one or more item files with vLLM and saves every raw output.

    python run_vllm.py --model qwen2.5-7b-instruct --items items/futoshiki_*_eval_smoke.jsonl --prompt canonical

Same script on Kaggle (T4 x2) and on the RTX 4060 (inside WSL2). Output goes to
responses/<model>/<item file stem>_<prompt version>.jsonl, appended after every batch; rerunning the
same command skips items that already have all their samples (Kaggle sessions end after 12 hours).
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import items
import prompts


def request_seed(item_id, base_seed):
    """A per-item seed, so a resumed run samples the same way as an uninterrupted one."""
    return (int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) + base_seed) % (2**31)


def done_ids(path, samples):
    if not path.exists():
        return set()
    counts = {}
    for rec in items.read_jsonl(path):
        counts[rec["item_id"]] = counts.get(rec["item_id"], 0) + 1
    return {i for i, c in counts.items() if c >= samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(config.MODELS))
    ap.add_argument("--items", required=True, nargs="+")
    ap.add_argument("--prompt", required=True, choices=config.PROMPT_VERSIONS)
    ap.add_argument("--samples", type=int, default=None, help="default: from the item file's size")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items per file (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=None, help="override the registry, e.g. if it does not fit")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    import vllm

    cfg = dict(config.MODELS[args.model])
    if args.max_model_len:
        cfg["max_model_len"] = args.max_model_len
    llm = LLM(model=cfg["hf_id"], revision=cfg.get("revision"), tokenizer_revision=cfg.get("revision"),
              dtype=cfg["dtype"], tensor_parallel_size=cfg["tensor_parallel_size"],
              max_model_len=cfg["max_model_len"], gpu_memory_utilization=cfg["gpu_memory_utilization"],
              seed=config.SAMPLING["seed"])
    tokenizer = llm.get_tokenizer()
    out_dir = config.RESPONSES_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.jsonl"

    for item_file in args.items:
        item_file = Path(item_file)
        size_name = item_file.stem.rsplit("_", 1)[-1]
        samples = args.samples or config.ITEM_COUNTS.get(size_name, {"samples": 1})["samples"]
        file_items = items.read_jsonl(item_file)
        if args.limit:
            file_items = file_items[:args.limit]
        out_path = out_dir / f"{item_file.stem}_{args.prompt}.jsonl"
        skip = done_ids(out_path, samples)
        todo = [it for it in file_items if it["id"] not in skip]
        print(f"{item_file.name}: {len(todo)} items to run ({len(skip)} already done), {samples} samples each")

        t0, gen_tokens, truncated, printed = time.time(), 0, 0, 0
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b:b + args.batch_size]
            texts, params = [], []
            for it in batch:
                text = prompts.format_for_model(it[f"prompt_{args.prompt}"], cfg, tokenizer)
                prompt_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                max_new = min(cfg["max_new_tokens"], cfg["max_model_len"] - prompt_len - 8)
                texts.append(text)
                params.append(SamplingParams(n=samples, temperature=config.SAMPLING["temperature"],
                                             top_p=config.SAMPLING["top_p"], max_tokens=max_new,
                                             seed=request_seed(it["id"], config.SAMPLING["seed"])))
            outputs = llm.generate(texts, params)
            with open(out_path, "a", encoding="utf-8") as f:
                for it, res, p in zip(batch, outputs, params):
                    for k, o in enumerate(res.outputs):
                        rec = {"item_id": it["id"], "model": args.model, "hf_id": cfg["hf_id"], "revision": cfg.get("revision"),
                               "prompt_version": args.prompt, "wrapper_version": it.get("wrapper_version"),
                               "item_file": item_file.name, "sample": k, "text": o.text,
                               "finish_reason": o.finish_reason, "gen_tokens": len(o.token_ids),
                               "max_new_tokens": p.max_tokens, "vllm": vllm.__version__}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        gen_tokens += rec["gen_tokens"]
                        truncated += o.finish_reason == "length"
                        if printed < 2:   # catch garbage output (fp16 on T4) early
                            print(f"--- first outputs [{it['id']}] finish={o.finish_reason}\n{o.text}\n---")
                            printed += 1
            print(f"  {min(b + args.batch_size, len(todo))}/{len(todo)} items, "
                  f"{gen_tokens / max(time.time() - t0, 1e-9):.0f} tok/s, {truncated} truncated")

        seconds = time.time() - t0
        log = {"time": datetime.now(timezone.utc).isoformat(), "model": args.model, "hf_id": cfg["hf_id"],
               "item_file": item_file.name, "prompt_version": args.prompt, "items": len(todo), "samples": samples,
               "seconds": round(seconds, 1), "gen_tokens": gen_tokens,
               "tokens_per_second": round(gen_tokens / seconds, 1) if seconds > 0 else None,
               "truncated": truncated, "machine": cfg["machine"], "vllm": vllm.__version__}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        print(json.dumps(log))


if __name__ == "__main__":
    main()
