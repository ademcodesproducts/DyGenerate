# Evaluation harness (Exp 0 and Exp 1)

One pipeline turns puzzle generators into scored model answers, so every model is judged on the same items, the same prompt wrapper and the same checker. Arul pulls training draws from it (Exp 3), Rishbha builds the gold rungs on it, and Exp 1 runs on it first.

**Exp 1 asks:** does SynLogic-7B (Futoshiki) or a 3B Countdown RL model show a home advantage over an independent generator? The advantage has to exceed what their parents and the broad models show. That comparison is the difference in differences.

## Setup

On Kaggle or inside WSL2 (vLLM does not run on native Windows):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm==<version>          # see requirements.txt
git clone https://github.com/MiniMax-AI/SynLogic.git external/SynLogic
git -C external/SynLogic checkout d8c527fd17edb739172619efb9b681805fc74b8d
python -m pytest tests -q            # must pass before anything is scored
```

## Workflow

```bash
# 1. Items (one file per task x generator x purpose x size; commit the item files)
python make_items.py --task futoshiki --generator synlogic      --purpose eval --size pilot
python make_items.py --task futoshiki --generator reasoning_gym --purpose eval --size pilot
python make_items.py --task countdown --generator tinyzero      --purpose eval --size pilot
python make_items.py --task countdown --generator reasoning_gym --purpose eval --size pilot

# 2. Cross-check our checkers against the generators' own verifiers (expect 0 disagreements)
python -c "import glob,items,verify; its=[i for f in glob.glob('items/*_eval_pilot.jsonl') for i in items.read_jsonl(f)]; print(verify.cross_check(its))"

# 3. Inference. Kaggle (T4 x2), Futoshiki 7B models; resumes if rerun
python run_vllm.py --model qwen2.5-7b-instruct --items items/futoshiki_*_eval_smoke.jsonl --prompt canonical --limit 5
python run_vllm.py --model synlogic-7b --items items/futoshiki_*_eval_pilot.jsonl --prompt canonical
#    RTX 4060 (WSL2), Countdown 3B models
python run_vllm.py --model qwen2.5-3b-instruct --items items/countdown_*_eval_pilot.jsonl --prompt canonical

# 4. Score and analyze
python score.py
python analyze.py      # writes results/exp1_summary.md, tables and plots
```

Run each model on both prompt versions (`native`, `canonical`). The gap between them separates surface wording from puzzle structure.

## Generators used

| Task | Home | Away | Version |
|---|---|---|---|
| Futoshiki | SynLogic | Reasoning Gym | SynLogic `d8c527f`, reasoning-gym 0.1.25 |
| Countdown | TinyZero-style (reimplemented, see below) | Reasoning Gym | TinyZero `95df88f`, reasoning-gym 0.1.25 |

Every item records its generator version in `commit`.

## Training range B

- **Futoshiki:** sizes 4 and 5, both levels. This is a provisional guess: SynLogic only ships 4×4 and 5×5 data. Confirm it from the SynLogic-7B dataset card (docs/lookups.md), then set `range_b_confirmed` in config.py.
- **Countdown:** 3 or 4 numbers. Countdown-Tasks-3to4 uses numbers 1–99 and targets 10–100, checked on 300 rows.

## Known limitations

- **SynLogic picks inequality signs at random**, so they are not read off the solution.
  - Most raw puzzles are unsolvable at the larger settings; those are dropped.
  - The generator's reference grid often breaks its own inequalities, so kept items get a reference from our solver (`answer_source: "solver"`).
  - Many puzzles have more than one solution (`n_solutions`; about 16% are unique in the pilot). Our checker accepts any valid grid, so scoring is unaffected.
  - The kept items are therefore a filtered slice of SynLogic's distribution. Say so in the paper.
- **The Countdown home adapter is a reimplementation:** fresh uniform draws in the training ranges, kept only if solvable. TinyZero does not include the script that made Countdown-Tasks-3to4, and its target distribution may not be uniform.
- **Rule matching:**
  - Countdown requires every number to be used exactly once, as both TinyZero's reward and Reasoning Gym's scorer do.
  - Futoshiki answers are any valid grid, as SynLogic's verifier allows.
- **fp16 on T4:** there's no bf16, so check the first outputs that run_vllm.py prints for garbage.
- **Truncation:** truncated outputs are kept and reported. A complete answer before the cut still counts.
- **Statistical power:** at pilot size, a 20-point difference in differences shows up clearly on Futoshiki (4 in-range cells) but not on Countdown (2 in-range cells). Countdown needs the final size.

## Layout

| Path | Contents |
|---|---|
| config.py | every setting in one place |
| items.py | item schema and canonical puzzle formats |
| gen_futoshiki.py, gen_countdown.py | adapters around existing generators |
| prompts.py | the shared instruction wrapper and the two prompt versions |
| solve.py | solution counting (uniqueness, solvability) |
| verify.py | our own checkers, used for all scoring, plus the cross-check |
| parse.py | extracting the final answer from a model's output |
| make_items.py | writes item files |
| run_vllm.py | runs a model over item files |
| score.py | joins answers with items and marks each correct or not |
| analyze.py | accuracy tables, home advantage, difference in differences, ranking flips, plots |
| tests/ | checks for the checkers and adapters |
| docs/ | setup notes and the lookups still to do |
| items/, responses/, results/ | data folders (see the README in each) |

**Done when:** item files exist, there's one accuracy-vs-difficulty plot per task, and we have the first home-advantage and difference-in-differences numbers with intervals, plus truncation rates. That result is the Gate A decision on Nov 23.
