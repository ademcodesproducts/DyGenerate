"""Converting each generator's native puzzle into canonical form must lose nothing.

Needs the generators: Reasoning Gym installed, and the SynLogic clone at config.SYNLOGIC_DIR.
Tests for a missing generator are skipped, not failed.
"""

import importlib.util

import pytest

import config
import gen_countdown
import gen_futoshiki
import items
import make_items
import prompts
import verify

HAS_RG = importlib.util.find_spec("reasoning_gym") is not None
HAS_SYNLOGIC = config.SYNLOGIC_DIR.exists()
needs_rg = pytest.mark.skipif(not HAS_RG, reason="reasoning_gym not installed")
needs_synlogic = pytest.mark.skipif(not HAS_SYNLOGIC, reason="SynLogic clone not found")

CASES = [
    pytest.param("futoshiki", "synlogic", marks=needs_synlogic),
    pytest.param("futoshiki", "reasoning_gym", marks=needs_rg),
    pytest.param("countdown", "tinyzero"),
    pytest.param("countdown", "reasoning_gym", marks=needs_rg),
]


@pytest.mark.parametrize("task,generator", CASES)
def test_kept_items_pass_our_checker(task, generator):
    """Every item make_items keeps has a reference answer that passes verify.py."""
    kept = 0
    for seed in range(60):
        size, level = make_items.cells(task)[seed % len(make_items.cells(task))]
        item, _ = make_items.make_one(task, generator, size, level, seed, "eval")
        if item is not None:
            kept += 1
            assert verify.check_item(item, item["answer"]) == 1
    assert kept > 0


@needs_rg
def test_reasoning_gym_reference_passes_without_solver():
    """Reasoning Gym's own solution must pass directly; if not, the conversion is wrong."""
    for seed in range(30):
        for size in (4, 5, 6):
            raw = gen_futoshiki.generate("reasoning_gym", size, seed % 2, seed)
            assert verify.check_futoshiki(raw["puzzle"], raw["answer"]) == 1
        raw = gen_countdown.generate("reasoning_gym", 4, 0, seed)
        assert verify.check_countdown(raw["puzzle"]["numbers"], raw["puzzle"]["target"], raw["answer"]) == 1


@needs_rg
def test_reasoning_gym_inequality_direction_hand_checked():
    """Each canonical lt pair matches the symbol drawn between those two cells in the native board."""
    raw = gen_futoshiki.generate("reasoning_gym", 5, 1, 3)
    lines = raw["native"]["question"].split("\n\n")[1].split("\n")
    for (r1, c1), (r2, c2) in raw["puzzle"]["lt"]:
        if r1 == r2:   # horizontal: the symbol sits between the two cells on row line 2*r
            symbol = lines[2 * r1][4 * min(c1, c2) + 2]
            assert symbol == ("<" if c1 < c2 else ">")
        else:          # vertical: the symbol sits on the line between the rows, under the column
            symbol = lines[2 * min(r1, r2) + 1][4 * c1]
            assert symbol == ("∧" if r1 < r2 else "∨")


@needs_synlogic
def test_synlogic_inequality_direction_hand_checked():
    """Each native '(r,c) > (r,c)' line (1-indexed) maps to the right 0-indexed lt pair."""
    raw = gen_futoshiki.generate("synlogic", 5, 0, 5)
    text = raw["native"]["question"]
    block = text.split("Inequality constraints:\n")[1].split("\n\n")[0]
    expected = set()
    for line in block.splitlines():
        left, sign, right = line.split(" ")
        a = tuple(int(x) - 1 for x in left.strip("()").split(","))
        b = tuple(int(x) - 1 for x in right.strip("()").split(","))
        expected.add((a, b) if sign == "<" else (b, a))
    got = {(tuple(a), tuple(b)) for a, b in raw["puzzle"]["lt"]}
    assert got == expected


@pytest.mark.parametrize("task,generator", CASES)
def test_generation_is_reproducible(task, generator):
    size, level = make_items.cells(task)[0]
    a = make_items.ADAPTERS[task].generate(generator, size, level, 12345)
    b = make_items.ADAPTERS[task].generate(generator, size, level, 12345)
    assert a["puzzle"] == b["puzzle"] and a["native_text"] == b["native_text"]


def test_seed_ranges_do_not_overlap():
    ranges = sorted(config.SEED_RANGES.values())
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
        assert a1 <= b0
    for start, end in ranges:
        n_cells = max(len(make_items.cells(t)) for t in config.TASKS)
        assert start + n_cells * config.SEEDS_PER_CELL <= end


@pytest.mark.parametrize("task,generator", CASES)
def test_purposes_never_share_a_canonical_hash(task, generator):
    size, level = make_items.cells(task)[0]
    hashes = {}
    for purpose, (start, _) in config.SEED_RANGES.items():
        hashes[purpose] = set()
        for seed in range(start, start + 30):
            raw = make_items.ADAPTERS[task].generate(generator, size, level, seed)
            hashes[purpose].add(items.canonical_hash(task, raw["puzzle"]))
    # Tiny puzzles can collide by chance; make_items drops those as duplicates. Here, collisions must be rare.
    overlap = (hashes["eval"] & hashes["train"]) | (hashes["eval"] & hashes["calib"]) | (hashes["train"] & hashes["calib"])
    assert len(overlap) <= 2


@pytest.mark.skipif(not config.TRAINING_SETS["countdown"]["path"].exists(), reason="training set not downloaded")
def test_known_training_row_is_in_the_exclusion_set():
    import training_sets
    hashes = training_sets.training_hashes("countdown")
    assert len(hashes) > 400_000
    # first row of Countdown-Tasks-3to4, numbers in a different order: same canonical puzzle
    assert items.canonical_hash("countdown", items.canonical_countdown([35, 44, 19], 98)) in hashes


def test_canonical_prompt_is_generator_independent():
    puzzle = items.canonical_futoshiki(4, [(0, 0, 1)], [((0, 1), (0, 2))])
    a = prompts.build_prompts({"task": "futoshiki", "puzzle": puzzle, "native_text": "A"})
    b = prompts.build_prompts({"task": "futoshiki", "puzzle": puzzle, "native_text": "B"})
    assert a["prompt_canonical"] == b["prompt_canonical"]
    assert a["prompt_native"] != b["prompt_native"]
