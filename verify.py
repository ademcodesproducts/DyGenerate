"""Our own checkers, used to score every response from every generator.

Scoring never calls a generator's own verifier, so it cannot differ by generator. The generators'
verifiers are only used in cross_check(), to find disagreements before the pilot runs.
"""

import contextlib
import copy
import importlib.util
import io
import random
import re
import sys
from fractions import Fraction

import config

# ---------------------------------------------------------------- Futoshiki


def check_futoshiki(puzzle, grid):
    """1 if grid solves the canonical puzzle, else 0."""
    n = puzzle["n"]
    if not isinstance(grid, list) or len(grid) != n or any(not isinstance(row, list) or len(row) != n for row in grid):
        return 0
    full = set(range(1, n + 1))
    if any(set(row) != full for row in grid):
        return 0
    if any({grid[r][c] for r in range(n)} != full for c in range(n)):
        return 0
    if any(grid[r][c] != v for r, c, v in puzzle["givens"]):
        return 0
    if any(not grid[a[0]][a[1]] < grid[b[0]][b[1]] for a, b in puzzle["lt"]):
        return 0
    return 1


# ---------------------------------------------------------------- Countdown

_TOKEN = re.compile(r"\s*(?:(\d+)|(.))")


class ExpressionError(ValueError):
    pass


def _tokenize(expr):
    tokens = []
    for number, other in _TOKEN.findall(expr):
        if number:
            tokens.append(("num", int(number)))
        elif other in "+-*/()":
            tokens.append(("op", other))
        elif other.strip():
            raise ExpressionError(f"character not allowed: {other!r}")
    return tokens


def evaluate_expression(expr):
    """Exact value (Fraction) and the integer literals used, from a small recursive-descent parser.

    Grammar: expr := term (('+'|'-') term)* ; term := unary (('*'|'/') unary)* ;
             unary := '-' unary | atom ; atom := number | '(' expr ')'
    Anything else raises ExpressionError. Python's eval is never used.
    """
    tokens = _tokenize(expr)
    if not tokens:
        raise ExpressionError("empty expression")
    pos = 0
    used = []

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def parse_expr():
        value = parse_term()
        while peek() in (("op", "+"), ("op", "-")):
            op = take()[1]
            rhs = parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def parse_term():
        value = parse_unary()
        while peek() in (("op", "*"), ("op", "/")):
            op = take()[1]
            rhs = parse_unary()
            if op == "*":
                value = value * rhs
            elif rhs == 0:
                raise ExpressionError("division by zero")
            else:
                value = value / rhs
        return value

    def parse_unary():
        if peek() == ("op", "-"):
            take()
            return -parse_unary()
        return parse_atom()

    def parse_atom():
        kind, val = peek()
        if kind == "num":
            take()
            used.append(val)
            return Fraction(val)
        if (kind, val) == ("op", "("):
            take()
            value = parse_expr()
            if peek() != ("op", ")"):
                raise ExpressionError("missing closing parenthesis")
            take()
            return value
        raise ExpressionError(f"unexpected token {val!r}")

    value = parse_expr()
    if pos != len(tokens):
        raise ExpressionError("trailing tokens")
    return value, used


def check_countdown(numbers, target, expr, use_all=None):
    """1 if expr reaches target exactly using the given numbers, else 0.

    use_all=True: every number exactly once (TinyZero and Reasoning Gym rule).
    use_all=False: each number at most once.
    """
    if use_all is None:
        use_all = config.COUNTDOWN_USE_ALL_NUMBERS
    if not isinstance(expr, str):
        return 0
    try:
        value, used = evaluate_expression(expr)
    except ExpressionError:
        return 0
    available = sorted(numbers)
    if use_all:
        if sorted(used) != available:
            return 0
    else:
        pool = list(available)
        for x in used:
            if x not in pool:
                return 0
            pool.remove(x)
    return int(value == target)


def check_item(item, answer):
    """Score a parsed answer against an item with our checker."""
    if answer is None:
        return 0
    if item["task"] == "futoshiki":
        return check_futoshiki(item["puzzle"], answer)
    if item["task"] == "countdown":
        return check_countdown(item["puzzle"]["numbers"], item["puzzle"]["target"], answer)
    raise ValueError(f"unknown task {item['task']}")


# ---------------------------------------------------------------- Cross-check against generator verifiers


def _broken_futoshiki(grid, rng):
    """A few deliberately wrong grids: one changed cell, two swapped cells in a row, a transposed grid."""
    n = len(grid)
    out = []
    g = copy.deepcopy(grid)
    r, c = rng.randrange(n), rng.randrange(n)
    g[r][c] = g[r][c] % n + 1
    out.append(g)
    g = copy.deepcopy(grid)
    r = rng.randrange(n)
    c1, c2 = rng.sample(range(n), 2)
    g[r][c1], g[r][c2] = g[r][c2], g[r][c1]
    out.append(g)
    out.append([list(col) for col in zip(*grid)])
    return out


def _broken_countdown(expr, numbers, rng):
    """A few deliberately wrong expressions: a changed number, a dropped number, a changed operator."""
    out = []
    literals = re.findall(r"\d+", expr)
    if literals:
        lit = rng.choice(literals)
        out.append(expr.replace(lit, str(int(lit) + 1), 1))
    out.append(f"({expr}) + {numbers[0]}")
    ops = [m.start() for m in re.finditer(r"[+\-*/]", expr)]
    if ops:
        i = rng.choice(ops)
        new = {"+": "-", "-": "+", "*": "+", "/": "*"}[expr[i]]
        out.append(expr[:i] + new + expr[i + 1:])
    return out


def _synlogic_verifier():
    sys.path.insert(0, str(config.SYNLOGIC_DIR))
    from base.data import Data
    from games.tasks.futoshiki.scripts.futoshiki_verifier import FutoshikiVerifier
    verifier = FutoshikiVerifier()

    def verdict(item, grid):
        native = item["native"]
        data = Data(question=native["question"], answer=native["answer"], metadata=native["metadata"])
        text = "[[" + ",".join(" ".join(str(v) for v in row) for row in grid) + "]]"
        with contextlib.redirect_stdout(io.StringIO()):  # SynLogic prints every failure reason
            return int(bool(verifier.verify(data, text)))
    return verdict


def _rg_verifier(task):
    import reasoning_gym
    dataset = reasoning_gym.create_dataset(task, size=1, seed=0)

    def verdict(item, answer):
        if task == "futoshiki":
            # Reasoning Gym's oracle text is the solved board in its own layout; render ours the same way.
            text = dataset._puzzle_to_string(answer, {((a, b), (c, d)): s for a, b, c, d, s in item["native"]["metadata"]["constraints"]})
        else:
            text = answer
        return int(dataset.score_answer(text, item["native"]) >= 0.999)
    return verdict


def _tinyzero_verifier():
    path = config.TINYZERO_DIR / "verl" / "utils" / "reward_score" / "countdown.py"
    spec = importlib.util.spec_from_file_location("tinyzero_countdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def verdict(item, expr):
        text = f"Assistant: <answer>{expr}</answer>"
        truth = {"target": item["puzzle"]["target"], "numbers": item["puzzle"]["numbers"]}
        with contextlib.redirect_stdout(io.StringIO()):  # TinyZero prints 1 in 64 calls
            return int(module.compute_score(text, truth) == 1.0)
    return verdict


def cross_check(items, per_generator=100, seed=0):
    """Compare our verdict with the generator's own verifier on correct and broken answers.

    Returns a list of disagreements (dicts). Every one must be explained before the pilot.
    """
    rng = random.Random(seed)
    by_gen = {}
    for item in items:
        by_gen.setdefault((item["task"], item["generator"]), []).append(item)
    factories = {
        ("futoshiki", "synlogic"): _synlogic_verifier,
        ("futoshiki", "reasoning_gym"): lambda: _rg_verifier("futoshiki"),
        ("countdown", "reasoning_gym"): lambda: _rg_verifier("countdown"),
        ("countdown", "tinyzero"): _tinyzero_verifier,
    }
    disagreements, checked = [], {}
    for key, group in by_gen.items():
        theirs = factories[key]()
        sample = rng.sample(group, min(per_generator, len(group)))
        checked[key] = 0
        for item in sample:
            ref = item["answer"]
            if item["task"] == "futoshiki":
                candidates = [ref] + _broken_futoshiki(ref, rng)
            else:
                candidates = [ref] + _broken_countdown(ref, item["puzzle"]["numbers"], rng)
            for cand in candidates:
                ours, other = check_item(item, cand), theirs(item, cand)
                checked[key] += 1
                if ours != other:
                    disagreements.append({"task": key[0], "generator": key[1], "id": item["id"],
                                          "answer": cand, "ours": ours, "theirs": other})
    return disagreements, checked
