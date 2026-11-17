"""Small exact solvers used while making items, never while scoring."""

from fractions import Fraction
from itertools import combinations

import config
import verify

# ---------------------------------------------------------------- Futoshiki


def solve_futoshiki(puzzle, limit=2):
    """Count solutions up to `limit` by backtracking. Returns (count, first solution or None).

    Picks the empty cell with the fewest candidates each step, so 7x7 boards stay fast.
    """
    n = puzzle["n"]
    grid = [[0] * n for _ in range(n)]
    for r, c, v in puzzle["givens"]:
        if grid[r][c] not in (0, v):
            return 0, None
        grid[r][c] = v
    less_than = {}   # cell -> cells it must be smaller than
    more_than = {}   # cell -> cells it must be larger than
    for a, b in puzzle["lt"]:
        less_than.setdefault(tuple(a), []).append(tuple(b))
        more_than.setdefault(tuple(b), []).append(tuple(a))

    def candidates(r, c):
        used = set(grid[r]) | {grid[i][c] for i in range(n)}
        options = []
        for v in range(1, n + 1):
            if v in used:
                continue
            if any(grid[x][y] and not v < grid[x][y] for x, y in less_than.get((r, c), ())):
                continue
            if any(grid[x][y] and not v > grid[x][y] for x, y in more_than.get((r, c), ())):
                continue
            options.append(v)
        return options

    # Givens must be consistent with each other before search.
    for r, c, v in puzzle["givens"]:
        grid[r][c] = 0
        ok = v in candidates(r, c)
        grid[r][c] = v
        if not ok:
            return 0, None

    count, first = 0, None

    def search():
        nonlocal count, first
        best, best_opts = None, None
        for r in range(n):
            for c in range(n):
                if grid[r][c] == 0:
                    opts = candidates(r, c)
                    if not opts:
                        return
                    if best is None or len(opts) < len(best_opts):
                        best, best_opts = (r, c), opts
        if best is None:
            count += 1
            if first is None:
                first = [row[:] for row in grid]
            return
        r, c = best
        for v in best_opts:
            grid[r][c] = v
            search()
            grid[r][c] = 0
            if count >= limit:
                return

    search()
    return count, first


# ---------------------------------------------------------------- Countdown


def _combine(a_vals, b_vals):
    """Every value reachable by one operation between a value from each side, with an expression."""
    out = {}
    for x, ex in a_vals.items():
        for y, ey in b_vals.items():
            out.setdefault(x + y, f"({ex} + {ey})")
            out.setdefault(x * y, f"({ex} * {ey})")
            out.setdefault(x - y, f"({ex} - {ey})")
            out.setdefault(y - x, f"({ey} - {ex})")
            if y != 0:
                out.setdefault(x / y, f"({ex} / {ey})")
            if x != 0:
                out.setdefault(y / x, f"({ey} / {ex})")
    return out


def solve_countdown(numbers, target, use_all=None):
    """One expression that reaches target exactly, or None.

    Builds the set of reachable values for every subset of the numbers (exact fractions allowed in
    intermediate steps, matching verify.check_countdown). For the full set it only looks up the
    value each split would need, so 6 numbers stay fast.
    """
    if use_all is None:
        use_all = config.COUNTDOWN_USE_ALL_NUMBERS
    numbers = [int(x) for x in numbers]   # numpy ints break Fraction hashing
    k = len(numbers)
    target = Fraction(int(target))
    full = (1 << k) - 1
    vals = {}
    for i, x in enumerate(numbers):
        vals[1 << i] = {Fraction(x): str(x)}

    def lookup(mask):
        if mask in vals:
            return vals[mask]
        result = {}
        members = [i for i in range(k) if mask >> i & 1]
        lowest = members[0]
        for size in range(1, len(members)):
            for left in combinations(members, size):
                if lowest not in left:   # each split once
                    continue
                lm = sum(1 << i for i in left)
                for v, e in _combine(lookup(lm), lookup(mask ^ lm)).items():
                    result.setdefault(v, e)
        vals[mask] = result
        return result

    def find_in(mask):
        """Reach target with exactly the numbers in mask, without building mask's full value set."""
        if mask & (mask - 1) == 0:
            return vals[mask].get(target)
        members = [i for i in range(k) if mask >> i & 1]
        lowest = members[0]
        for size in range(1, len(members)):
            for left in combinations(members, size):
                if lowest not in left:
                    continue
                lm = sum(1 << i for i in left)
                a, b = lookup(lm), lookup(mask ^ lm)
                for x, ex in a.items():
                    # For each operation, the value y the other side would need to hit the target.
                    options = [(target - x, "{ex} + {ey}"), (x - target, "{ex} - {ey}"), (x + target, "{ey} - {ex}")]
                    if x != 0:
                        options += [(target / x, "{ex} * {ey}"), (target * x, "{ey} / {ex}")]
                        if target != 0:
                            options.append((x / target, "{ex} / {ey}"))
                    for y, template in options:
                        ey = b.get(y)
                        if ey is not None:
                            return "(" + template.format(ex=ex, ey=ey) + ")"
        return None

    masks = [full] if use_all else sorted(range(1, full + 1), key=lambda m: -bin(m).count("1"))
    for mask in masks:
        expr = find_in(mask)
        if expr is not None:
            return _strip_outer(expr)
    return None


def _strip_outer(expr):
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for i, ch in enumerate(expr):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(expr) - 1:
                return expr
        return expr[1:-1]
    return expr


# ---------------------------------------------------------------- Reference check


def reference_passes(item):
    """True if the item's reference answer passes our own checker (verify.py)."""
    return bool(verify.check_item(item, item["answer"]))
