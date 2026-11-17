"""Turns a model's raw output into a final answer, with one rule for every model and generator.

Every output gets a status:
  ok         an answer was parsed
  truncated  the model hit the token limit (finish reason "length"); if a complete answer block came
             before the cut it is still parsed and scored, so truncation is visible but not double-counted
  no_answer  finished but gave no answer tags
  malformed  tags present but the content could not be parsed
"""

import re

ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
# Characters allowed in a Futoshiki answer besides digits: separators, and inequality symbols a model
# may copy from the puzzle layout (stripped before reading numbers).
_GRID_NOISE = re.compile(r"[,\s<>∧∨^|]+")
_GRID_ALLOWED = re.compile(r"^[\d,\s<>∧∨^|\[\]]*$")
_SYMBOLS = {"×": "*", "÷": "/", "−": "-", "–": "-", "·": "*",
            "\\times": "*", "\\div": "/", "\\cdot": "*", "\\left": "", "\\right": ""}
_EXPR_ALLOWED = re.compile(r"^[\d+\-*/()\s]+$")


def last_answer_block(text):
    blocks = ANSWER_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def parse_futoshiki_block(block, n):
    """n lines of n integers; commas and extra spaces tolerated; one comma-separated line accepted.

    Brackets are allowed and dropped: SynLogic trained its model to answer "[[1 2 3,2 3 1,3 1 2]]".
    """
    block = block.strip().strip("`").strip()
    if not _GRID_ALLOWED.match(block):
        return None
    block = block.replace("[", " ").replace("]", " ").strip()
    lines = [ln for ln in block.splitlines() if re.search(r"\d", ln)]
    if len(lines) == 1 and "," in lines[0]:
        lines = lines[0].split(",")
        if len(lines) != n:
            return None
    grid = []
    for ln in lines:
        tokens = [t for t in _GRID_NOISE.split(ln) if t]
        if len(tokens) == n and all(t.isdigit() for t in tokens):
            grid.append([int(t) for t in tokens])
        elif len(tokens) == 1 and len(tokens[0]) == n and n < 10:
            grid.append([int(ch) for ch in tokens[0]])   # "1234" style rows
        else:
            return None
    return grid if len(grid) == n else None


def parse_countdown_block(block):
    expr = block.strip().strip("`").strip()
    for sym, rep in _SYMBOLS.items():
        expr = expr.replace(sym, rep)
    expr = expr.replace("$", "").replace("\\(", "").replace("\\)", "")
    if "=" in expr:   # tolerate "expr = target"
        expr = expr.split("=")[0]
    expr = expr.strip().rstrip(".").strip()
    if not expr or not _EXPR_ALLOWED.match(expr):
        return None
    return expr


def _fallback(task, text, n):
    """Off by default: the last grid or expression in the text when there are no tags."""
    if task == "futoshiki":
        lines = text.splitlines()
        for end in range(len(lines), n - 1, -1):
            grid = parse_futoshiki_block("\n".join(lines[end - n:end]), n)
            if grid:
                return grid
        return None
    candidates = re.findall(r"[\d(][\d+\-*/()\s]*[\d)]", text)
    for cand in reversed(candidates):
        if re.search(r"[+\-*/]", cand):
            return cand.strip()
    return None


def parse(task, text, finish_reason, n=None, fallback=False):
    """Returns (answer, status). answer is a grid (Futoshiki), an expression string (Countdown) or None."""
    truncated = finish_reason == "length"
    block = last_answer_block(text)
    if block is None:
        answer = _fallback(task, text, n) if fallback else None
        if answer is not None:
            return answer, "truncated" if truncated else "ok"
        return None, "truncated" if truncated else "no_answer"
    answer = parse_futoshiki_block(block, n) if task == "futoshiki" else parse_countdown_block(block)
    if truncated:
        return answer, "truncated"
    return answer, "ok" if answer is not None else "malformed"
