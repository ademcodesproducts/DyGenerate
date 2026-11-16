"""Builds the exact text each model sees.

Every generator and every model gets the same instruction wrapper and answer format, so a home
advantage cannot come from friendlier instructions. Two versions per item:
  native:    the generator's own puzzle text (answer-format lines removed) inside the shared wrapper
  canonical: the puzzle re-rendered from its canonical form, identical whichever generator made it
The difference between the two splits a home advantage into surface wording and puzzle structure.
"""

WRAPPER_VERSION = "w1"

FUTOSHIKI_WRAPPER = """{puzzle}

Solve the puzzle. Think it through step by step, then give your final grid inside <answer> </answer> tags as {n} lines of {n} space-separated numbers. For example, a solved 3x3 grid would be written:
<answer>
1 2 3
3 1 2
2 3 1
</answer>"""

COUNTDOWN_WRAPPER = """{puzzle}

Use every given number exactly once, with only + - * / and parentheses. Think it through step by step, then give one expression inside <answer> </answer> tags, without "=" or the target. For example: <answer>(1 + 2) * 3</answer>"""

# Base models (the parents) get the same text inside one plain conversation frame, ending where the
# model's turn starts. The same frame for every base model; no system prompts anywhere.
PLAIN_FRAME = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
               "User: {text}\nAssistant:")


def canonical_futoshiki_text(puzzle):
    n = puzzle["n"]
    grid = [["_"] * n for _ in range(n)]
    for r, c, v in puzzle["givens"]:
        grid[r][c] = str(v)
    rows = "\n".join(" ".join(row) for row in grid)
    if puzzle["lt"]:
        lines = "\n".join(f"(row {a[0] + 1}, column {a[1] + 1}) < (row {b[0] + 1}, column {b[1] + 1})"
                          for a, b in puzzle["lt"])
    else:
        lines = "(none)"
    return (f"This is a {n}x{n} Futoshiki puzzle. Fill the grid so that every row and every column contains "
            f"each number from 1 to {n} exactly once. Some cells are given; blanks are shown as _. "
            f"Every inequality below must also hold. Rows and columns are numbered from 1, starting top left.\n\n"
            f"Grid:\n{rows}\n\nInequalities:\n{lines}")


def canonical_countdown_text(puzzle):
    numbers = ", ".join(str(x) for x in puzzle["numbers"])
    return f"Numbers: {numbers}\nTarget: {puzzle['target']}\nCombine the numbers to reach the target exactly."


def wrap(task, puzzle_text, puzzle):
    if task == "futoshiki":
        return FUTOSHIKI_WRAPPER.format(puzzle=puzzle_text, n=puzzle["n"])
    if task == "countdown":
        return COUNTDOWN_WRAPPER.format(puzzle=puzzle_text)
    raise ValueError(task)


def build_prompts(item):
    """Fill prompt_native and prompt_canonical on an item (the user-message text, before model formatting)."""
    task, puzzle = item["task"], item["puzzle"]
    canonical = canonical_futoshiki_text(puzzle) if task == "futoshiki" else canonical_countdown_text(puzzle)
    item["prompt_native"] = wrap(task, item["native_text"], puzzle)
    item["prompt_canonical"] = wrap(task, canonical, puzzle)
    item["wrapper_version"] = WRAPPER_VERSION
    return item


def format_for_model(text, model_cfg, tokenizer=None):
    """Chat models: the model's own chat template around one user message. Base models: the plain frame."""
    if model_cfg["format"] == "chat":
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return PLAIN_FRAME.format(text=text)
