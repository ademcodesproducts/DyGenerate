"""The checkers and the parser must be right before any model is scored with them."""

import copy

import parse
import solve
from verify import check_countdown, check_futoshiki, evaluate_expression, ExpressionError

import pytest

# A 4x4 puzzle and a hand-checked solution of it.
SOLUTION = [[1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]]
PUZZLE = {"n": 4, "givens": [[0, 0, 1], [1, 1, 4], [2, 2, 4], [3, 3, 1]],
          "lt": [[[0, 1], [0, 2]], [[2, 1], [2, 0]], [[3, 2], [3, 1]]]}


def test_futoshiki_solution_passes():
    assert check_futoshiki(PUZZLE, SOLUTION) == 1


def test_futoshiki_repeat_in_row_fails():
    g = copy.deepcopy(SOLUTION)
    g[0][3] = 3
    assert check_futoshiki(PUZZLE, g) == 0


def test_futoshiki_repeat_in_column_fails():
    # every row is a permutation, but every column repeats one number
    assert check_futoshiki({"n": 4, "givens": [], "lt": []}, [SOLUTION[0][:] for _ in range(4)]) == 0


def test_futoshiki_wrong_given_fails():
    puzzle = dict(PUZZLE, givens=PUZZLE["givens"] + [[0, 1, 3]])
    assert check_futoshiki(puzzle, SOLUTION) == 0


def test_futoshiki_broken_inequality_fails():
    puzzle = dict(PUZZLE, lt=PUZZLE["lt"] + [[[0, 3], [0, 2]]])   # 4 < 3 is false
    assert check_futoshiki(puzzle, SOLUTION) == 0


def test_futoshiki_wrong_size_fails():
    assert check_futoshiki(PUZZLE, [row[:3] for row in SOLUTION[:3]]) == 0
    assert check_futoshiki(PUZZLE, SOLUTION + [[1, 2, 3, 4]]) == 0
    assert check_futoshiki(PUZZLE, None) == 0


def test_futoshiki_solver_agrees():
    count, first = solve.solve_futoshiki(PUZZLE, limit=2)
    assert count >= 1 and check_futoshiki(PUZZLE, first) == 1


def test_countdown_correct_passes():
    assert check_countdown([25, 3, 7], 100, "(7 - 3) * 25") == 1


def test_countdown_number_twice_fails():
    assert check_countdown([25, 3, 7], 100, "(7 - 3) * 25 + 3 - 3") == 0
    assert check_countdown([25, 3, 7], 100, "(7 - 3) * 25 + 3 - 3", use_all=False) == 0


def test_countdown_number_not_given_fails():
    assert check_countdown([25, 3, 7], 100, "4 * 25") == 0


def test_countdown_missing_number_fails_under_use_all():
    assert check_countdown([25, 4, 7], 100, "4 * 25", use_all=True) == 0
    assert check_countdown([25, 4, 7], 100, "4 * 25", use_all=False) == 1


def test_countdown_misses_target_fails():
    assert check_countdown([25, 3, 7], 100, "(7 + 3) * 25") == 0


def test_countdown_division_is_exact():
    assert check_countdown([7, 2, 1], 3, "7 / 2 - 1") == 0        # 2.5, not 3
    assert check_countdown([7, 2, 1], 4, "(7 + 1) / 2") == 1
    assert check_countdown([1, 3, 3], 1, "(1 / 3) * 3") == 1       # exact fractions, no float error


@pytest.mark.parametrize("expr", ["__import__('os')", "2 ** 3", "7 % 2", "1e3", "3.5 + 1", "abs(-3)", "", "(1 + 2"])
def test_countdown_unsafe_or_bad_input_rejected(expr):
    assert check_countdown([1, 2, 3], 6, expr) == 0


def test_expression_parser_raises_on_letters():
    with pytest.raises(ExpressionError):
        evaluate_expression("1 + x")


def test_parser_takes_last_answer_block():
    text = "<answer>1 2\n2 1</answer> wait, no: <answer>\n2 1\n1 2\n</answer>"
    assert parse.parse("futoshiki", text, "stop", n=2) == ([[2, 1], [1, 2]], "ok")


def test_parser_statuses():
    assert parse.parse("countdown", "thinking...", "length") == (None, "truncated")
    assert parse.parse("countdown", "I think it is 5", "stop") == (None, "no_answer")
    assert parse.parse("countdown", "<answer>five</answer>", "stop") == (None, "malformed")
    assert parse.parse("futoshiki", "<answer>1 2 3</answer>", "stop", n=2) == (None, "malformed")


def test_parser_normalises_countdown_symbols():
    assert parse.parse("countdown", "<answer>(7 − 3) × 25 = 100</answer>", "stop") == ("(7 - 3) * 25", "ok")


def test_parser_tolerates_commas_and_one_line_grids():
    assert parse.parse("futoshiki", "<answer>1, 2\n2, 1</answer>", "stop", n=2)[0] == [[1, 2], [2, 1]]
    assert parse.parse("futoshiki", "<answer>1 2, 2 1</answer>", "stop", n=2)[0] == [[1, 2], [2, 1]]


def test_parser_fallback_is_off_by_default():
    text = "the answer is (7 - 3) * 25"
    assert parse.parse("countdown", text, "stop") == (None, "no_answer")
    assert parse.parse("countdown", text, "stop", fallback=True) == ("(7 - 3) * 25", "ok")
