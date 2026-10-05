import random

import pytest
from py_libraries.type.string import isValidParenthese


def reduceBrackets(s: str) -> str:
    """Remove the adjacent matching pairs until none is left."""
    previous = None
    while previous != s:
        previous = s
        for pair in ('()', '[]', '{}'):
            s = s.replace(pair, '')
    return s


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('()', True),
        ('()[]{}', True),
        ('{[]}', True),
        ('', True),
        ('(]', False),
        ('([)]', False),
        ('(', False),
        (')', False),
        ('((', False),
        (')(', False),
    ],
)
def test_isValidParenthese_known_values(s: str, expected: bool) -> None:
    assert isValidParenthese(s) is expected


def test_isValidParenthese_ignores_other_characters() -> None:
    assert isValidParenthese('a(b[c]d)e')
    assert not isValidParenthese('a(b[c)d]e')


def test_isValidParenthese_custom_brackets() -> None:
    assert isValidParenthese('<<>>', open_=('<',), close_=('>',))
    assert not isValidParenthese('<>>', open_=('<',), close_=('>',))


def test_isValidParenthese_lists_are_accepted_as_brackets() -> None:
    assert isValidParenthese('([])', open_=['(', '['], close_=[')', ']'])


def test_isValidParenthese_matches_pair_reduction() -> None:
    rng = random.Random(0)

    for _ in range(300):
        s = ''.join(rng.choice('()[]{}') for _ in range(rng.randint(0, 10)))

        assert isValidParenthese(s) == (reduceBrackets(s) == '')
