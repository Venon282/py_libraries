import pytest
from py_libraries.type.string import removeOccurrences


@pytest.mark.parametrize(
    ('s', 'part', 'expected'),
    [
        ('daabcbaabcbc', 'abc', 'dab'),
        ('axxxxyyyyb', 'xy', 'ab'),
        ('abc', 'x', 'abc'),
        ('abc', 'abc', ''),
        ('', 'a', ''),
        ('aabb', 'ab', ''),
    ],
)
def test_removeOccurrences_known_values(s: str, part: str, expected: str) -> None:
    assert removeOccurrences(s, part) == expected


@pytest.mark.parametrize(
    ('s', 'part'),
    [('daabcbaabcbc', 'abc'), ('axxxxyyyyb', 'xy'), ('aaabbb', 'ab')],
)
def test_removeOccurrences_result_has_no_occurrence_left(s: str, part: str) -> None:
    result = removeOccurrences(s, part)

    assert part not in result
    assert (len(s) - len(result)) % len(part) == 0
