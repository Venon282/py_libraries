import pytest
from py_libraries.type.string import prefixCount


@pytest.mark.parametrize(
    ('words', 'pref', 'expected'),
    [
        (['pay', 'attention', 'practice', 'attend'], 'at', 2),
        (['leetcode', 'win', 'loops', 'success'], 'code', 0),
        (['abc', 'abd', 'ab'], 'abc', 1),
        (['a', 'b'], 'z', 0),
    ],
)
def test_prefixCount_known_values(words: list[str], pref: str, expected: int) -> None:
    assert prefixCount(words, pref) == expected


def test_prefixCount_empty_prefix_matches_every_word() -> None:
    words = ['a', 'bb', '']

    assert prefixCount(words, '') == len(words)


def test_prefixCount_no_words_gives_zero() -> None:
    assert prefixCount([], 'a') == 0
