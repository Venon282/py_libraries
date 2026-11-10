"""Unit tests for HappyString.nThHappyStringOfLength."""

import pytest

from py_libraries.other.HappyString import HappyString


@pytest.mark.parametrize(
    ('size', 'k', 'expected'),
    [
        (0, 1, ''),
        (1, 1, 'a'),
        (1, 2, 'b'),
        (1, 3, 'c'),
        (1, 4, ''),
        (2, 1, 'ab'),
        (2, 2, 'ac'),
        (2, 3, 'ba'),
        (2, 5, 'ca'),
        (2, 6, 'cb'),
        (2, 7, ''),
        (3, 1, 'aba'),
        (3, 6, 'bac'),
        (3, 12, 'cbc'),
        (3, 13, ''),
    ],
)
def test_nThHappyStringOfLength_known_values(size: int, k: int, expected: str) -> None:
    """Return the k-th happy string, or an empty string beyond the last one.

    Args:
        size: Length of the happy string.
        k: 1-based rank of the happy string.
        expected: Expected happy string.
    """
    assert HappyString().nThHappyStringOfLength(size, k) == expected


@pytest.mark.parametrize(
    ('chars', 'size', 'k', 'expected'),
    [
        ('ab', 1, 1, 'a'),
        ('ab', 1, 2, 'b'),
        ('ab', 1, 3, ''),
        ('ab', 2, 1, 'ab'),
        ('ab', 2, 2, 'ba'),
        ('ab', 2, 3, ''),
        ('ab', 3, 1, 'aba'),
        ('ab', 3, 2, 'bab'),
        ('ab', 3, 3, ''),
    ],
)
def test_nThHappyStringOfLength_two_char_alphabet(
    chars: str, size: int, k: int, expected: str
) -> None:
    """Rank the happy strings of an alphabet of two characters.

    Args:
        chars: Allowed characters.
        size: Length of the happy string.
        k: 1-based rank of the happy string.
        expected: Expected happy string.
    """
    assert HappyString(chars).nThHappyStringOfLength(size, k) == expected


def test_nThHappyStringOfLength_matches_happyStringsSizeN() -> None:
    """Agree with the full enumeration for every rank."""
    happy = HappyString('abc')
    strings = happy.happyStringsSizeN(4)

    assert len(strings) == 3 * 2**3
    for k, expected in enumerate(strings, start=1):
        assert happy.nThHappyStringOfLength(4, k) == expected


def test_nThHappyStringOfLength_returns_a_happy_string_of_the_right_length() -> None:
    """Return a string of the requested length that has no equal neighbors."""
    happy = HappyString('abcd')

    for k in range(1, 4 * 3**3 + 1):
        result = happy.nThHappyStringOfLength(4, k)
        assert len(result) == 4
        assert happy.isHappyString(result)


def test_nThHappyStringOfLength_negative_size_raises() -> None:
    """Raise ValueError for a negative length."""
    with pytest.raises(ValueError):
        HappyString().nThHappyStringOfLength(-1, 1)


@pytest.mark.parametrize('k', [0, -1, -10])
def test_nThHappyStringOfLength_k_below_one_raises(k: int) -> None:
    """Raise ValueError for a rank below 1.

    Args:
        k: Invalid rank.
    """
    with pytest.raises(ValueError):
        HappyString().nThHappyStringOfLength(3, k)
