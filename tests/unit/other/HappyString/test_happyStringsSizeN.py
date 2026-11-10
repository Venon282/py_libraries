"""Unit tests for HappyString.happyStringsSizeN."""

import pytest

from py_libraries.other.HappyString import HappyString


def test_happyStringsSizeN_zero_length_gives_the_empty_string() -> None:
    """Return the empty string as the only happy string of length 0."""
    assert HappyString().happyStringsSizeN(0) == ['']


def test_happyStringsSizeN_matches_the_count_formula() -> None:
    """Return m * (m - 1) ** (n - 1) strings for an alphabet of m characters."""
    happy = HappyString('abcd')

    for n in range(1, 6):
        assert len(happy.happyStringsSizeN(n)) == 4 * 3 ** (n - 1)


def test_happyStringsSizeN_returns_only_happy_strings_of_the_right_length() -> None:
    """Return strings of the requested length that have no equal neighbors."""
    happy = HappyString('abc')
    strings = happy.happyStringsSizeN(4)

    assert strings
    assert all(len(s) == 4 and happy.isHappyString(s) for s in strings)


def test_happyStringsSizeN_is_sorted_and_matches_nThHappyStringOfLength() -> None:
    """Return the strings in lexicographic order, agreeing with the k-th string."""
    happy = HappyString('ab')
    strings = happy.happyStringsSizeN(5)

    assert strings == sorted(strings)
    for k, expected in enumerate(strings, start=1):
        assert happy.nThHappyStringOfLength(5, k) == expected


def test_happyStringsSizeN_negative_size_raises() -> None:
    """Raise ValueError for a negative length."""
    with pytest.raises(ValueError):
        HappyString().happyStringsSizeN(-1)
