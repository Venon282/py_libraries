import pytest
from py_libraries.other.HappyString import HappyString


def test_happyStringsSizeN_zero_length_gives_the_empty_string() -> None:
    assert HappyString().happyStringsSizeN(0) == ['']


def test_happyStringsSizeN_matches_the_count_formula() -> None:
    happy = HappyString('abcd')

    for n in range(1, 6):
        assert len(happy.happyStringsSizeN(n)) == 4 * 3 ** (n - 1)


def test_happyStringsSizeN_returns_only_happy_strings_of_the_right_length() -> None:
    happy = HappyString('abc')
    strings = happy.happyStringsSizeN(4)

    assert strings
    assert all(len(s) == 4 and happy.isHappyString(s) for s in strings)


def test_happyStringsSizeN_is_sorted_and_matches_nThHappyStringOfLength() -> None:
    happy = HappyString('ab')
    strings = happy.happyStringsSizeN(5)

    assert strings == sorted(strings)
    for k, expected in enumerate(strings, start=1):
        assert happy.nThHappyStringOfLength(5, k) == expected


def test_happyStringsSizeN_negative_size_raises() -> None:
    with pytest.raises(ValueError):
        HappyString().happyStringsSizeN(-1)
