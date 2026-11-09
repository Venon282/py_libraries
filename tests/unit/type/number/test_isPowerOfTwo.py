"""Unit tests for isPowerOfTwo."""

import pytest

from py_libraries.type.number import isPowerOfTwo


@pytest.mark.parametrize('exponent', range(0, 63))
def test_powers_of_two_are_detected(exponent: int) -> None:
    """Accept every power of two, and reject its neighbours.

    Args:
        exponent: Exponent of the power of two.
    """
    n = 2**exponent

    assert isPowerOfTwo(n) is True
    assert isPowerOfTwo(n + 1) is (exponent == 0)
    assert isPowerOfTwo(n - 1) is (exponent == 1)


@pytest.mark.parametrize('n', [0, -1, -2, -4, -1024])
def test_zero_and_negative_numbers_are_rejected(n: int) -> None:
    """Reject zero and the negative numbers, even the opposite of a power of two.

    Args:
        n: Zero or negative integer.
    """
    assert isPowerOfTwo(n) is False


@pytest.mark.parametrize('n', [3, 5, 6, 7, 9, 10, 12, 100, 1000])
def test_other_numbers_are_rejected(n: int) -> None:
    """Reject the positive numbers that are not a power of two.

    Args:
        n: Positive integer that is not a power of two.
    """
    assert isPowerOfTwo(n) is False


def test_matches_the_single_bit_count() -> None:
    """Agree with the definition: a single bit set in the binary form."""
    for n in range(1, 2000):
        assert isPowerOfTwo(n) == (bin(n).count('1') == 1)
