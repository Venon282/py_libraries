"""Unit tests for isPalindrome."""

import pytest

from py_libraries.type.number import isPalindrome


def reverseDigits(n: int) -> int:
    """Reverse the digits of a non-negative integer with arithmetic only.

    Args:
        n: Non-negative integer to reverse.

    Returns:
        The integer made of the digits of `n` in the opposite order.
    """
    reversed_n = 0
    while n > 0:
        reversed_n = reversed_n * 10 + n % 10
        n //= 10
    return reversed_n


@pytest.mark.parametrize(
    ('x', 'expected'),
    [
        (0, True),
        (7, True),
        (11, True),
        (121, True),
        (1221, True),
        (12321, True),
        (10, False),
        (123, False),
        (1000021, False),
    ],
)
def test_known_values(x: int, expected: bool) -> None:
    """Detect whether the digits read the same in both directions.

    Args:
        x: Number to check.
        expected: Expected result.
    """
    assert isPalindrome(x) is expected


@pytest.mark.parametrize('x', [-1, -121, -1221])
def test_negative_numbers_are_never_palindromes(x: int) -> None:
    """Reject the negative numbers, because of the minus sign.

    Args:
        x: Negative number to check.
    """
    assert isPalindrome(x) is False


def test_matches_the_arithmetic_reversal() -> None:
    """Agree with a digit reversal done without strings."""
    for x in range(3000):
        assert isPalindrome(x) == (x == reverseDigits(x))


@pytest.mark.parametrize('half', [1, 12, 123, 4096, 98765])
def test_mirrored_digits_are_palindromes(half: int) -> None:
    """Accept the numbers built by mirroring a half, with an even or odd length.

    Args:
        half: Digits of the first half.
    """
    digits = str(half)

    assert isPalindrome(int(digits + digits[::-1])) is True
    assert isPalindrome(int(digits + digits[-2::-1])) is True
