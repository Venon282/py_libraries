"""Helpers for integers, Roman numerals, number formatting and Gaussian sampling."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import SupportsFloat

import numpy as np

ROMAN_VALUES: Mapping[str, int] = MappingProxyType(
    {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
)

# Symbols of the subtractive notation, from the largest value to the smallest.
ROMAN_PAIRS: tuple[tuple[str, int], ...] = (
    ('M', 1000),
    ('CM', 900),
    ('D', 500),
    ('CD', 400),
    ('C', 100),
    ('XC', 90),
    ('L', 50),
    ('XL', 40),
    ('X', 10),
    ('IX', 9),
    ('V', 5),
    ('IV', 4),
    ('I', 1),
)


def getNextMultipleOf(n: int, multiple: int) -> int:
    """Return the smallest multiple of a number that is greater than or equal to n.

    Args:
        n: The number to find the next multiple for.
        multiple: The multiple to align to. It must be positive: with a negative
            value the result can be lower than `n`.

    Returns:
        The smallest multiple of `multiple` that is greater than or equal to `n`.

    Raises:
        ZeroDivisionError: If `multiple` is 0.
    """
    return n if n % multiple == 0 else (n // multiple + 1) * multiple


def isPalindrome(x: int) -> bool:
    """Check if a number is a palindrome.

    A negative number is never a palindrome, because of its minus sign.

    Args:
        x: The number to check.

    Returns:
        True if the decimal digits of `x` read the same in both directions, False
        otherwise.
    """
    x_str = str(x)
    return x_str == x_str[::-1]


def computeMaxAscendingSum(nums: Sequence[int]) -> int:
    """Compute the maximum sum of a strictly ascending contiguous subarray.

    Args:
        nums: Positive integers to process. With negative values, the best
            subarray can be a part of an ascending run, but only the whole runs
            are considered.

    Returns:
        The maximum sum of a contiguous subarray whose elements are strictly
        increasing. A single element is such a subarray.

    Raises:
        IndexError: If `nums` is empty.
    """
    current_sum = max_sum = nums[0]
    for i in range(1, len(nums)):
        if nums[i - 1] < nums[i]:
            current_sum += nums[i]
        else:
            max_sum = max(max_sum, current_sum)
            current_sum = nums[i]
    return max(max_sum, current_sum)


def romanToInt(s: str) -> int:
    """Convert a Roman numeral string to an integer.

    The numeral is read from left to right, and a symbol lower than the next one
    is subtracted from it. The numeral is not validated: a malformed one such as
    `'IC'` is converted without error (it gives 99).

    Args:
        s: Roman numeral string to convert, in upper case.

    Returns:
        The integer value of the Roman numeral, 0 for an empty string.

    Raises:
        KeyError: If `s` contains a character that is not an upper-case Roman symbol.
    """
    num = i = 0
    while i < len(s):
        if i < len(s) - 1 and ROMAN_VALUES[s[i + 1]] > ROMAN_VALUES[s[i]]:
            num += ROMAN_VALUES[s[i + 1]] - ROMAN_VALUES[s[i]]
            i += 2
        else:
            num += ROMAN_VALUES[s[i]]
            i += 1

    return num


def intToRoman(num: int) -> str:
    """Convert an integer to a Roman numeral string, in subtractive notation.

    Zero and negative numbers give an empty string. The notation has no symbol
    above `'M'`, so numbers above 3999 repeat it (4000 gives `'MMMM'`).

    Args:
        num: Integer to convert.

    Returns:
        The Roman numeral of `num`, in upper case.
    """
    res = ''

    for symbol, value in ROMAN_PAIRS:
        while num >= value:
            res += symbol
            num -= value

    return res


def isPowerOfTwo(n: int) -> bool:
    """Check if a number is a power of two.

    Args:
        n: The number to check.

    Returns:
        True if `n` is a positive power of two (1, 2, 4, ...), False otherwise.
    """
    return n > 0 and (n & (n - 1)) == 0


def computeDigitalRoot(num: int) -> int:
    """Compute the digital root of a number (repeated sum of digits until single digit).

    Negative numbers are not supported: the result is then not the digital root of
    the absolute value.

    Args:
        num: The non-negative number to compute the digital root for.

    Returns:
        The digital root of the number.

    Example:
        >>> computeDigitalRoot(38)
        2

        The process is 38 -> 3 + 8 = 11, then 11 -> 1 + 1 = 2. Since 2 has only
        one digit, it is returned.
    """
    return 0 if num == 0 else (num % 9 or 9)


def getSign(n: int | float) -> int:
    """Get the sign of a number.

    Args:
        n: The number to get the sign for.

    Returns:
        1 if positive, -1 if negative, 0 if zero (or NaN).
    """
    return (n > 0) - (n < 0)


def reverseNumber(n: int) -> int:
    """Reverse the digits of a number, keeping its sign.

    Args:
        n: The number to reverse.

    Returns:
        The number with its digits reversed. The trailing zeros of `n` are dropped,
        so 120 gives 21.
    """
    return int(str(abs(n))[::-1]) * getSign(n)


def _trimTrailingZeros(s: str) -> str:
    """Remove the trailing zeros of the fractional part of a decimal string.

    Args:
        s: Decimal string, with or without a fractional part.

    Returns:
        `s` without its trailing fractional zeros and without a dangling point. A
        string without a point is returned unchanged, so `'100'` keeps its zeros.
    """
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def formatNumber(
    num: SupportsFloat | str,
    bound_min: float = 0.001,
    bound_max: float = 1e6,
    precision: int = 3,
) -> str:
    """Return a human-readable string representation of a number.

    Zero gives `'0'`. A magnitude in `[bound_min, bound_max)` is written in
    fixed-point notation, any other magnitude in scientific notation. The trailing
    zeros of the fractional part (or of the mantissa) are removed, and the exponent
    is written without a `+` sign or leading zeros. In fixed-point notation, a
    magnitude lower than the resolution of `precision` decimals gives `'0'`.

    Args:
        num: Number to format. A string is converted with `float`. A value that
            `float` cannot convert is returned as `str(num)`.
        bound_min: Lower bound of the magnitudes written in fixed-point notation.
        bound_max: Upper bound (excluded) of the magnitudes written in fixed-point
            notation.
        precision: Maximum number of decimals, in the fractional part or in the
            mantissa.

    Returns:
        The formatted number. `nan` and `inf` are returned as such.

    Example:
        >>> formatNumber(0.0000000256)
        '2.56e-8'
        >>> formatNumber(0.654212574854)
        '0.654'
        >>> formatNumber(455351356)
        '4.554e8'
        >>> formatNumber(1.0)
        '1'
    """
    try:
        value = float(num)
    except (TypeError, ValueError):
        return str(num)

    if value == 0:
        return '0'

    if abs(value) < bound_min or abs(value) >= bound_max:
        s = f'{value:.{precision}e}'
        # An infinite value has no exponent to split
        if 'e' not in s:
            return s
        mantissa, exponent = s.split('e')
        # int() drops the sign and the leading zeros of the exponent
        return f'{_trimTrailingZeros(mantissa)}e{int(exponent)}'

    return _trimTrailingZeros(f'{value:.{precision}f}')


def _drawGaussian(mean: float, std: float, rng: np.random.Generator | None) -> float:
    """Draw one value from a Gaussian distribution.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
        rng: Random generator to draw from. If None, uses the global numpy state.

    Returns:
        The drawn value.

    Raises:
        ValueError: If `std` is negative.
    """
    if rng is None:
        return float(np.random.normal(loc=mean, scale=std))
    return float(rng.normal(loc=mean, scale=std))


def generateRandomGaussianWithBounds(
    bound_min: float,
    bound_max: float,
    mean: float | None = None,
    std: float | None = None,
    retry: bool = True,
    dispersion: int = 3,
    rng: np.random.Generator | None = None,
) -> float:
    """Generate a random number from a Gaussian distribution within bounds.

    The number is drawn again while it falls outside the bounds if `retry` is
    True, then clamped to `[bound_min, bound_max]`. With `retry` and a distribution
    that has very little mass inside the bounds, the call can take a long time.

    Args:
        bound_min: Lower bound of the allowed range.
        bound_max: Upper bound of the allowed range.
        mean: Mean of the Gaussian distribution. Default is the midpoint of the
            bounds.
        std: Standard deviation of the Gaussian distribution. Default is
            `(mean - bound_min) / dispersion`, so that about 99.7 % of the draws
            fall within the bounds when `dispersion` is 3 and `mean` is centered.
        retry: If True, draw again while the number falls outside the bounds.
            Otherwise the number is only clamped.
        dispersion: Factor used to compute the default `std`. A larger value gives
            a smaller `std`.
        rng: Random generator to draw from, for reproducible results. If None, the
            global numpy random state is used.

    Returns:
        A random value of the Gaussian distribution, within `[bound_min, bound_max]`.

    Raises:
        ValueError: If the standard deviation is negative, which happens by default
            when `bound_min` is above `bound_max`.
    """
    if mean is None:
        mean = (bound_min + bound_max) / 2
    if std is None:
        std = (mean - bound_min) / dispersion

    rand = _drawGaussian(mean, std, rng)

    while retry and (rand < bound_min or rand > bound_max):
        rand = _drawGaussian(mean, std, rng)

    return float(min(max(bound_min, rand), bound_max))
