import random

import pytest
from py_libraries.type.string import myAtoi

INT_MAX = 2**31 - 1
INT_MIN = -(2**31)


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        ('42', 42),
        ('   -042', -42),
        ('1337c0d3', 1337),
        ('0-1', 0),
        ('words and 987', 0),
        ('+1', 1),
        ('', 0),
        ('   ', 0),
        ('-', 0),
        ('+-12', 0),
    ],
)
def test_myAtoi_known_values(s: str, expected: int) -> None:
    assert myAtoi(s) == expected


@pytest.mark.parametrize(
    ('s', 'expected'),
    [
        (str(INT_MAX), INT_MAX),
        (str(INT_MAX + 1), INT_MAX),
        ('91283472332', INT_MAX),
        (str(INT_MIN), INT_MIN),
        (str(INT_MIN - 1), INT_MIN),
        ('-91283472332', INT_MIN),
    ],
)
def test_myAtoi_clamps_to_the_32_bit_range(s: str, expected: int) -> None:
    assert myAtoi(s) == expected


def test_myAtoi_reads_back_any_32_bit_integer() -> None:
    rng = random.Random(0)

    for _ in range(200):
        value = rng.randint(INT_MIN, INT_MAX)

        assert myAtoi(f'  {value}abc') == value
