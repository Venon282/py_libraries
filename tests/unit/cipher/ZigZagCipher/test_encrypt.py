"""Unit tests for ZigZagCipher.encrypt."""

import pytest
from py_libraries.cipher.ZigZagCipher import ZigZagCipher

CIPHER = ZigZagCipher()


@pytest.mark.parametrize(
    ('data', 'key', 'expected'),
    [
        ('BANANA', '3', 'BNAAAN'),
        ('ABCD', '3', 'ABDC'),
        ('ABCDE', '2', 'ACEBD'),
    ],
)
def test_encrypt_known_values(data: str, key: str, expected: str) -> None:
    """Produce the rows of the zigzag pattern concatenated top to bottom.

    Args:
        data: Plain text.
        key: Number of rows, as a string.
        expected: Expected cipher text.
    """
    assert CIPHER.encrypt(data, key) == expected


@pytest.mark.parametrize(
    ('data', 'key'),
    [
        ('ABC', '1'),
        ('ABC', '5'),
        ('', '3'),
    ],
)
def test_encrypt_returns_the_input_when_the_pattern_is_trivial(
    data: str, key: str
) -> None:
    """Return the input unchanged when the pattern has one row or does not wrap.

    Args:
        data: Plain text.
        key: Number of rows, as a string.
    """
    assert CIPHER.encrypt(data, key) == data


@pytest.mark.parametrize('key', ['x', '0', '-1'])
def test_encrypt_rejects_an_invalid_key(key: str) -> None:
    """Raise a ValueError for a key that is not a positive integer.

    Args:
        key: Invalid number of rows.
    """
    with pytest.raises(ValueError):
        CIPHER.encrypt('ABC', key)
