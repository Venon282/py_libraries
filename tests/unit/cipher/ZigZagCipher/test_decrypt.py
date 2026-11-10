"""Unit tests for ZigZagCipher.decrypt."""

import pytest
from py_libraries.cipher.ZigZagCipher import ZigZagCipher

CIPHER = ZigZagCipher()


@pytest.mark.parametrize(
    ('data', 'key', 'expected'),
    [
        ('BNAAAN', '3', 'BANANA'),
        ('ABDC', '3', 'ABCD'),
        ('ACEBD', '2', 'ABCDE'),
    ],
)
def test_decrypt_known_values(data: str, key: str, expected: str) -> None:
    """Recover the plain text from the rows of the pattern.

    Args:
        data: Cipher text.
        key: Number of rows, as a string.
        expected: Expected plain text.
    """
    assert CIPHER.decrypt(data, key) == expected


@pytest.mark.parametrize(
    ('data', 'key'),
    [
        ('BANANA', '3'),
        ('ABCD', '3'),
        ('ABCDE', '2'),
        ('HELLO WORLD', '4'),
        ('a', '1'),
    ],
)
def test_decrypt_inverts_encrypt(data: str, key: str) -> None:
    """Recover exactly the text that `encrypt` produced.

    Args:
        data: Plain text to encrypt and then decrypt.
        key: Number of rows, as a string.
    """
    assert CIPHER.decrypt(CIPHER.encrypt(data, key), key) == data


@pytest.mark.parametrize('key', ['x', '0', '-1'])
def test_decrypt_rejects_an_invalid_key(key: str) -> None:
    """Raise a ValueError for a key that is not a positive integer.

    Args:
        key: Invalid number of rows.
    """
    with pytest.raises(ValueError):
        CIPHER.decrypt('ABC', key)
