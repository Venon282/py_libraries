"""Unit tests for ICipher.decrypt."""

import pytest
from py_libraries.cipher.ICipher import ICipher


def test_decrypt_is_abstract() -> None:
    """A class that only implements `encrypt` cannot be instantiated."""

    class OnlyEncrypt(ICipher):
        def encrypt(self, data: str, key: str) -> str:
            return data

    with pytest.raises(TypeError):
        OnlyEncrypt()
