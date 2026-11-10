"""Unit tests for ICipher.encrypt."""

import pytest
from py_libraries.cipher.ICipher import ICipher


def test_encrypt_is_abstract() -> None:
    """A class that only implements `decrypt` cannot be instantiated."""

    class OnlyDecrypt(ICipher):
        def decrypt(self, data: str, key: str) -> str:
            return data

    with pytest.raises(TypeError):
        OnlyDecrypt()
