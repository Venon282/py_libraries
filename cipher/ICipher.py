"""Abstract interface shared by the string ciphers of the project."""

from abc import ABC, abstractmethod


class ICipher(ABC):
    """Abstract base class for a cipher that encrypts and decrypts strings.

    A cipher is fully defined by its `encrypt` and `decrypt` methods, which take
    a string and a key and return a string.
    """

    @abstractmethod
    def encrypt(self, data: str, key: str) -> str:
        """Encrypt a string with the given key.

        Args:
            data: Plain text to encrypt.
            key: Key that selects the encryption.

        Returns:
            The encrypted text.
        """
        ...

    @abstractmethod
    def decrypt(self, data: str, key: str) -> str:
        """Decrypt a string with the given key.

        Args:
            data: Encrypted text to decrypt.
            key: Key that selects the decryption, the one used to encrypt.

        Returns:
            The recovered plain text.
        """
        ...
