"""
Module: cipher_module
Description: Contains classes for various encoding/decoding methods.
Note: The ZigZagCipher class implements a basic transposition cipher using the zigzag method.
"""

from abc import ABC, abstractmethod

# For strings
class ICipher(ABC):
    @abstractmethod
    def encrypt(self, data: str, key: str) -> str:
        """Chiffre les données avec la clé fournie."""
        pass
    
    @abstractmethod
    def decrypt(self, data: str, key: str) -> str:
        """Déchiffre les données avec la clé fournie."""
        pass