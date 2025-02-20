from abc import ABC, abstractmethod


class IEncoder(ABC):
    @abstractmethod
    def encode(self, data: str) -> str:
        """Convertit les données en un autre format (ex: Base64)."""
        pass
    
    @abstractmethod
    def decode(self, data: str) -> str:
        """Retourne les données à leur format d'origine."""
        pass