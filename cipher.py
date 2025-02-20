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
    
class ZigZagCipher(ICipher):
    """
    Implémente le chiffrement par zigzag (transposition en zigzag).
    
    La clé (key) est attendue sous forme d'une chaîne représentant un entier,
    indiquant le nombre de lignes à utiliser pour le motif en zigzag.
    """
    
    def encrypt(self, data: str, key: str) -> str:
        try:
            num_rows = int(key)
        except ValueError:
            raise ValueError("La clé doit être un entier représentant le nombre de lignes.")
        
        if num_rows < 1:
            raise ValueError("Le nombre de lignes doit être au moins 1.")
        if num_rows == 1 or len(data) <= num_rows:
            return data
        
        # Création du motif en zigzag
        rows = [''] * num_rows
        curr_row = 0
        direction = 1  # 1 pour descendre, -1 pour remonter
        
        for char in data:
            rows[curr_row] += char
            # Changement de direction si on atteint le haut ou le bas
            if curr_row == 0:
                direction = 1
            elif curr_row == num_rows - 1:
                direction = -1
            curr_row += direction
        
        return ''.join(rows)
    
    def decrypt(self, data: str, key: str) -> str:
        try:
            num_rows = int(key)
        except ValueError:
            raise ValueError("La clé doit être un entier représentant le nombre de lignes.")
        
        if num_rows < 1:
            raise ValueError("Le nombre de lignes doit être au moins 1.")
        if num_rows == 1 or len(data) <= num_rows:
            return data
        
        n = len(data)
        # Reconstituer le parcours (l'indice de ligne pour chaque caractère)
        row_indices = []
        curr_row = 0
        direction = 1
        for _ in range(n):
            row_indices.append(curr_row)
            if curr_row == 0:
                direction = 1
            elif curr_row == num_rows - 1:
                direction = -1
            curr_row += direction
        
        # Compter le nombre de caractères par ligne dans le motif original
        counts = [0] * num_rows
        for r in row_indices:
            counts[r] += 1
        
        # Découper le texte chiffré en segments correspondant à chaque ligne
        rows = []
        index = 0
        for count in counts:
            rows.append(data[index:index+count])
            index += count
        
        # Reconstituer le message original en lisant le motif selon row_indices
        pointers = [0] * num_rows
        result_chars = []
        for r in row_indices:
            result_chars.append(rows[r][pointers[r]])
            pointers[r] += 1
        
        return ''.join(result_chars)
