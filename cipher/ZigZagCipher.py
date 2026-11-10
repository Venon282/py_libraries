"""Zigzag transposition cipher.

The text is written in a zigzag path across a fixed number of rows and read back
row by row. The key is a string that holds the number of rows of the pattern.
"""

from py_libraries.cipher.ICipher import ICipher


class ZigZagCipher(ICipher):
    """Transposition cipher that writes the text in a zigzag pattern.

    The key is a string that represents a positive integer: the number of rows of
    the zigzag pattern. The text is laid out along the zigzag path and the cipher
    text is the concatenation of the rows, read top to bottom.
    """

    def encrypt(self, data: str, key: str) -> str:
        """Encrypt a string by writing it in a zigzag pattern and reading the rows.

        Args:
            data: Plain text to encrypt.
            key: String that holds the number of rows of the zigzag pattern.

        Returns:
            The rows of the zigzag pattern concatenated, top to bottom.

        Raises:
            ValueError: If `key` is not an integer, or if the number of rows is
                less than 1.
        """
        num_rows = self._rows(key)
        if num_rows == 1 or len(data) <= num_rows:
            return data

        rows = [''] * num_rows
        curr_row = 0
        direction = 1
        for char in data:
            rows[curr_row] += char
            if curr_row == 0:
                direction = 1
            elif curr_row == num_rows - 1:
                direction = -1
            curr_row += direction

        return ''.join(rows)

    def decrypt(self, data: str, key: str) -> str:
        """Recover the plain text from the rows of a zigzag-encrypted string.

        Args:
            data: Encrypted text, the rows of the zigzag pattern concatenated.
            key: String that holds the number of rows of the zigzag pattern, the
                one used to encrypt.

        Returns:
            The recovered plain text.

        Raises:
            ValueError: If `key` is not an integer, or if the number of rows is
                less than 1.
        """
        num_rows = self._rows(key)
        if num_rows == 1 or len(data) <= num_rows:
            return data

        n = len(data)
        row_indices = self._rowIndices(n, num_rows)

        # Count the characters that fall on each row of the original pattern
        counts = [0] * num_rows
        for r in row_indices:
            counts[r] += 1

        # Cut the cipher text into the segments that make up each row
        rows = []
        index = 0
        for count in counts:
            rows.append(data[index : index + count])
            index += count

        # Read the pattern back along row_indices to recover the plain text
        pointers = [0] * num_rows
        result_chars = []
        for r in row_indices:
            result_chars.append(rows[r][pointers[r]])
            pointers[r] += 1

        return ''.join(result_chars)

    @staticmethod
    def _rows(key: str) -> int:
        """Parse the key into the number of rows of the zigzag pattern.

        Args:
            key: String that holds the number of rows.

        Returns:
            The number of rows.

        Raises:
            ValueError: If `key` is not an integer, or if the number of rows is
                less than 1.
        """
        try:
            num_rows = int(key)
        except ValueError:
            raise ValueError(
                'The key must be an integer that holds the number of rows.'
            ) from None
        if num_rows < 1:
            raise ValueError('The number of rows must be at least 1.')
        return num_rows

    @staticmethod
    def _rowIndices(length: int, num_rows: int) -> list[int]:
        """Return the row index of every character of a zigzag pattern.

        Args:
            length: Number of characters of the pattern.
            num_rows: Number of rows of the pattern.

        Returns:
            The row index of each character, in the order of the zigzag path.
        """
        row_indices = []
        curr_row = 0
        direction = 1
        for _ in range(length):
            row_indices.append(curr_row)
            if curr_row == 0:
                direction = 1
            elif curr_row == num_rows - 1:
                direction = -1
            curr_row += direction
        return row_indices
