"""Tools to build, enumerate and validate happy strings.

A happy string is a string in which no two adjacent characters are the same.
"""


class HappyString:
    """Builder and checker for happy strings.

    A happy string is a string in which no two adjacent characters are the same.

    Args:
        chars: Distinct characters that can be used to build the happy strings.
    """

    def __init__(self, chars: str = 'abc') -> None:
        self.chars = chars

    def nThHappyStringOfLength(self, size: int, k: int) -> str:
        """Return the kth happy string of a given length.

        The happy strings of length `size` are ranked in the lexicographic order
        induced by the order of `self.chars`.

        Args:
            size: Length of the happy string to build.
            k: 1-based rank of the happy string.

        Returns:
            The kth happy string of length `size`, or an empty string if `k` is
            greater than the total number of happy strings.

        Raises:
            ValueError: If `size` is negative or `k` is less than 1.
        """
        if size < 0:
            raise ValueError('size must be a non-negative integer.')
        if k < 1:
            raise ValueError('k must be a positive integer.')

        m = len(self.chars)
        if size == 0:
            return ''

        # Total number of happy strings: m * (m - 1)^(size - 1)
        total = m * ((m - 1) ** (size - 1))
        if k > total:
            return ''

        k -= 1  # convert k to 0-indexed

        # Determine the first character.
        first_count = (m - 1) ** (size - 1)
        first_index = k // first_count
        res = [self.chars[first_index]]
        k %= first_count

        # Build the rest of the string.
        for i in range(1, size):
            count = (m - 1) ** (size - i - 1)
            prev = res[-1]
            # Options: all characters in self.chars except the previous character.
            options = [c for c in self.chars if c != prev]
            index = k // count
            res.append(options[index])
            k %= count

        return ''.join(res)

    def happyStringsSizeN(self, n: int) -> list[str]:
        """Return all the happy strings of a given length.

        The strings are returned in the lexicographic order induced by the order
        of `self.chars`.

        Args:
            n: Length of the happy strings to generate.

        Returns:
            All the happy strings of length `n` built from `self.chars`. The list
            contains the single empty string when `n` is 0.

        Raises:
            ValueError: If `n` is negative.
        """
        if n < 0:
            raise ValueError('n must be a non-negative integer.')

        def happyStringsRec(word: str, n: int) -> list[str]:
            if len(word) == n:
                return [word]
            res = []
            for c in self.chars:
                if not word or c != word[-1]:
                    res.extend(happyStringsRec(word + c, n))
            return res

        return happyStringsRec('', n)

    def isHappyString(self, string: str) -> bool:
        """Check whether a string is a happy string built from the allowed chars.

        Args:
            string: String to check.

        Returns:
            True if every character of `string` belongs to `self.chars` and no two
            adjacent characters are equal. The empty string is happy. False
            otherwise.
        """
        for i, char in enumerate(string):
            if char not in self.chars or (i > 0 and char == string[i - 1]):
                return False
        return True

    def substringsContainingAllChars(self, string: str) -> int:
        """Count the substrings of a string that contain every allowed character.

        Args:
            string: String to scan.

        Returns:
            Number of substrings of `string` that contain at least one occurrence
            of every character of `self.chars`. Characters of `string` that are
            not in `self.chars` are ignored.
        """
        last_seen = {c: -1 for c in self.chars}

        total = 0
        for i in range(len(string)):
            last_seen[string[i]] = i
            total += min(last_seen.values()) + 1

        return total
