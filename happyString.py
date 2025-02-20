from typing import List

class HappyString:
    """Happy string is a string that doesn't have two adjacent characters are the same.
        Args:
            chars (string): list of the chars that can be used to build the happy string. (default: 'abc')
    """
    def __init__(self, chars='abc'):
        self.chars = chars
                

    def getHappyString(self, size: int, k: int) -> str:
        """
        Returns the kth happy string of length 'size' using the characters in self.chars.
        A happy string is one where no two adjacent characters are the same.
        """
        m = len(self.chars)
        if size == 0:
            return ""
        # Total number of happy strings: m * (m - 1)^(size - 1)
        total = m * ((m - 1) ** (size - 1))
        if k > total:
            return ""
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

        return "".join(res)

    def happyStrings(self, n: int) -> List[str]:
        """get all happy strings of size n

        Args:
            n (int): _description_

        Returns:
            List[str]: _description_
        """
        def happyStringsRec(word, n):
            if len(word) == n:
                return [word]
            res = []
            for c in self.chars:
                if not word or c != word[-1]:
                    res.extend(happyStringsRec(word + c, n))
            return res
        
        return happyStringsRec("", n)

    def isHappyString(self, string):
        if string[0] not in self.chars:
            return False
        
        for i in range(1, len(string)):
            if string[i] not in self.chars or string[i] == string[i - 1]:
                return False
        return True