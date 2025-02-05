def prefixCount(words:list[str], pref:str):
        """
        :type words: List[str]
        :type pref: str
        :rtype: int
        """
    return sum([1 for word in words if word.startswith(pref)])

def is1SwapAreEqual(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) != len(s2):
            return False

        if s1 == s2:
            return True

        different = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                different.append(i)

        if len(different) != 2:
            return False

        i, j = different
        return s1[i] == s2[j] and s1[j] == s2[i]
