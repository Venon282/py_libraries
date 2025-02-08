def prefixCount(words:list[str], pref:str):
    """
    :type words: List[str]
    :type pref: str
    :rtype: int
    """
    return sum([1 for word in words if word.startswith(pref)])

def is1SwapAreEqual(s1, s2):
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

def longestCommonPrefix(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    prefix = strs[0]
    for str_ in strs[1:]:
        i = len(prefix)
        while i > 0:
            if prefix[:i] == str_[:i]:
                break
            i-=1
        if i == 0:
            return ""
        prefix = prefix[:i]

    return prefix

def longestUnicSubstringLength(s):
    """
    :type s: str
    :rtype: int
    """
    chars = {}
    max_length = 0
    start = 0

    for end, char in enumerate(s):
        if char in chars and chars[char] >= start:
            start = chars[char] + 1
        else:
            max_length = max(max_length, end - start + 1)
        chars[char] = end

    return max_length

def isValidParenthese(s, open_ = ['(', '[', '{'], close_ = [')', ']', '}']):
        """
        :type s: str
        :rtype: bool
        """
        corresponding = {key: value for key, value in zip(close_, open_)}
        pile = []

        for c in s:
            if c in open_:
                pile.append(c)
            elif c in close_:
                if len(pile) == 0 or (c in corresponding and pile[-1] != corresponding[c]):
                    return False
                pile.pop()
        return len(pile) == 0
