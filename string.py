from typing import List


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

def toPlurialFr(word):
    if word[-1] in ('s', 'x', 'z'):
        return word  # Pas de changement
    elif word.endswith('au') or word.endswith('eau') or word.endswith('eu'):
        return word + 'x'
    elif word.endswith('al'):
        return word[:-2] + 'aux'
    elif word.endswith('ail'):
        if word in ['travail']:  # Cas spécifique
            return word[:-3] + 'aux'
        else:
            return word + 's'
    else:
        return word + 's'
    

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

def removeOccurrences(s, part):
    """
    :type s: str
    :type part: str
    :rtype: str
    """
    while part in s:
        idx = s.index(part)
        s = s[:idx] + s[idx+len(part):]
    return s

def isInterleave(s1, s2, s3):
    """
    :type s1: str
    :type s2: str
    :type s3: str
    :rtype: bool
    """
    # If lengths do not match, return False
    if len(s1) + len(s2) != len(s3):
        return False

    # DP table to store results for subproblems
    dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # Base case: empty s1 and s2 interleave to form empty s3
    dp[0][0] = True

    # Fill DP table
    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i > 0 and s1[i - 1] == s3[i + j - 1]:
                dp[i][j] = dp[i][j] or dp[i - 1][j]
            if j > 0 and s2[j - 1] == s3[i + j - 1]:
                dp[i][j] = dp[i][j] or dp[i][j - 1]

    # The result is stored in dp[len(s1)][len(s2)]
    return dp[len(s1)][len(s2)]

def longestPalindrome(s):
        """
        :type s: str
        :rtype: str
        """

        ls = len(s)
        ans = s[0]
        for i in range(ls):
            l,r = i,i
            while r < ls-1 and s[r+1] == s[l]:
                r +=1
            while l >0 and r < ls-1 and s[l-1] == s[r+1]:
                l -=1
                r +=1
            if r+1-l > len(ans):
                ans = s[l:r+1]
        return ans

def myAtoi(s: str) -> int:
    sign = 1
    res = 0
    idx = 0

    # Ignore leading whitespaces
    while idx < len(s) and s[idx] == ' ':
        idx += 1

    # Store the sign of number
    if idx < len(s) and (s[idx] == '-' or s[idx] == '+'):
        if s[idx] == '-':
            sign = -1
        idx += 1

    # Construct the number digit by digit
    while idx < len(s) and '0' <= s[idx] <= '9':

        # Append current digit to the result
        res = 10 * res + (ord(s[idx]) - ord('0'))

        # Handling overflow/underflow test case
        if res > (2**31 - 1):
            return sign * (2**31 - 1) if sign == 1 else -2**31

        idx += 1

    return res * sign
