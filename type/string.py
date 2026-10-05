"""Helpers to search, validate and convert strings."""

from collections.abc import Sequence

import regex


def prefixCount(words: list[str], pref: str) -> int:
    """Count the words that start with a given prefix.

    Args:
        words: Words to scan.
        pref: Prefix to look for.

    Returns:
        Number of words of `words` that start with `pref`.
    """
    return sum([1 for word in words if word.startswith(pref)])


def is1SwapAreEqual(s1: str, s2: str) -> bool:
    """Check whether two strings are equal or can be made equal with one swap.

    A swap exchanges two characters, at different positions, of one of the strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        True if the strings are equal, or if one swap in one of them makes them
        equal. False otherwise.
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


def longestCommonPrefix(strs: Sequence[str]) -> str:
    """Find the longest common prefix of several strings.

    Args:
        strs: Strings to compare.

    Returns:
        The longest prefix shared by all the strings, or an empty string if there
        is none.

    Raises:
        IndexError: If `strs` is empty.
    """
    prefix = strs[0]
    for str_ in strs[1:]:
        i = len(prefix)
        while i > 0:
            if prefix[:i] == str_[:i]:
                break
            i -= 1
        if i == 0:
            return ''
        prefix = prefix[:i]

    return prefix


def longestUnicSubstringLength(s: str) -> int:
    """Find the length of the longest substring without repeated characters.

    Args:
        s: String to scan.

    Returns:
        Length of the longest substring of `s` whose characters are all different.
    """
    chars: dict[str, int] = {}
    max_length = 0
    start = 0

    for end, char in enumerate(s):
        if char in chars and chars[char] >= start:
            start = chars[char] + 1
        else:
            max_length = max(max_length, end - start + 1)
        chars[char] = end

    return max_length


def toPlurialFr(word: str) -> str:
    """Put a French word in the plural, using the regular rules only.

    A word ending with `s`, `x` or `z` is unchanged, `au` and `eu` take an `x`,
    `al` becomes `aux`, `travail` becomes `travaux` and any other word takes an
    `s`. Irregular plurals are not handled.

    Args:
        word: Singular word.

    Returns:
        The plural form of `word`.

    Raises:
        IndexError: If `word` is empty.
    """
    if word[-1] in ('s', 'x', 'z'):
        return word  # No change
    elif word.endswith('au') or word.endswith('eau') or word.endswith('eu'):
        return word + 'x'
    elif word.endswith('al'):
        return word[:-2] + 'aux'
    elif word.endswith('ail'):
        if word in ['travail']:  # Specific case
            return word[:-3] + 'aux'
        else:
            return word + 's'
    else:
        return word + 's'


def isValidParenthese(
    s: str,
    open_: Sequence[str] = ('(', '[', '{'),
    close_: Sequence[str] = (')', ']', '}'),
) -> bool:
    """Check whether the brackets of a string are balanced and well nested.

    The characters that are neither opening nor closing brackets are ignored.

    Args:
        s: String to check.
        open_: Opening brackets.
        close_: Closing brackets, in the same order as `open_`: `close_[i]` closes
            `open_[i]`.

    Returns:
        True if every opening bracket is closed by its matching bracket in the
        right order, False otherwise.
    """
    corresponding = {key: value for key, value in zip(close_, open_, strict=False)}
    pile: list[str] = []

    for c in s:
        if c in open_:
            pile.append(c)
        elif c in close_:
            if len(pile) == 0 or (c in corresponding and pile[-1] != corresponding[c]):
                return False
            pile.pop()
    return len(pile) == 0


def removeOccurrences(s: str, part: str) -> str:
    """Remove the leftmost occurrence of a substring until there is none left.

    The search restarts from the beginning after each removal, so a removal can
    create a new occurrence.

    Args:
        s: String to clean.
        part: Substring to remove. It must not be empty, otherwise the removal
            never ends.

    Returns:
        `s` without any occurrence of `part`.
    """
    while part in s:
        idx = s.index(part)
        s = s[:idx] + s[idx + len(part) :]
    return s


def isInterleave(s1: str, s2: str, s3: str) -> bool:
    """Check whether a string is an interleaving of two other strings.

    `s3` is an interleaving of `s1` and `s2` if it can be built by merging the
    characters of `s1` and `s2` while keeping the order of the characters of each.

    Args:
        s1: First string.
        s2: Second string.
        s3: String to check.

    Returns:
        True if `s3` is an interleaving of `s1` and `s2`, False otherwise.
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


def longestPalindrome(s: str) -> str:
    """Find the longest palindromic substring.

    Args:
        s: String to scan.

    Returns:
        The longest palindromic substring of `s`. If several have the same length,
        the leftmost one is returned.

    Raises:
        IndexError: If `s` is empty.
    """
    ls = len(s)
    ans = s[0]
    for i in range(ls):
        left, right = i, i
        while right < ls - 1 and s[right + 1] == s[left]:
            right += 1
        while left > 0 and right < ls - 1 and s[left - 1] == s[right + 1]:
            left -= 1
            right += 1
        if right + 1 - left > len(ans):
            ans = s[left : right + 1]
    return ans


def myAtoi(s: str) -> int:
    """Convert a string to a 32-bit signed integer.

    Leading spaces are skipped, then an optional sign is read, then the digits up
    to the first other character. A string without digits gives 0. The result is
    clamped to the 32-bit signed integer range.

    Args:
        s: String to convert.

    Returns:
        The integer read at the beginning of `s`, clamped to the range
        [-2**31, 2**31 - 1].
    """
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
            return sign * (2**31 - 1) if sign == 1 else -(2**31)

        idx += 1

    return res * sign


def toCamelCase(s: str, separator: str = ' ') -> str:
    """Convert a string to camelCase.

    The string is split on `separator`: the first part is put in lowercase and
    each following part is capitalized.

    Args:
        s: String to convert.
        separator: Separator between the words of `s`.

    Returns:
        The camelCase form of `s`, without the separators.
    """
    components = s.split(separator)
    return components[0].lower() + ''.join(x.title() for x in components[1:])


def maxOccurenceAdjacentChar(s: str) -> int:
    """Find the longest run of identical adjacent characters.

    Args:
        s: String to scan.

    Returns:
        Length of the longest run of one character repeated immediately, or 0 if no
        character is immediately repeated.
    """
    prev = ''
    max_oc = 0
    cur_oc = 0
    for c in s:
        if c == prev:
            cur_oc += 1
            max_oc = max(max_oc, cur_oc)
        else:
            prev = c
            cur_oc = 1
    return max_oc


def maxOccurenceAdjacentEmoji(s: str) -> int:
    """Find the maximum number of consecutive emojis, not necessarily the same.

    The string is split in grapheme clusters, so an emoji sequence joined with
    zero width joiners counts as one. A cluster is an emoji if it contains a
    character with the Unicode `Emoji` property, which includes the ASCII digits,
    `#` and `*`.

    Args:
        s: String to scan.

    Returns:
        The maximum number of consecutive emoji clusters in `s`.
    """
    # Matches a grapheme cluster if it contains at least one emoji
    max_count = 0
    current_count = 0

    for cluster in regex.findall(r'\X', s):
        # check if the cluster contains an emoji
        if any(regex.match(r'\p{Emoji}', c) for c in cluster):
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


def containsChineseChar(s: str) -> bool:
    """Check whether a string contains a Chinese character.

    Only the common CJK Unified Ideographs range (U+4E00 to U+9FFF) is checked.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one character of that range, False otherwise.
    """
    return any('\u4e00' <= char <= '\u9fff' for char in s)
