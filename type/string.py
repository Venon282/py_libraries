"""Helpers to search, validate and convert strings."""

from collections.abc import Sequence

import regex

# Unicode script names, spelled as the `regex` module expects them.
JAPANESE_SCRIPTS: tuple[str, ...] = ('Hiragana', 'Katakana')
INDIAN_SCRIPTS: tuple[str, ...] = (
    'Devanagari',
    'Bengali',
    'Gurmukhi',
    'Gujarati',
    'Oriya',
    'Tamil',
    'Telugu',
    'Kannada',
    'Malayalam',
    'Ol_Chiki',
    'Meetei_Mayek',
)

# Letters minus the Latin, Common and Inherited scripts. Common and Inherited
# letters (for example mathematical alphanumeric symbols) belong to no particular
# script, so they must not make a string non-Latin. The set subtraction needs the
# regex.V1 flag.
_NON_LATIN_LETTER_PATTERN = (
    r'[\p{L}--\p{Script=Latin}--\p{Script=Common}--\p{Script=Inherited}]'
)

# Restricts script names to what a Unicode script name can contain, so that a name
# can never alter the structure of the pattern it is inserted in.
_SCRIPT_NAME_PATTERN = r'[A-Za-z0-9_ -]+'


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
    """Check whether a string contains a Chinese (Han) character.

    Covers the whole Unicode Han script: the CJK Unified Ideographs and all their
    extensions, the compatibility ideographs, the CJK radicals and a few ideographic
    marks such as the iteration mark U+3005. Han ideographs are shared with Japanese
    kanji and Korean hanja, so this cannot tell Chinese from those languages. CJK
    punctuation belongs to no script and does not match.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Han character, False otherwise.
    """
    return containsScriptChar(s, 'Han')


def _scriptProperty(name: str) -> str:
    """Build the `regex` expression matching the characters of one Unicode script.

    Args:
        name: Name of the Unicode script.

    Returns:
        The property expression to insert in a character class.

    Raises:
        ValueError: If `name` is not a known Unicode script.
    """
    if not regex.fullmatch(_SCRIPT_NAME_PATTERN, name):
        raise ValueError(f'Unknown Unicode script: {name!r}.')

    expression = f'\\p{{Script={name}}}'
    try:
        regex.compile(expression)
    except regex.error as err:
        raise ValueError(f'Unknown Unicode script: {name!r}.') from err

    return expression


def containsScriptChar(s: str, *scripts: str) -> bool:
    """Check whether a string contains a character of at least one Unicode script.

    This is the building block of the `contains<Script>Char` helpers. Use it
    directly to test a script that has no dedicated helper. The script names are
    those of the `regex` module (the Unicode `Script` property, for example
    `'Cyrillic'` or `'Ol_Chiki'`) and are matched without regard to case.

    Args:
        s: String to scan.
        *scripts: Names of the Unicode scripts to look for. At least one is required.

    Returns:
        True if `s` contains at least one character of any of the given scripts,
        False otherwise.

    Raises:
        ValueError: If no script name is given, or if a name is not a known Unicode
            script.

    Example:
        >>> containsScriptChar('abc', 'Latin')
        True
        >>> containsScriptChar('abc', 'Greek', 'Cyrillic')
        False
    """
    if not scripts:
        raise ValueError('At least one script name is required.')

    pattern = '[' + ''.join(_scriptProperty(name) for name in scripts) + ']'
    return regex.search(pattern, s) is not None


def isNotLatin(s: str) -> bool:
    """Check whether a string contains a letter that is not written in Latin script.

    Only letters are considered. Digits (even those of another script, such as the
    Arabic-Indic digits), spaces, punctuation, symbols, emoji and combining marks
    are not letters, so they never make a string non-Latin: `'Hello, 123!'` is Latin
    and a string without any letter gives False. Letters of no particular script
    (for example mathematical alphanumeric symbols) are ignored as well. A string is
    flagged as soon as one letter of any other script is present, so mixed text such
    as Latin followed by Cyrillic gives True.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one letter that belongs to a script other
        than Latin, False otherwise (including for an empty string).
    """
    return regex.search(_NON_LATIN_LETTER_PATTERN, s, flags=regex.V1) is not None


def containsLatinChar(s: str) -> bool:
    """Check whether a string contains a Latin character.

    Covers the Latin script as a whole (basic letters, accented letters and the extended
    blocks), the script of English, French, Vietnamese and many other languages. Digits,
    spaces and punctuation belong to no script and never match.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Latin character, False otherwise.
    """
    return containsScriptChar(s, 'Latin')


def containsGreekChar(s: str) -> bool:
    """Check whether a string contains a Greek character.

    Covers modern (monotonic) and polytonic Greek letters.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Greek character, False otherwise.
    """
    return containsScriptChar(s, 'Greek')


def containsCyrillicChar(s: str) -> bool:
    """Check whether a string contains a Cyrillic character.

    Script of Russian, Ukrainian, Bulgarian, Serbian and many other languages.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Cyrillic character, False otherwise.
    """
    return containsScriptChar(s, 'Cyrillic')


def containsArmenianChar(s: str) -> bool:
    """Check whether a string contains an Armenian character.

    Script of the Armenian language.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Armenian character, False otherwise.
    """
    return containsScriptChar(s, 'Armenian')


def containsGeorgianChar(s: str) -> bool:
    """Check whether a string contains a Georgian character.

    Script of the Georgian language.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Georgian character, False otherwise.
    """
    return containsScriptChar(s, 'Georgian')


def containsHebrewChar(s: str) -> bool:
    """Check whether a string contains a Hebrew character.

    Script of Hebrew and Yiddish.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Hebrew character, False otherwise.
    """
    return containsScriptChar(s, 'Hebrew')


def containsArabicChar(s: str) -> bool:
    """Check whether a string contains an Arabic character.

    Script of Arabic, Persian, Urdu and many other languages. The Arabic-Indic digits
    match, but the Arabic comma belongs to no script and does not.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Arabic character, False otherwise.
    """
    return containsScriptChar(s, 'Arabic')


def containsIndianChar(s: str) -> bool:
    """Check whether a string contains an Indian-script character.

    Checks the main scripts of India: Devanagari (Hindi, Marathi, Sanskrit), Bengali,
    Gurmukhi (Punjabi), Gujarati, Oriya (Odia), Tamil, Telugu, Kannada, Malayalam,
    Ol Chiki (Santali) and Meetei Mayek (Meitei). Arabic (used for Urdu) is not
    included, see `containsArabicChar`. The danda punctuation marks belong to no
    script and do not match. Use `containsScriptChar` for any other script of the
    Indian subcontinent.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Indian-script character, False otherwise.
    """
    return containsScriptChar(s, *INDIAN_SCRIPTS)


def containsThaiChar(s: str) -> bool:
    """Check whether a string contains a Thai character.

    Script of the Thai language. The baht currency sign belongs to no script and does
    not match.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Thai character, False otherwise.
    """
    return containsScriptChar(s, 'Thai')


def containsKhmerChar(s: str) -> bool:
    """Check whether a string contains a Khmer character.

    Script of the Khmer language.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Khmer character, False otherwise.
    """
    return containsScriptChar(s, 'Khmer')


def containsLaoChar(s: str) -> bool:
    """Check whether a string contains a Lao character.

    Script of the Lao language.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Lao character, False otherwise.
    """
    return containsScriptChar(s, 'Lao')


def containsMyanmarChar(s: str) -> bool:
    """Check whether a string contains a Myanmar character.

    Script of Burmese and several other languages of Myanmar.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Myanmar character, False otherwise.
    """
    return containsScriptChar(s, 'Myanmar')


def containsTibetanChar(s: str) -> bool:
    """Check whether a string contains a Tibetan character.

    Script of Tibetan and Dzongkha.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Tibetan character, False otherwise.
    """
    return containsScriptChar(s, 'Tibetan')


def containsSinhalaChar(s: str) -> bool:
    """Check whether a string contains a Sinhala character.

    Script of the Sinhala language.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Sinhala character, False otherwise.
    """
    return containsScriptChar(s, 'Sinhala')


def containsEthiopicChar(s: str) -> bool:
    """Check whether a string contains an Ethiopic character.

    Script of Amharic, Tigrinya and other languages of the Horn of Africa.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Ethiopic character, False otherwise.
    """
    return containsScriptChar(s, 'Ethiopic')


def containsMongolianChar(s: str) -> bool:
    """Check whether a string contains a Mongolian character.

    Covers the traditional Mongolian script. Mongolian written in Cyrillic is detected
    by `containsCyrillicChar`.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Mongolian character, False otherwise.
    """
    return containsScriptChar(s, 'Mongolian')


def containsKoreanChar(s: str) -> bool:
    """Check whether a string contains a Korean (Hangul) character.

    Detects Hangul only. Hanja (Chinese characters used in Korean text) are not
    detected, see `containsChineseChar`.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Korean (Hangul) character, False otherwise.
    """
    return containsScriptChar(s, 'Hangul')


def containsJapaneseChar(s: str) -> bool:
    """Check whether a string contains a Japanese (kana) character.

    Detects kana only (Hiragana and Katakana). Kanji are Han characters shared with
    Chinese, so they cannot tell Japanese from Chinese and are not detected, see
    `containsChineseChar`. The prolonged sound mark and the middle dot belong to no
    script and do not match.

    Args:
        s: String to scan.

    Returns:
        True if `s` contains at least one Japanese (kana) character, False otherwise.
    """
    return containsScriptChar(s, *JAPANESE_SCRIPTS)
