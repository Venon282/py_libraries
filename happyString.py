from typing import List


def getHappyString( size: int, k: int) -> str:
    """get the kème happy string of size size

    Args:
        size (int): _description_
        k (int): _description_

    Returns:
        str: _description_
    """
    total = 3 * (2 ** (size - 1))
    if k > total:
        return ""
    k -= 1  # convert k to 0-index

    # Determine the first character.
    first_count = 2 ** (size - 1)
    if k < first_count:
        res = ['a']
    elif k < 2 * first_count:
        res = ['b']
        k -= first_count
    else:
        res = ['c']
        k -= 2 * first_count

    # Build the rest of the string.
    for i in range(1, size):
        count = 2 ** (size - i - 1)
        prev = res[-1]
        # Options are the two characters different from prev in lex order.
        options = [c for c in "abc" if c != prev]
        if k < count:
            res.append(options[0])
        else:
            res.append(options[1])
            k -= count

    return "".join(res)

def happyStrings(n: int) -> List[str]:
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
        for c in "abc":
            if not word or c != word[-1]:
                res.extend(happyStringsRec(word + c, n))
        return res
    
    return happyStringsRec("", n)

def isHappyString(string):
    if string[0] not in 'abc':
        return False
    
    for i in range(1, len(string)):
        if string[i] not in 'abc' or string[i] == string[i - 1]:
            return False
    return True