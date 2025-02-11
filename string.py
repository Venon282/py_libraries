def prefixCount(words:list[str], pref:str):
    """
    :type words: List[str]
    :type pref: str
    :rtype: int
    """
    return sum([1 for word in words if word.startswith(pref)])

