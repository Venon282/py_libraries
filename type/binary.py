from typing import Any


def getLackingBinaries(lst: list[str], length: int | None = None) -> list[str]:
    """Return the binary numbers that are not present in the list at the defined length.

    Args:
        lst: List of binary strings to check against.
        length: Length of binary numbers to generate. If None, uses the maximum length
            of binary strings in the list minus 2.

    Returns:
        List of binary strings that are missing from the input list.
    """
    if length is None:
        length = max(len(x) for x in lst) - 2
    return [bin(i) for i in range(2**length) if bin(i) not in lst]
