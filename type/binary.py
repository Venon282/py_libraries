"""Helpers for binary number strings."""


def getLackingBinaries(lst: list[str], length: int | None = None) -> list[str]:
    """Return the `bin()` strings below 2**length that are missing from a list.

    Strings follow the `bin()` format: a `0b` prefix and no zero padding, so `'0b101'`
    stands for 5 and `'0b1'` for 1. Entries of `lst` in any other format never match.

    Args:
        lst: List of `bin()` strings to check against.
        length: Number of bits: the integers from 0 to 2**length - 1 are generated. If
            None, uses the length of the longest string in `lst` minus 2 (the `0b`
            prefix).

    Returns:
        The `bin()` strings of the integers below 2**length that are not in `lst`, in
        ascending numeric order.

    Raises:
        ValueError: If lst is empty and length is None.
    """
    if length is None:
        length = max(len(x) for x in lst) - 2
    return [bin(i) for i in range(2**length) if bin(i) not in lst]
