"""Helpers to display dictionaries."""

from collections.abc import Mapping


def displayDictionary(
    dct: Mapping[str, object], indent: str = '    ', index: int = 0
) -> str:
    """Recursively format a dictionary as a string with indentation.

    Each key is written on its own line, followed by its value. The keys of a
    nested `dict` are written on the following lines, one indentation level deeper.
    An empty nested `dict` leaves an empty line below its key. The string has no
    trailing newline.

    Args:
        dct: Dictionary to display.
        indent: String used for one indentation level. Default is 4 spaces.
        index: Current indentation level. Default is 0.

    Returns:
        String representation of the dictionary with indentation. It is empty for
        an empty dictionary.

    Example:
        >>> print(displayDictionary({'a': 1, 'b': {'c': 2}}))
        a: 1
        b:
            c: 2
    """
    result = []
    for key, value in dct.items():
        if isinstance(value, dict):
            result.append(f'{indent * index}{key}:')
            result.append(displayDictionary(value, indent, index + 1))
        else:
            result.append(f'{indent * index}{key}: {value}')
    return '\n'.join(result)
