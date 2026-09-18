from typing import Any


def displayDictionary(
    dct: dict[str, Any], indent: str = "    ", index: int = 0
) -> str:
    """Recursively format a dictionary as a string with indentation.

    Args:
        dct: Dictionary to display.
        indent: String used for indentation. Default is 4 spaces.
        index: Current indentation level. Default is 0.

    Returns:
        String representation of the dictionary with indentation.
    """
    result = []
    for key, value in dct.items():
        if isinstance(value, dict):
            result.append(f"{indent * index}{key}:")
            result.append(displayDictionary(value, indent, index + 1))
        else:
            result.append(f"{indent * index}{key}: {value}")
    return "\n".join(result)
