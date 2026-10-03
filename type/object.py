"""Helpers for arbitrary Python objects."""

from collections.abc import Iterable
from typing import Any

import numpy as np


def flattenObject(obj: Any) -> list[Any]:
    """Recursively flatten nested containers into a flat list of elements.

    Handles list, tuple, set, dict, np.ndarray and any other iterable. Dicts are
    flattened by their values (keys are discarded). Non-iterables, strings, bytes and
    bytearrays are treated as atomic and returned as single items.

    Args:
        obj: The object to flatten.

    Returns:
        A flat list of all elements in the nested structure.
    """
    # Strings and bytes iterate into characters or ints, but must stay atomic
    if isinstance(obj, (str, bytes, bytearray)):
        return [obj]

    iterable: Iterable[Any] | None = None
    if isinstance(obj, dict):
        iterable = obj.values()
    elif isinstance(obj, np.ndarray):
        iterable = obj.flat
    elif isinstance(obj, Iterable):
        iterable = obj

    if iterable is None:
        return [obj]

    result = []
    for item in iterable:
        result.extend(flattenObject(item))
    return result
