
import numpy as np
from collections.abc import Iterable

def flatten(obj):
    """
    Recursively flatten nested containers (list, tuple, set, dict, np.ndarray)
    into a flat list of elements.

    Dicts are flattened by their values (keys are discarded).
    Non-iterables (and strings) are treated as atomic and returned as single items.
    """
    # Treat strings as atomic, not as iterable of characters
    if isinstance(obj, str):
        return [obj]

    iterable = None
    # If it's a dict, iterate over its values
    if isinstance(obj, dict):
        iterable = obj.values()
    # If it's a NumPy array, use its flat iterator
    elif isinstance(obj, np.ndarray):
        iterable = obj.flat
    # If it's any other iterable (list, tuple, set, generator, etc.)
    elif isinstance(obj, Iterable):
        iterable = obj

    # Base case: if there's nothing iterable to traverse, return the object itself
    if iterable is None:
        return [obj]

    # Recursive case: flatten each item in the iterable
    result = []
    for item in iterable:
        result.extend(flatten(item))
    return result

