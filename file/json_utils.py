"""JSON serialization helpers for NumPy values."""

import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serializes NumPy scalars and arrays.

    NumPy floating and integer scalars are converted to their Python
    equivalents, and arrays to nested lists. Any other unsupported value is
    left to the default encoder, which raises `TypeError`.
    """

    def default(self, obj: object) -> Any:
        """Convert a NumPy value to a JSON-serializable one.

        Args:
            obj: Value to serialize.

        Returns:
            A Python float, int or list for NumPy values, or the result of the
            default encoder for anything else.

        Raises:
            TypeError: If `obj` is not a NumPy scalar or array.
        """
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)