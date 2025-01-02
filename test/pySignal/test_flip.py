import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import flip

def test_flip_typical_case():
    signal = np.array([1, 2, 3, 4, 5])
    result = flip(signal)
    expected = np.array([5, 4, 3, 2, 1])
    np.testing.assert_array_equal(result, expected)

def test_flip_empty_signal():
    signal = np.array([])
    result = flip(signal)
    np.testing.assert_array_equal(signal, result)