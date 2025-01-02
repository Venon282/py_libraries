import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import quantize

def test_quantize_typical_case():
    signal = np.array([0.1, 0.5, 0.9])
    result = quantize(signal, levels=5)
    expected = np.array([0.1, 0.5, 0.9])  # Assuming evenly spaced levels
    np.testing.assert_array_almost_equal(result, expected)

def test_quantize_single_level():
    signal = np.array([0.1, 0.5, 0.9])
    result = quantize(signal, levels=1)
    expected = np.array([0.5, 0.5, 0.5])  # Midpoint of the range
    np.testing.assert_array_almost_equal(result, expected)

def test_quantize_empty_signal():
    signal = np.array([])
    result = quantize(signal)
    np.testing.assert_array_equal(signal, result)