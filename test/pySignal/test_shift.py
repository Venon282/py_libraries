import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import shift

def test_shift_typical_case():
    signal = np.array([0.1, 0.5, 0.9])
    result = shift(signal, shift_value=0.2)
    expected = np.array([0.3, 0.7, 1.1])
    np.testing.assert_allclose(result, expected)

def test_shift_no_shift():
    signal = np.array([0.1, 0.5, 0.9])
    result = shift(signal, shift_value=0.0)
    np.testing.assert_array_equal(result, signal)

def test_shift_empty_signal():
    signal = np.array([])
    result = shift(signal)
    np.testing.assert_array_equal(signal, result)