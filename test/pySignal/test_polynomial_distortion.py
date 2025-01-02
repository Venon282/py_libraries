import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import polynomial_distortion

def test_polynomial_distortion_typical_case():
    signal = np.array([1, 2, 3])
    coefficients = [1, 0.5, -0.2]
    result = polynomial_distortion(signal, coefficients)
    expected = np.array([1.3, 1.2, 0.7])
    np.testing.assert_array_almost_equal(result, expected)

def test_polynomial_distortion_default_coefficients():
    signal = np.array([1, 2, 3])
    result = polynomial_distortion(signal)
    expected = np.array([1.3, 1.2, 0.7])  # Default is a linear transformation
    np.testing.assert_array_almost_equal(result, expected)

def test_polynomial_distortion_empty_signal():
    signal = np.array([])
    result = polynomial_distortion(signal)
    np.testing.assert_array_equal(signal, result)