import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import scale

def test_scale_typical_case():
    signal = np.array([1.0, 2.0, 3.0])
    result = scale(signal, scale_factor=2.0)
    np.testing.assert_array_equal(result, signal * 2.0)

def test_scale_no_scaling():
    signal = np.array([1.0, 2.0, 3.0])
    result = scale(signal, scale_factor=1.0)
    np.testing.assert_array_equal(result, signal)

def test_scale_negative_scaling():
    signal = np.array([1.0, 2.0, 3.0])
    result = scale(signal, scale_factor=-1.0)
    np.testing.assert_array_equal(result, signal * -1.0)

def test_scale_empty_signal():
    signal = np.array([])
    result = scale(signal)
    np.testing.assert_array_equal(signal, result)