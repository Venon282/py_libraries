import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import add_noise

def test_add_noise_typical_case():
    signal = np.array([0.5, 0.5, 0.5])
    noise_level = 0.1
    result = add_noise(signal, noise_level)
    assert result.shape == signal.shape
    assert not np.allclose(signal, result)

def test_add_noise_no_noise():
    signal = np.array([0.5, 0.5, 0.5])
    result = add_noise(signal, noise_level=0.0)
    np.testing.assert_array_equal(signal, result)

def test_add_noise_large_noise():
    signal = np.array([0.5, 0.5, 0.5])
    result = add_noise(signal, noise_level=1.0)
    assert result.shape == signal.shape

def test_add_noise_empty_signal():
    signal = np.array([])
    result = add_noise(signal)
    np.testing.assert_array_equal(signal, result)