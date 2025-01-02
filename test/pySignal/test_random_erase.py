import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import random_erase

def test_random_erase_typical_case():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    erase_fraction = 0.4
    result = random_erase(signal, erase_fraction=erase_fraction)
    assert np.sum(result == 0) == int(len(signal) * erase_fraction)

def test_random_erase_no_erase():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = random_erase(signal, erase_fraction=0.0)
    np.testing.assert_array_equal(result, signal)

def test_random_erase_full_erase():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = random_erase(signal, erase_fraction=1.0)
    assert np.all(result == 0)

def test_random_erase_empty_signal():
    signal = np.array([])
    result = random_erase(signal)
    np.testing.assert_array_equal(signal, result)