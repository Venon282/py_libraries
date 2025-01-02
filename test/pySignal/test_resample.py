import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import resample

def test_resample_typical_case():
    signal = np.array([1, 2, 3, 4, 5])
    result = resample(signal, target_length=7)
    assert len(result) == 7

def test_resample_shorter_signal():
    signal = np.array([1, 2, 3, 4, 5])
    result = resample(signal, target_length=3)
    assert len(result) == 3

def test_resample_same_length():
    signal = np.array([1, 2, 3, 4, 5])
    result = resample(signal, target_length=5)
    np.testing.assert_array_equal(result, signal)

def test_resample_empty_signal():
    signal = np.array([])
    result = resample(signal, target_length=5)
    np.testing.assert_array_equal(result, signal)