import sys
import os
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, grandparent_dir)

import pytest
import numpy as np
from pySignal import clip

def test_clip_typical_case():
    signal = np.array([0.1, 0.5, 0.9, 1.2, -0.1])
    result = clip(signal, min_value=0.0, max_value=1.0)
    expected = np.array([0.1, 0.5, 0.9, 1.0, 0.0])
    np.testing.assert_array_equal(result, expected)

def test_clip_no_clipping():
    signal = np.array([0.5, 0.7, 0.9])
    result = clip(signal, min_value=0.0, max_value=1.0)
    np.testing.assert_array_equal(result, signal)

def test_clip_empty_signal():
    signal = np.array([])
    result = clip(signal)
    np.testing.assert_array_equal(signal, result)