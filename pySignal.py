import numpy as np

def add_noise(signal, noise_level=0.05):
    """Add random Gaussian noise to the signal."""
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def scale(signal, scale_factor=1.2):
    """Scale the signal by a random factor."""
    return signal * scale_factor

def shift(signal, shift_value=0.2):
    """Shift the signal values by a constant."""
    return signal + shift_value

def flip(signal):
    """Flip the signal (mirror)."""
    return np.flip(signal)

def clip(signal, min_value=0.0, max_value=1.0):
    """Clip the signal within a range."""
    return np.clip(signal, min_value, max_value)

def quantize(signal, levels=10):
    """Quantize the signal to a fixed number of levels."""
    if len(signal) == 0:
        return signal
    if levels == 1:
        return np.full_like(signal, np.mean([np.min(signal), np.max(signal)]))
    min_val, max_val = np.min(signal), np.max(signal)
    interval = np.linspace(min_val, max_val, levels)
    indices = np.digitize(signal, interval) - 1
    return interval[indices]

def random_erase(signal, erase_fraction=0.1):
    """Randomly set a fraction of the signal to zero."""
    signal_copy = signal.copy()
    num_erase = int(len(signal_copy) * erase_fraction)
    erase_indices = np.random.choice(len(signal_copy), num_erase, replace=False)
    signal_copy[erase_indices] = 0
    return signal_copy

def resample(signal, target_length):
    """Resample the signal to a different length."""
    if signal.size == 0:
        return np.array([])
    return np.interp(
        np.linspace(0, len(signal) - 1, target_length),
        np.arange(len(signal)),
        signal
    )

def polynomial_distortion(signal, coefficients=[1, 0.5, -0.2]):
    """Apply polynomial distortion to the signal."""
    distorted_signal = np.zeros_like(signal, dtype=float)
    for i, coef in enumerate(coefficients):
        distorted_signal += coef * (signal ** i)
    return distorted_signal

def jitter(signal, std=0.01):
    """Add small random jitters to the signal."""
    jitter = np.random.normal(0, std, size=signal.shape)
    return signal + jitter
