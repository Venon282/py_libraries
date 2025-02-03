import numpy as np
import pywt
from scipy.signal import cwt, morlet
from typing import Union

def noise(size, noise_level=0.05):
    """Generate random Gaussian noise."""
    return np.random.normal(0, noise_level, size)

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

def fourier_phase(signal, perturbation_level=0.1):
    """
    Perturb the phase component of the Fourier Transform.
    """
    transformed_signal = np.fft.fft(signal)
    magnitude = np.abs(transformed_signal)
    phase = np.angle(transformed_signal)
    phase += np.random.uniform(-perturbation_level, perturbation_level, size=phase.shape)
    perturbed_signal = magnitude * np.exp(1j * phase)
    return np.fft.ifft(perturbed_signal).real

def fourier_amplitude(signal, perturbation_level=0.1):
    """
    Perturb the amplitude component of the Fourier Transform.
    """
    transformed_signal = np.fft.fft(signal)
    magnitude = np.abs(transformed_signal)
    phase = np.angle(transformed_signal)
    magnitude += np.random.uniform(-perturbation_level, perturbation_level, size=magnitude.shape)
    perturbed_signal = magnitude * np.exp(1j * phase)
    return np.fft.ifft(perturbed_signal).real

def wavelet_augment(signal, wavelet_range=(5, 20), amplitude_range=(0.1, 0.5), width_range=(5, 20)):
    """Augment the signal by adding wavelets.

    Args:
        signal (np.ndarray): The input signal to augment.
        wavelet_range (tuple, optional): Range for the number of wavelets to add. Defaults to (5, 20).
        amplitude_range (tuple, optional): Range for the amplitude of the wavelets. Defaults to (0.1, 0.5).
        width_range (tuple, optional): Range for the width of the wavelets. Defaults to (5, 20).

    Returns:
        np.ndarray: The augmented signal.
    """
    augmented_signal = signal.copy()
    signal_length = len(signal)
    num_wavelets = np.random.randint(*wavelet_range) if isinstance(wavelet_range, tuple) else wavelet_range
    
    for _ in range(num_wavelets):
        # Randomly choose a center point for the wavelet
        center = np.random.randint(0, signal_length)
        
        # Random amplitude and width within specified ranges
        amplitude = np.random.uniform(*amplitude_range)
        width = np.random.uniform(*width_range)
        
        # Generate the wavelet (only real part is used)
        wavelet = amplitude * morlet(signal_length, w=width).real
        
        # Shift the wavelet to the selected center
        shifted_wavelet = np.roll(wavelet, center - signal_length // 2)
        
        # Add the wavelet to the signal
        augmented_signal += shifted_wavelet
    
    return augmented_signal

def derivative(y:Union[list, np.ndarray], x:Union[list, np.ndarray] = [0, 1]) -> np.array:
    dx = x[1] - x[0]
    return np.gradient(y, dx)

def custom_average(*args):
    """
    Compute the element-wise average of input arrays.
    
    Parameters:
        *args: A variable number of NumPy arrays of the same shape.
    
    Returns:
        np.ndarray: Element-wise average of the input arrays.
    """
    return (1 / len(args)) * np.sum(args, axis=0)

# def cwt(data, wavelet='cmor', scales=np.arange(1, 128)):
#     """
#     Calculates the coefficients of the continuous wavelet transform for a signal.
    
#     Args:
#         data: array-like, input signal.
#         wavelet: str, type of wavelet (example: "cmor" for complex Morlet).
#         scales: array-like, scale for coefficients.
#     Returns:
#         CWT coefficients as a 2D array.
#     """
#     cwt_coeffs, _ = pywt.cwt(data, scales, wavelet)
#     return np.abs(cwt_coeffs)
