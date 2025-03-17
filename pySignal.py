import numpy as np
import pywt
from scipy.signal import cwt, morlet
from typing import Union

def noise(size, noise_level=0.05):
    """Generate random Gaussian noise."""
    return np.random.normal(0, noise_level, size)

def addNoise(signal, noise_level=0.05):
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

def randomErase(signal, erase_fraction=0.1):
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

def polynomialDistortion(signal, coefficients=[1, 0.5, -0.2]):
    """Apply polynomial distortion to the signal."""
    distorted_signal = np.zeros_like(signal, dtype=float)
    for i, coef in enumerate(coefficients):
        distorted_signal += coef * (signal ** i)
    return distorted_signal

def jitter(signal, std=0.01):
    """Add small random jitters to the signal."""
    jitter = np.random.normal(0, std, size=signal.shape)
    return signal + jitter

def fourierPhase(signal, perturbation_level=0.1):
    """
    Perturb the phase component of the Fourier Transform.
    """
    transformed_signal = np.fft.fft(signal)
    magnitude = np.abs(transformed_signal)
    phase = np.angle(transformed_signal)
    phase += np.random.uniform(-perturbation_level, perturbation_level, size=phase.shape)
    perturbed_signal = magnitude * np.exp(1j * phase)
    return np.fft.ifft(perturbed_signal).real

def fourierAmplitude(signal, perturbation_level=0.1):
    """
    Perturb the amplitude component of the Fourier Transform.
    """
    transformed_signal = np.fft.fft(signal)
    magnitude = np.abs(transformed_signal)
    phase = np.angle(transformed_signal)
    magnitude += np.random.uniform(-perturbation_level, perturbation_level, size=magnitude.shape)
    perturbed_signal = magnitude * np.exp(1j * phase)
    return np.fft.ifft(perturbed_signal).real

def waveletAugment(signal, wavelet_range=(5, 20), amplitude_range=(0.1, 0.5), width_range=(5, 20)):
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

def customAverage(*args):
    """
    Compute the element-wise average of input arrays.
    
    Parameters:
        *args: A variable number of NumPy arrays of the same shape.
    
    Returns:
        np.ndarray: Element-wise average of the input arrays.
    """
    return (1 / len(args)) * np.sum(args, axis=0)

def toBarcodeImage(signal: np.ndarray, n_rows: int = 64) -> np.ndarray:
    """
    Replicates a 1D spectral signal into a 2D image by repeating the signal along a new axis.
    
    Parameters:
        signal (np.ndarray): 1D array representing the spectral intensity.
        n_rows (int): Number of times to replicate the signal (default: 64).
    
    Returns:
        np.ndarray: 2D image with shape (n_rows, len(signal)).
    """
    return np.tile(signal, (n_rows, 1))


# Hankel Matrix Embedding
def toHankelMatrix(signal: np.ndarray, window_length: int) -> np.ndarray:
    """
    Constructs a Hankel (trajectory) matrix from the spectral signal.
    
    Parameters:
        signal (np.ndarray): 1D array representing the spectral intensity.
        window_length (int): The length of the sliding window to form the Hankel matrix.
    
    Returns:
        np.ndarray: 2D Hankel matrix embedding of the signal.
    """
    from scipy.linalg import hankel
    if window_length > len(signal):
        raise ValueError("window_length must be less than or equal to the length of the signal.")
    return hankel(signal[:window_length], signal[window_length-1:])


# Gramian Angular Field (GAF)
def toGramianAngularField(signal: np.ndarray, image_size: int = 64, method: str = 'summation') -> np.ndarray:
    """
    Computes the Gramian Angular Field (GAF) of a normalized spectral signal.
    
    Parameters:
        signal (np.ndarray): 1D array representing the spectral intensity.
                             It should be normalized to the range [0, 1].
        image_size (int): The size of the output GAF image (default: 64).
        method (str): Method for GAF computation; either 'summation' or 'difference' (default: 'summation').
    
    Returns:
        np.ndarray: 2D GAF image.
    """
    from pyts.image import GramianAngularField
    # Reshape signal for pyts (expects 2D input: [n_samples, series_length])
    x_reshaped = signal.reshape(1, -1)
    gaf = GramianAngularField(image_size=image_size, method=method)
    return gaf.fit_transform(x_reshaped)[0]


# Markov Transition Field (MTF)
def toMarkovTransitionField(signal: np.ndarray, image_size: int = 64, n_bins: int = 8) -> np.ndarray:
    """
    Computes the Markov Transition Field (MTF) for a normalized spectral signal.
    
    Parameters:
        signal (np.ndarray): 1D array representing the spectral intensity.
                             It should be normalized to the range [0, 1].
        image_size (int): The size of the output MTF image (default: 64).
        n_bins (int): Number of bins to discretize the signal (default: 8).
    
    Returns:
        np.ndarray: 2D MTF image.
    """
    from pyts.image import MarkovTransitionField
    x_reshaped = signal.reshape(1, -1)
    mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
    return mtf.fit_transform(x_reshaped)[0]


# 5. Recurrence Plot (RP)
def toRecurrentPlot(signal: np.ndarray, threshold: str = 'point', percentage: float = 10) -> np.ndarray:
    """
    Computes the Recurrence Plot (RP) of a spectral signal.
    
    Parameters:
        signal (np.ndarray): 1D array representing the spectral intensity.
        threshold (str or float): Thresholding method or value. If 'point', uses a percentage-based threshold.
        percentage (float): The percentage for thresholding when threshold is set to 'point' (default: 10).
    
    Returns:
        np.ndarray: 2D recurrence plot image (typically binary).
    """
    from pyts.image import RecurrencePlot
    x_reshaped = signal.reshape(1, -1)
    rp = RecurrencePlot(threshold=threshold, percentage=percentage)
    return rp.fit_transform(x_reshaped)[0]


# 6. Continuous Wavelet Transform (CWT) Scalogram
def toContinuousWaveletTransform(signal: np.ndarray, scales: np.ndarray = None, wavelet: str = 'morl'):
    """
    Computes the Continuous Wavelet Transform (CWT) scalogram of the spectral signal.
    
    Parameters:
        signal (np.ndarray): 1D array representing the spectral intensity.
        scales (np.ndarray): 1D array of scales to use. If None, defaults to np.arange(1, 64).
        wavelet (str): The type of wavelet to use (default: 'morl').
    
    Returns:
        tuple: A tuple (coefficients, frequencies), where:
               - coefficients (np.ndarray): 2D array of CWT coefficients.
               - frequencies (np.ndarray): 1D array of corresponding frequencies (or scales).
    """
    import pywt
    if scales is None:
        scales = np.arange(1, len(signal))
    return pywt.cwt(signal, scales, wavelet)


# 7. Self-Similarity (Distance) Matrix
def toSelfSimilarityDistanceMatrix(signal: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Computes the self-similarity (distance) matrix of the spectral signal.
    
    Parameters:
        signal (np.ndarray): 1D array representing the spectral intensity.
        metric (str): The distance metric to use (default: 'euclidean').
    
    Returns:
        np.ndarray: 2D self-similarity (distance) matrix.
    """
    from scipy.spatial.distance import pdist, squareform
    # Reshape signal to ensure it is 2D (each point as a 1D vector)
    return squareform(pdist(signal.reshape(-1, 1), metric=metric))

