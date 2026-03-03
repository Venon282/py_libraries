
import numpy as np
import pywt
from scipy.signal import cwt, morlet
from typing import Union

def addNoise(signals:list|np.ndarray, noise_level:float|np.ndarray=0.05):
    """Add random Gaussian noise to the signals."""
    noise = np.random.normal(0, noise_level, size=signals.shape)
    return signals + noise

def scale(signals:list|np.ndarray, scale_factor:float|np.ndarray=1.2):
    """Scale the signals by a factor."""
    return signals * scale_factor

def shift(signals:list|np.ndarray, shift_value:float|np.ndarray=0.2):
    """Shift the signals values by a constant."""
    return signals + shift_value

# todo rebuild below function for allow list of signal
# def quantize(signals:list|np.ndarray, levels=10):
#     """Quantize the signals to a fixed number of levels."""
#     if len(signals) == 0:
#         return signals
    
#     if levels == 1:
#         return np.full_like(signals, np.mean([np.min(signals), np.max(signals)]))
    
#     min_val, max_val = np.min(signals), np.max(signals)
    
#     interval = np.linspace(min_val, max_val, levels)
#     indices = np.digitize(signals, interval) - 1
#     return interval[indices]

# def randomErase(signals:list|np.ndarray, erase_fraction:float=0.1, erase_value=0):
#     """Randomly set a fraction of the signals to zero."""
#     signal_copy = signals.copy()
#     num_erase = int(len(signal_copy) * erase_fraction)
#     erase_indices = np.random.choice(len(signal_copy), num_erase, replace=False)
#     signal_copy[erase_indices] = erase_value
#     return signal_copy

# def resample(signals:list|np.ndarray, target_length):
#     """Resample the signals to a different length."""
#     if signals.size == 0:
#         return np.array([])
    
#     return np.interp(
#         np.linspace(0, len(signals) - 1, target_length),
#         np.arange(len(signals)),
#         signals
#     )

# def polynomialDistortion(signals:list|np.ndarray, coefficients=[1, 0.5, -0.2]):
#     """Apply polynomial distortion to the signals."""
#     distorted_signal = np.zeros_like(signals, dtype=float)
#     for i, coef in enumerate(coefficients):
#         distorted_signal += coef * (signals ** i)
#     return distorted_signal

# def fourierPhase(signals:list|np.ndarray, perturbation_level=0.1):
#     """
#     Perturb the phase component of the Fourier Transform.
#     """
#     transformed_signal = np.fft.fft(signals)
#     magnitude = np.abs(transformed_signal)
#     phase = np.angle(transformed_signal)
#     phase += np.random.uniform(-perturbation_level, perturbation_level, size=phase.shape)
#     perturbed_signal = magnitude * np.exp(1j * phase)
#     return np.fft.ifft(perturbed_signal).real

# def fourierAmplitude(signals:list|np.ndarray, perturbation_level=0.1):
#     """
#     Perturb the amplitude component of the Fourier Transform.
#     """
#     transformed_signal = np.fft.fft(signals)
#     magnitude = np.abs(transformed_signal)
#     phase = np.angle(transformed_signal)
#     magnitude += np.random.uniform(-perturbation_level, perturbation_level, size=magnitude.shape)
#     perturbed_signal = magnitude * np.exp(1j * phase)
#     return np.fft.ifft(perturbed_signal).real

# def waveletAugment(signals:list|np.ndarray, wavelet_range=(5, 20), amplitude_range=(0.1, 0.5), width_range=(5, 20)):
#     """Augment the signals by adding wavelets.

#     Args:
#         signals (np.ndarray): The input signals to augment.
#         wavelet_range (tuple, optional): Range for the number of wavelets to add. Defaults to (5, 20).
#         amplitude_range (tuple, optional): Range for the amplitude of the wavelets. Defaults to (0.1, 0.5).
#         width_range (tuple, optional): Range for the width of the wavelets. Defaults to (5, 20).

#     Returns:
#         np.ndarray: The augmented signals.
#     """
#     augmented_signal = signals.copy()
#     signal_length = len(signals)
#     num_wavelets = np.random.randint(*wavelet_range) if isinstance(wavelet_range, tuple) else wavelet_range

#     for _ in range(num_wavelets):
#         # Randomly choose a center point for the wavelet
#         center = np.random.randint(0, signal_length)
        
#         # Random amplitude and width within specified ranges
#         amplitude = np.random.uniform(*amplitude_range)
#         width = np.random.uniform(*width_range)
        
#         # Generate the wavelet (only real part is used)
#         wavelet = amplitude * morlet(signal_length, w=width).real
        
#         # Shift the wavelet to the selected center
#         shifted_wavelet = np.roll(wavelet, center - signal_length // 2)
        
#         # Add the wavelet to the signals
#         augmented_signal += shifted_wavelet
    
#     return augmented_signal
# def derivative(y:Union[list, np.ndarray], x:Union[list, np.ndarray] = [0, 1]) -> np.array:
#     dx = x[1] - x[0]
#     return np.gradient(y, dx)
# def toBarcodeImage(signals:list|np.ndarray, n_rows: int = 64) -> np.ndarray:
#     """
#     Replicates a 1D spectral signals into a 2D image by repeating the signals along a new axis.
    
#     Parameters:
#         signals (np.ndarray): 1D array representing the spectral intensity.
#         n_rows (int): Number of times to replicate the signals (default: 64).
    
#     Returns:
#         np.ndarray: 2D image with shape (n_rows, len(signals:list|np.ndarray)).
#     """
#     return np.tile(signals, (n_rows, 1))


# # Hankel Matrix Embedding
# def toHankelMatrix(signals:list|np.ndarray, window_length: int) -> np.ndarray:
#     """
#     Constructs a Hankel (trajectory) matrix from the spectral signals.
    
#     Parameters:
#         signals (np.ndarray): 1D array representing the spectral intensity.
#         window_length (int): The length of the sliding window to form the Hankel matrix.
    
#     Returns:
#         np.ndarray: 2D Hankel matrix embedding of the signals.
#     """
#     from scipy.linalg import hankel
#     if window_length > len(signals):
#         raise ValueError("window_length must be less than or equal to the length of the signals.")
#     return hankel(signals[:window_length], signals[window_length-1:])


# # Gramian Angular Field (GAF)
# def toGramianAngularField(signals:list|np.ndarray, image_size: int = 64, method: str = 'summation') -> np.ndarray:
#     """
#     Computes the Gramian Angular Field (GAF) of a normalized spectral signals.
    
#     Parameters:
#         signals (np.ndarray): 1D array representing the spectral intensity.
#                              It should be normalized to the range [0, 1].
#         image_size (int): The size of the output GAF image (default: 64).
#         method (str): Method for GAF computation; either 'summation' or 'difference' (default: 'summation').
    
#     Returns:
#         np.ndarray: 2D GAF image.
#     """
#     from pyts.image import GramianAngularField
#     # Reshape signals for pyts (expects 2D input: [n_samples, series_length])
#     x_reshaped = signals.reshape(1, -1)
#     gaf = GramianAngularField(image_size=image_size, method=method)
#     return gaf.fit_transform(x_reshaped)[0]


# # Markov Transition Field (MTF)
# def toMarkovTransitionField(signals:list|np.ndarray, image_size: int = 64, n_bins: int = 8) -> np.ndarray:
#     """
#     Computes the Markov Transition Field (MTF) for a normalized spectral signals.
    
#     Parameters:
#         signals (np.ndarray): 1D array representing the spectral intensity.
#                              It should be normalized to the range [0, 1].
#         image_size (int): The size of the output MTF image (default: 64).
#         n_bins (int): Number of bins to discretize the signals (default: 8).
    
#     Returns:
#         np.ndarray: 2D MTF image.
#     """
#     from pyts.image import MarkovTransitionField
#     x_reshaped = signals.reshape(1, -1)
#     mtf = MarkovTransitionField(image_size=image_size, n_bins=n_bins)
#     return mtf.fit_transform(x_reshaped)[0]


# # Recurrence Plot (RP)
# def toRecurrentPlot(signals:list|np.ndarray, threshold: str = 'point', percentage: float = 10) -> np.ndarray:
#     """
#     Computes the Recurrence Plot (RP) of a spectral signals.
    
#     Parameters:
#         signals (np.ndarray): 1D array representing the spectral intensity.
#         threshold (str or float): Thresholding method or value. If 'point', uses a percentage-based threshold.
#         percentage (float): The percentage for thresholding when threshold is set to 'point' (default: 10).
    
#     Returns:
#         np.ndarray: 2D recurrence plot image (typically binary).
#     """
#     from pyts.image import RecurrencePlot
#     x_reshaped = signals.reshape(1, -1)
#     rp = RecurrencePlot(threshold=threshold, percentage=percentage)
#     return rp.fit_transform(x_reshaped)[0]


# # Continuous Wavelet Transform (CWT) Scalogram
# def toContinuousWaveletTransform(signals:list|np.ndarray, scales: np.ndarray = None, wavelet: str = 'morl'):
#     """
#     Computes the Continuous Wavelet Transform (CWT) scalogram of the spectral signals.
    
#     Parameters:
#         signals (np.ndarray): 1D array representing the spectral intensity.
#         scales (np.ndarray): 1D array of scales to use. If None, defaults to np.arange(1, 64).
#         wavelet (str): The type of wavelet to use (default: 'morl').
    
#     Returns:
#         tuple: A tuple (coefficients, frequencies), where:
#                - coefficients (np.ndarray): 2D array of CWT coefficients.
#                - frequencies (np.ndarray): 1D array of corresponding frequencies (or scales).
#     """
#     import pywt
#     if scales is None:
#         scales = np.arange(1, len(signals))
#     return pywt.cwt(signals, scales, wavelet)


# # Self-Similarity (Distance) Matrix
# def toSelfSimilarityDistanceMatrix(signals:list|np.ndarray, metric: str = 'euclidean') -> np.ndarray:
#     """
#     Computes the self-similarity (distance) matrix of the spectral signals.
    
#     Parameters:
#         signals (np.ndarray): 1D array representing the spectral intensity.
#         metric (str): The distance metric to use (default: 'euclidean').
    
#     Returns:
#         np.ndarray: 2D self-similarity (distance) matrix.
#     """
#     from scipy.spatial.distance import pdist, squareform
#     # Reshape signals to ensure it is 2D (each point as a 1D vector)
#     return squareform(pdist(signals.reshape(-1, 1), metric=metric))

