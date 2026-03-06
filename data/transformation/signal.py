
import numpy as np

def addNoise(signals: list | np.ndarray, noise_level: float | np.ndarray | list | tuple[float, float] = 0.05):
    """
    Add random Gaussian noise to the signals.
    when noise_level is:
        - float: same noise level for all signals
        - list/np.ndarray: a noise level defined per signal
        - tuple(min, max): randomly sample a noise level per signal in [min, max[
    """
    signals = np.asarray(signals)

    if isinstance(noise_level, tuple):
        if len(noise_level) != 2:
            raise ValueError('When noise_level is a tuple it must be (min_noise, max_noise)')
        noise_level = np.random.uniform(*noise_level, size=signals.shape[0])

    if isinstance(noise_level, (list, np.ndarray)):
        noise_level = np.asarray(noise_level)  # shape (N,)
        # Reshape to (N, 1, 1, ...) to broadcast against signals of shape (N, T, ...)
        noise_level = noise_level.reshape([-1] + [1] * (signals.ndim - 1))

    noise = np.random.normal(0, noise_level, size=signals.shape)
    return signals + noise

def scale(signals:list|np.ndarray, scale_factor:float|np.ndarray=1.2):
    """Scale the signals by a factor."""
    return signals * scale_factor

def shift(signals:list|np.ndarray, shift_value:float|np.ndarray=0.2):
    """Shift the signals values by a constant."""
    return signals + shift_value

def addWavelet(
    signals: list | np.ndarray, 
    wavelet_range:tuple[int,int]|list[int]|int=(5, 20), 
    amplitude_range:tuple[int,int]|list[int]|int=(0.1, 0.5), 
    width_range:tuple[int,int]|list[int]|int=(5, 20), 
    scale_range:tuple[int,int]|list[int]|int=1, 
    copy=True, 
    complete=True,
    rng: np.random.Generator | None = None
    ):
    """
    Add randomized real Morlet wavelets to one or multiple signals.

    Each wavelet is a Gaussian-windowed cosine:

        ψ(t) = A · exp(-0.5 * x^2) · cos(ω x)

    where:
        x = (t - center) * mapping_factor
        ω = width parameter
        A = amplitude

    Parameters
    ----------
    signals : array-like, shape (L,) or (N, L)
        Input signal(s). A 1D array is treated as a single signal.
        A 2D array must have shape (n_signals, signal_length).

    wavelet_range : int or tuple(int, int) or sequence of int, default=(5, 20)
        Number of wavelets added per signal.
        - int → fixed number per signal
        - tuple(min, max) → random integer in range per signal
        - sequence → explicit number per signal

    amplitude_range : int or tuple(int, int) or sequence of int,, default=(0.1, 0.5)
        Uniform range for peak amplitude scaling.
        Controls the vertical height of each wavelet.

    width_range : int or tuple(int, int) or sequence of int, default=(5, 20)
        Controls oscillation frequency inside the Gaussian envelope.
        Larger values → higher oscillation frequency.
        Smaller values → lower frequency.

    scale : int or tuple(int, int) or sequence of int, default=1
        Global scaling factor affecting time stretching.
        Larger scale → wider Gaussian envelope (longer wavelets).

    copy : bool, default=True
        If True, operate on a copy of input.
        If False, modify in-place.

    complete : bool, default=True
        If True, apply admissibility correction term
        (zero-mean Morlet wavelet).

    rng : numpy.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    ndarray
        Modified signal(s), preserving input dimensionality.
    """
    def getRandom(element, randFunc, size):
        # tuple mean between two bornes
        if isinstance(element, tuple):
            if len(element) != 2:
                raise ValueError(f'The tuple size must be 2 (min, max), not {len(element)}')
            return randFunc(element[0], element[1] + 1, size=size)
        # value mean we want it size time
        elif isinstance(element, (float, int)):
            return np.full(size, element)
        # a list mean the selection was made outside the function
        elif isinstance(element, (list, np.ndarray)):
            nonlocal n_signals
            if len(element) != n_signals:
                raise ValueError(f'When element is a {type(element)}, its size must be the same than signals')
            return np.array(element)
        else:
            raise TypeError(f'Unsupport type {type(element)} for element.')
        
    if amplitude_range[0] > amplitude_range[1]:
        raise ValueError("amplitude_range must be (min, max)")
    
    if copy:
        signals = np.array(signals, copy=True)
    else:
        signals = np.asarray(signals)

    # Standardize shape to (n_signals, signal_length)
    if signals.ndim == 1:
        signals = signals[np.newaxis, :]
    
    n_signals, signal_length = signals.shape
    
    if rng is None:
        rng = np.random.default_rng()
        
    num_wavelets_per_signal = getRandom(element=wavelet_range, randFunc=rng.integers, size=n_signals)
    max_wavelets = np.max(num_wavelets_per_signal)
    
    # Pre-create a time axis: shape (signal_length,)
    t = np.arange(signal_length)

    # Iterate up to the maximum number of wavelets needed
    for i in range(max_wavelets):
        # Only process signals that still need more wavelets
        mask = num_wavelets_per_signal > i
        active_count = np.sum(mask)
        
        # Vectorized generation of parameters for 'active' signals
        centers = rng.integers(0, signal_length, size=(active_count, 1))
        amplitudes = getRandom(element=amplitude_range, randFunc=rng.uniform, size=(active_count, 1))
        widths = getRandom(element=width_range, randFunc=rng.uniform, size=(active_count, 1)) 
        scales = getRandom(element=scale_range, randFunc=rng.uniform, size=(active_count, 1))
        
        mapping_factors = (scales * 4 * np.pi) / signal_length
        
        x_shift = (t - centers) * mapping_factors
        wavelets = np.cos(widths * x_shift)
        
        # Admissibility correction (complete Morlet)
        if complete:
            wavelets -= np.exp(-0.5 * (widths**2))
            
        wavelets *= np.exp(-0.5 * (x_shift**2)) * np.pi**(-0.25)
        wavelets = amplitudes * wavelets

        # Add to the active signals
        signals[mask] += wavelets

    return signals[0] if n_signals == 1 else signals

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

