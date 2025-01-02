import numpy as np

class SignalAugmentation:
    @staticmethod
    def add_noise(signal, noise_level=0.05):
        """Add random Gaussian noise to the signal."""
        noise = np.random.normal(0, noise_level, size=signal.shape)
        return signal + noise

    @staticmethod
    def scale(signal, scale_factor=1.2):
        """Scale the signal by a random factor."""
        return signal * scale_factor

    @staticmethod
    def shift(signal, shift_value=0.2):
        """Shift the signal values by a constant."""
        return signal + shift_value

    @staticmethod
    def flip(signal):
        """Flip the signal (mirror)."""
        return np.flip(signal)

    @staticmethod
    def clip(signal, min_value=0.0, max_value=1.0):
        """Clip the signal within a range."""
        return np.clip(signal, min_value, max_value)

    @staticmethod
    def quantize(signal, levels=10):
        """Quantize the signal to a fixed number of levels."""
        min_val, max_val = np.min(signal), np.max(signal)
        interval = np.linspace(min_val, max_val, levels)
        indices = np.digitize(signal, interval) - 1
        return interval[indices]

    @staticmethod
    def random_erase(signal, erase_fraction=0.1):
        """Randomly set a fraction of the signal to zero."""
        num_erase = int(len(signal) * erase_fraction)
        erase_indices = np.random.choice(len(signal), num_erase, replace=False)
        signal[erase_indices] = 0
        return signal

    @staticmethod
    def resample(signal, target_length):
        """Resample the signal to a different length."""
        return np.interp(
            np.linspace(0, len(signal) - 1, target_length),
            np.arange(len(signal)),
            signal
        )

    @staticmethod
    def polynomial_distortion(signal, coefficients=[1, 0.5, -0.2]):
        """Apply polynomial distortion to the signal."""
        distorted_signal = np.zeros_like(signal)
        for i, coef in enumerate(coefficients):
            distorted_signal += coef * (signal ** i)
        return distorted_signal

    @staticmethod
    def jitter(signal, std=0.01):
        """Add small random jitters to the signal."""
        jitter = np.random.normal(0, std, size=signal.shape)
        return signal + jitter
