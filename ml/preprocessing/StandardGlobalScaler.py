import numpy as np

class StandardGlobalScaler:
    """
    Globally standardize numeric data by removing the global mean and scaling to unit variance.
    Works on arbitrarily nested structures (lists, tuples, object arrays).

    The standard score of a sample x is calculated as:
        z = (x - mean) / std

    Parameters
    ----------
    copy : bool, default=True
        If True, work on a copy of the input data.
    with_mean : bool, default=True
        If True, center data before scaling.
    with_std : bool, default=True
        If True, scale data to unit variance.
    dtype : data-type, default=np.float64
        Precision for computations.

    Attributes
    ----------
    mean_ : float
        Global mean of fitted data (None if with_mean=False).
    var_ : float
        Global variance of fitted data (None if with_std=False).
    std_ : float
        Global standard deviation of fitted data (None if with_std=False).
    scale_ : float
        Scaling factor = 1/std_ (None if with_std=False).
    n_samples_seen_ : int
        Number of samples used to compute statistics.
    """

    def __init__(self, *, copy=True, with_mean=True, with_std=True, dtype=np.float64):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.dtype = dtype

    def _flatten(self, X):
        if isinstance(X, (list, tuple, np.ndarray)):
            for item in X:
                yield from self._flatten(item)
        else:
            yield X

    def fit(self, X, y=None):
        """
        Compute global mean and standard deviation from nested data X.

        Returns
        -------
        self
        """
        # Attempt numpy conversion
        try:
            arr = np.asarray(X, dtype=self.dtype)
            if arr.dtype != np.object_:
                flat = arr.ravel()
            else:
                raise ValueError
        except Exception:
            flat = np.fromiter(self._flatten(X), dtype=self.dtype)

        # Cast and compute
        self.n_samples_seen_ = flat.size
        if self.with_mean:
            self.mean_ = float(np.mean(flat))
        else:
            self.mean_ = None
        if self.with_std:
            self.var_ = float(np.var(flat, ddof=0))
            self.std_ = np.sqrt(self.var_) if self.var_ > 0 else 1.0
            self.scale_ = 1.0 / self.std_
        else:
            self.var_ = None
            self.std_ = None
            self.scale_ = None
        return self

    def _transform_value(self, value):
        val = self.dtype(value)
        if self.with_mean:
            val = val - self.mean_
        if self.with_std:
            val = val * self.scale_
        return val

    def _transform_recursive(self, X):
        if isinstance(X, list):
            return [self._transform_recursive(item) for item in X]
        if isinstance(X, tuple):
            return tuple(self._transform_recursive(item) for item in X)
        if isinstance(X, np.ndarray) and X.dtype == np.object_:
            return np.array([self._transform_recursive(item) for item in X], dtype=object)
        try:
            return self._transform_value(X)
        except Exception:
            raise ValueError("Non-numeric data encountered in transform.")

    def transform(self, X):
        """
        Standardize nested data X using fitted statistics.

        Returns
        -------
        Transformed data, same structure or numpy array.
        """
        try:
            arr = np.asarray(X, dtype=self.dtype)
            result = arr.copy() if self.copy else arr
            if self.with_mean:
                result = result - self.mean_
            if self.with_std:
                result = result * self.scale_
            return result
        except Exception:
            return self._transform_recursive(X)

    def inverse_transform(self, X):
        """
        Reverse standardization of nested data X.

        Returns
        -------
        Original-scale data, same structure or numpy array.
        """
        def _inv_val(val):
            v = self.dtype(val)
            if self.with_std:
                v = v / self.scale_
            if self.with_mean:
                v = v + self.mean_
            return v

        def _inv_rec(Y):
            if isinstance(Y, list):
                return [_inv_rec(item) for item in Y]
            if isinstance(Y, tuple):
                return tuple(_inv_rec(item) for item in Y)
            if isinstance(Y, np.ndarray) and Y.dtype == np.object_:
                return np.array([_inv_rec(item) for item in Y], dtype=object)
            try:
                return _inv_val(Y)
            except Exception:
                raise ValueError("Non-numeric data encountered in inverse_transform.")

        try:
            arr = np.asarray(X, dtype=self.dtype)
            result = arr.copy() if self.copy else arr
            if self.with_std:
                result = result / self.scale_
            if self.with_mean:
                result = result + self.mean_
            return result
        except Exception:
            return _inv_rec(X)

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y).transform(X)
