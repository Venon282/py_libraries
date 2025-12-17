import numpy as np
# from ...lst import flatten
import logging
logger = logging.getLogger(__name__)

class MinMaxGlobalScaler:
    """
    Globally scale numeric data to a given feature range using the minimum and maximum
    computed over all fitted values. The fitted data can be arbitrarily nested (e.g., [[1], [2]], [1, 2, 3],
    or deeper structures), and transformed data can have a different shape.

    The transformation is:
        X_std = (X - global_min) / (global_max - global_min)
        X_scaled = X_std * (max - min) + min
    where (min, max) is the desired feature_range.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range for the transformed data.
    copy : bool, default=True
        If True, work on a copy of the input data.
    clip : bool, default=False
        If True, clip transformed values to the given feature_range.
    dtype : data-type, default=np.float64
        Desired floating point precision for the computations (e.g., np.float32 for lower precision).

    Attributes
    ----------
    data_min_ : float
        Global minimum of the fitted data.
    data_max_ : float
        Global maximum of the fitted data.
    data_range_ : float
        Global range (data_max_ - data_min_).
    scale_ : float
        Scaling factor computed as:
            (feature_range[1] - feature_range[0]) / (data_range_)
    min_ : float
        Offset computed as:
            feature_range[0] - data_min_ * scale_
    """

    def __init__(self, feature_range=(0, 1), copy=True, clip=False, dtype=np.float64):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.dtype = dtype

    def _flatten(self, X):
        """
        Recursively traverse nested lists/tuples (or object arrays) to yield numeric values.
        """
        
        if isinstance(X, (list, tuple, np.ndarray)):
            for item in X:
                yield from self._flatten(item)
        elif isinstance(X, np.ndarray) and X.dtype == np.object_:
            for item in X:
                yield from self._flatten(item)
        else:
            yield X
            
        

    def fit(self, X, y=None):
        """
        Compute the global minimum and maximum from the fitted data.

        Parameters
        ----------
        X : any nested structure of numbers
            Data to compute the global minimum and maximum.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        try:
            # Attempt to convert to a float array (works for regular numeric arrays)
            arr = np.asarray(X, dtype=self.dtype)
            if arr.dtype != np.object_:
                flat = arr.ravel()
            else:
                raise ValueError
        except Exception:
            # Fall back to recursive flattening for irregularly nested data.
            flat = np.array(list(self._flatten(X)), dtype=self.dtype)

        #self.data_min_ = min(flatten(X)) # np.min(flat)
        self.data_min_ = np.min(flat)
        logger.debug(f'Got min {self.data_min_}')
        #self.data_max_ = max(flatten(X)) # np.max(flat)
        self.data_max_ = np.max(flat)
        logger.debug(f'Got max {self.data_max_}')
        self.data_range_ = self.data_max_ - self.data_min_
        # Use a safe range to avoid division by zero
        safe_range = self.data_range_ if self.data_range_ != 0 else 1.0
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / safe_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def _transform_value(self, value):
        """
        Transform a single numeric value.
        """
        val = value * self.scale_ + self.min_
        if self.clip:
            val = np.clip(val, self.feature_range[0], self.feature_range[1])
        return val

    def _transform_recursive(self, X):
        """
        Recursively transform an arbitrarily nested structure elementwise.
        """
        if isinstance(X, (list, tuple)):
            return type(X)(self._transform_recursive(item) for item in X)
        elif isinstance(X, np.ndarray) and X.dtype == np.object_:
            return np.array([self._transform_recursive(item) for item in X], dtype=object)
        else:
            try:
                return self._transform_value(self.dtype(X))
            except TypeError:
                raise ValueError("Non-numeric data encountered in transform.")

    def transform(self, X):
        """
        Scale the input data using the global minimum and maximum computed during fit.

        Parameters
        ----------
        X : any nested structure of numbers
            Data to be transformed.

        Returns
        -------
        Transformed data in the same structure (or as a numpy array if possible).
        """
        try:
            # Try to convert X into a numeric numpy array
            arr = np.asarray(X, dtype=self.dtype)
            arr_transformed = arr * self.scale_ + self.min_
            if self.clip:
                arr_transformed = np.clip(arr_transformed, self.feature_range[0], self.feature_range[1])
            return arr_transformed.copy() if self.copy else arr_transformed
        except Exception:
            # Fall back to recursive elementwise transformation for irregular data.
            return self._transform_recursive(X)

    def inverse_transform(self, X):
        """
        Reverse the transformation.

        Parameters
        ----------
        X : any nested structure of numbers
            Transformed data.

        Returns
        -------
        Inverse transformed data in the same structure as the input.
        """
        def _inverse_value(val):
            return (val - self.min_) / self.scale_

        def _inverse_recursive(Y):
            if isinstance(Y, (list, tuple)):
                return type(Y)(_inverse_recursive(item) for item in Y)
            elif isinstance(Y, np.ndarray) and Y.dtype == np.object_:
                return np.array([_inverse_recursive(item) for item in Y], dtype=object)
            else:
                try:
                    return _inverse_value(self.dtype(Y))
                except TypeError:
                    raise ValueError("Non-numeric data encountered in inverse_transform.")

        try:
            arr = np.asarray(X, dtype=self.dtype)
            arr_inv = (arr - self.min_) / self.scale_
            return arr_inv.copy() if self.copy else arr_inv
        except Exception:
            return _inverse_recursive(X)

    def fit_transform(self, X, y=None):
        """
        Fit the scaler to X, then transform X.

        Parameters
        ----------
        X : any nested structure of numbers
            Data to fit and transform.
        y : None
            Ignored.

        Returns
        -------
        Transformed data in the same structure as X (or as a numpy array if possible).
        """
        self.fit(X, y)
        return self.transform(X)
