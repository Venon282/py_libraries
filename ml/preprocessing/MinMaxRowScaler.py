import numpy as np

class MinMaxRowScaler:
    """
    Scale each row of the input data to a given range.

    The transformation is given by:
        X_std = (X - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1))
        X_scaled = X_std * (max - min) + min
    where (min, max) is the desired feature_range.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization if the input is already a numpy array.
    clip : bool, default=False
        If True, clip transformed values of held-out data to the provided feature_range.

    Attributes
    ----------
    data_min_ : ndarray of shape (n_samples,)
        Per-row minimum seen in the data.
    data_max_ : ndarray of shape (n_samples,)
        Per-row maximum seen in the data.
    data_range_ : ndarray of shape (n_samples,)
        Per-row range (data_max_ - data_min_) seen in the data.
    scale_ : ndarray of shape (n_samples,)
        Per-row scaling factor, computed as:
            (feature_range[1] - feature_range[0]) / (data_max_ - data_min_)
        (with division-by-zero handled for constant rows).
    min_ : ndarray of shape (n_samples,)
        Per-row adjustment for minimum, computed as:
            feature_range[0] - data_min_ * scale_
    n_rows_seen_ : int
        Number of rows seen during fit.
    """
    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def fit(self, X, y=None):
        """
        Compute the minimum and maximum for each row to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-row minimum and maximum.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.array(X, copy=self.copy)
        # Compute row-wise min and max
        self.data_min_ = np.min(X, axis=1)
        self.data_max_ = np.max(X, axis=1)
        self.data_range_ = self.data_max_ - self.data_min_
        # Avoid division by zero: for constant rows, set the range to 1.
        safe_range = np.where(self.data_range_ == 0, 1, self.data_range_)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / safe_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self.n_rows_seen_ = X.shape[0]
        return self

    def transform(self, X):
        """
        Scale the data row-wise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        X = np.array(X, copy=self.copy)
        if X.shape[0] != self.n_rows_seen_:
            raise ValueError("Number of rows in X does not match the fitted data.")
        # Apply the scaling row-wise using broadcasting
        X_scaled = X * self.scale_[:, np.newaxis] + self.min_[:, np.newaxis]
        if self.clip:
            X_scaled = np.clip(X_scaled, self.feature_range[0], self.feature_range[1])
        return X_scaled

    def inverse_transform(self, X):
        """
        Undo the scaling of X according to the feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be inverse transformed.

        Returns
        -------
        X_orig : ndarray of shape (n_samples, n_features)
            Original data before transformation.
        """
        X = np.array(X, copy=self.copy)
        if X.shape[0] != self.n_rows_seen_:
            raise ValueError("Number of rows in X does not match the fitted data.")
        X_orig = (X - self.min_[:, np.newaxis]) / self.scale_[:, np.newaxis]
        return X_orig

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X, y).transform(X)
