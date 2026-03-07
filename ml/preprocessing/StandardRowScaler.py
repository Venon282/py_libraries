import numpy as np

class StandardRowScaler:
    """
    Standardize each row of the input data by removing the mean and scaling to unit variance.

    The transformation is given by:
        X_scaled = (X - mean) / std

    Parameters
    ----------
    copy : bool, default=True
        Set to False to perform inplace row normalization if the input is already a numpy array.
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance (or unit standard deviation).

    Attributes
    ----------
    mean_ : ndarray of shape (n_samples,)
        Per-row mean seen in the data.
    scale_ : ndarray of shape (n_samples,)
        Per-row standard deviation seen in the data.
        If a row is constant (std=0), the scale is set to 1.0 to avoid division by zero.
    n_rows_seen_ : int
        Number of rows seen during fit.
    """
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        """
        Compute the mean and std for each row to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-row mean and standard deviation.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        X = np.array(X, copy=self.copy)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=1)
        else:
            self.mean_ = None

        if self.with_std:
            self.scale_ = np.std(X, axis=1)
            # Avoid division by zero: for constant rows, set the scale to 1.
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = None

        self.n_rows_seen_ = X.shape[0]
        return self

    def transform(self, X):
        """
        Standardize the data row-wise.

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

        # Apply centering
        if self.with_mean:
            X -= self.mean_[:, np.newaxis]

        # Apply scaling
        if self.with_std:
            X /= self.scale_[:, np.newaxis]

        return X

    def inverse_transform(self, X):
        """
        Undo the scaling of X according to the stored mean and std.

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

        # Reverse scaling
        if self.with_std:
            X *= self.scale_[:, np.newaxis]

        # Reverse centering
        if self.with_mean:
            X += self.mean_[:, np.newaxis]

        return X

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