import numpy as np


class FinerTimeSplit:
    """Time series cross-validator.

    Provide train/test indices to split data in train/test sets.
    The corresponding training set consists only of observations that occurred prior
    to the observation that forms the test set. Thus, no future observations
    can be used in constructing the forecast.


    Parameters
    ----------
    n_splits: int
        Number of splits.

    horizon: int
        Number of steps ahead to make the forecast for.

    between_split_lag: int
        Number of observations between individual splits.
    """

    def __init__(self, n_splits=10, horizon=10, between_split_lag=None):
        self.n_splits = n_splits
        self.horizon = horizon
        self.between_split_lag = between_split_lag

    def split(self, X, y=None, groups=None):
        """Generate indices to split the data into training and test sets.

        Similar to scikit-learn API split. It takes n_splits*horizon from the tail of the
        data and use it for sequential generator of train/test indices.

        Parameters
        ----------
        X : array-like
            Data container to be splitted to train and test data
        y : Any
            ignored
        groups : Any
            ignored

        Yields
        -------
        int
            The next index to split the data into training and test set in a cross-validation.
        """
        return self._split(X)

    def _split(self, data):
        """Generate indices to split the data into training and test sets.

        Similar to scikit-learn API split. It takes n_splits*horizon from the tail of the
        data and use it for sequential generator of train/test indices.

        Parameters
        ----------
        data: array-like
            Data container to be splitted to train and test data

        Yields
        ------
        int
            The next index to split the data into training and test set in a cross-validation.
        """
        try:
            n_samples = len(data)
        except TypeError as exc:
            raise TypeError(
                f"Data must be an array-like object, but it does not seem to be the case. "
                f"You provided {data}"
            ) from exc

        if (self.between_split_lag is not None and self.between_split_lag < 1) or self.horizon < 1:
            raise ValueError(
                f"`between_split_lag`({self.between_split_lag} and "
                f"`horizon`({self.horizon}) have to be greater than 1'"
            )

        max_obs = (
            self.horizon if self.between_split_lag is None else max(self.horizon, self.between_split_lag)
        )
        if n_samples < self.n_splits * max_obs:
            raise ValueError(
                f"Cannot have number of samples({n_samples}) lower than the number "
                f"of `n_splits`({self.n_splits}) * `horizon`({self.horizon}),"
                f"or `n_splits`({self.n_splits}) * `between_split_lag`({self.between_split_lag}) "
                f"if you provided `between_split_lag`"
            )

        indices = np.arange(n_samples)
        if self.between_split_lag is not None:
            test_starts = range(
                n_samples
                - (self.between_split_lag * self.n_splits)
                - (self.horizon - self.between_split_lag),
                n_samples - (self.horizon - self.between_split_lag),
                self.between_split_lag,
            )
        else:
            test_starts = range(n_samples - (self.horizon * self.n_splits), n_samples, self.horizon)

        for test_start in test_starts:
            yield (
                indices[:test_start],
                indices[test_start : test_start + self.horizon],
            )

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits regarles of provided parameters

        Returns
        -------
        int
            Number of splits
        """
        return self.n_splits
