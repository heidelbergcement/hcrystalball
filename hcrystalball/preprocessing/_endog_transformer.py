import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TargetTransformer(TransformerMixin, BaseEstimator):
    """Enable transformation of the target.

    Wrapper for applying an estimator to a transformed version of the target y
    and automatically transforming back predictions
    """

    def __init__(self, estimator, y_transformer, omit_inverse_transformation=False):
        self.estimator = estimator
        self.y_transformer = y_transformer
        self.omit_inverse_transformation = omit_inverse_transformation
        self.steps = self.estimator.steps if hasattr(self.estimator, "steps") else None

    def _reshape_2d(self, y):
        """Ensure correct array size

        Parameters
        ----------
        y : numpy.ndarray
            Target values.

        Returns
        -------
        numpy.ndarray
            Target values in 1d dimension
        """
        if y.ndim == 1:
            return y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        return y

    def _restore_shape(self, y):
        """Restores correct array shape

        Parameters
        ----------
        y : numpy.ndarray
            Target values.

        Returns
        -------
        numpy.ndarray
            Target values in 1d dimension if one negligible
        """
        if y.ndim == 2 and y.shape[1] == 1:
            return y.squeeze(axis=1)
        return y

    def fit(self, X, y=None):
        """Fit after reshaping and rescaling the target

        Reshape target to 2d, call fit_transform on 2d, return to 1d form
        and fit estimator on transformed target

        Parameters
        ----------
        X : Any
            Ignored.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        `TargetTransformer`
            Fitted target transformer
        """
        self._training_dim = y.ndim

        # things are made more complicated by the fact that sklearn
        # transformers expect 2D arrays. Thus need to reshape
        y_2d = self._reshape_2d(y)

        y_t = self.y_transformer.fit_transform(y_2d, y)

        # restore 1D if necessary
        y_t = self._restore_shape(y_t)

        # fit estimator on transformed target
        self.estimator.fit(X, y_t)
        return self

    def transform(self, X, y=None):
        """Transforms the features

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : Any
            Ignored.

        Returns
        -------
        numpy.ndarray
            Result of estimator transform
        """
        return self.estimator.transform(X)

    def _predict(self, X, y=None):
        pred_t = self.estimator.predict(X)

        if self.omit_inverse_transformation:
            return pred_t
        else:
            # again, transformer expects 2D input for doing inverse transform
            pred = self.y_transformer.inverse_transform(self._reshape_2d(pred_t))

        # if output is expected to be 1D, squeeze if necessary
        if self._training_dim == 1:
            return self._restore_shape(pred)
        return pred

    def predict(self, X, y=None):
        """Ensure correct estimator.predict with scaled target values

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        pandas.DataFrame
            Results of estimators prediction
        """
        preds = self._predict(X, y)
        name = (
            self.estimator.steps[-1][1].name if isinstance(self.estimator, Pipeline) else self.estimator.name
        )
        return pd.DataFrame(preds, index=X.index, columns=[name])

    def score(self, X, y=None):
        """Ensures correct estimator.score with scaled target values

        Parameters
        ----------
        X : numpy.ndarray
            Input features.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        Any
            Results of estimators score function
        """
        y_2d = self._reshape_2d(y)
        y_t = self.y_transformer.transform(y_2d)
        if self._training_dim == 1:
            y_t = self._restore_shape(y_t)
        return self.estimator.score(X, y_t)

    def named_steps(self):
        """Provide access to named steps for `~sklearn.pipeline.Pipeline`

        Returns
        -------
        dict
            Dictionary of steps
        """
        return dict(self.steps)
