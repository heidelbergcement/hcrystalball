from abc import ABCMeta, abstractmethod
import pandas as pd
from hcrystalball.exceptions import InsufficientDataLengthError
from hcrystalball.wrappers._base import TSModelWrapper
from hcrystalball.wrappers._base import tsmodel_wrapper_constructor_factory
from hcrystalball.utils import check_X_y, enforce_y_type, check_fit_before_predict


class BaseSklearnWrapper(TSModelWrapper, metaclass=ABCMeta):
    def __reduce__(self):
        """Resorting to reduce for unpickling to sneak in
        a class definition created at runtime (see _ClassInitializer below)
        """
        return (_ClassInitializer(), (self.model_class,), self.__dict__)

    @abstractmethod
    def __init__(self):
        pass

    def _transform_data_to_tsmodel_input_format(self, X, y=None, horizon=None):
        """Trasnform data into Sklearn API required form and shift them.

        Shift is done in autoregressive format with `lags` columns based on prediction horizon which
        is derived from length of provided input data for `predict` call.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : array_like, (1d)
            Target vector

        horizon: int
            Number of steps used to shift the data

        Returns
        -------
        X, y
            X - pandas.DataFrame
            y - numpy.ndarray
        """
        if y is not None:
            y = self._y[self.lags + horizon - 1 :]
        X = self._add_lag_features(X, self._y, horizon)
        if X.filter(like="_holiday_").shape[1] > 0:
            X = self._adjust_holidays(X)
        X = X.astype(float)

        return X, y

    @staticmethod
    def _adjust_holidays(X):
        """Transform 'holiday' columns to binary feature.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features with 'holiday' column.

        Returns
        -------
        pandas.DataFrame
            Holiday feature in numeric form
        """
        return X.assign(**{col: X[col] != "" for col in X.filter(like="_holiday_").columns})

    @enforce_y_type
    @check_X_y
    def fit(self, X, y):
        """Store X in self._X and y in self._y and instantiate the model.

        Actual model fitting is done in `predict` method since the way model is fitted
        depends on `prediction` horizon which is known only during `predict` call.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : array_like, (1d)
            Target vector.

        Returns
        -------
        self
        """
        self._X, self._y = X, y
        self.model = self._init_tsmodel(self.model_class)
        self.fitted = True
        return self

    def _predict(self, X):
        """Transform stored training data to autoregressive form with `lags` features,
        fit the model and output prediction based on transformed X features.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Prediction is stored in column with name being the `name` of the wrapper.
        """
        X_fit, y_fit = self._transform_data_to_tsmodel_input_format(self._X, self._y, len(X))
        self.model = (
            self.model.fit(X_fit, y_fit, **self.fit_params)
            if self.fit_params
            else self.model.fit(X_fit, y_fit)
        )
        X_pred, _ = self._transform_data_to_tsmodel_input_format(X)
        pred = self.model.predict(X_pred)
        return pd.DataFrame(data=pred.reshape(-1, 1), columns=[self.name], index=X.index)

    @check_fit_before_predict
    def predict(self, X):
        """Predict using provided Sklearn compatible regressor.

        If `optimize_for_horizon` is set to True, then new model is created for
        each new horizon and fitted independently
        (i.e. len(X)=5 --> horizon=5 --> 5 models will be fitted).
        The final prediction is then combination of single point forecast of individual models
        for different horizons.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Prediction is stored in column with name being the `name` of the wrapper.
        """
        if len(X) > len(self._X) + 3:
            raise InsufficientDataLengthError(
                f"`X` must have at least {len(self._X) + 3} observations. Please provide valid data."
            )

        if self.optimize_for_horizon:
            preds = pd.concat(
                [self._predict(X.iloc[:index, :]).tail(1) for index in range(1, X.shape[0] + 1)]
            )
        else:
            preds = self._predict(X)
        preds.index = X.index
        return self._clip_predictions(preds)

    def _add_lag_features(self, X, y, horizon=None):
        """Transform input data X, y into autoregressive form - shift
        them appropriately based on horizon and create `lags` columns.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : array_like, (1d)
            Target vector.

        horizon : int
            length of X for `predict` method

        Returns
        -------
        pandas.DataFrame
            shifted dataframe with `lags` columns
        """
        lag_features = []
        shift = horizon if horizon else 0
        y = y if horizon else y[-(len(X) + self.lags - 1) :]

        for i in range(0, self.lags):
            lag_features.append(pd.Series(y, name=f"lag_{i}").shift(i + shift))

        X_lags = pd.concat(lag_features, axis=1)
        X_lags = X_lags if horizon else X_lags.dropna().reset_index(drop=True)
        X = X.reset_index(drop=True).join(X_lags).dropna()

        return X


def _get_sklearn_wrapper(model_cls):
    """Factory function returning the model specific SklearnWrapper with provided `model_cls` parameters.

    This function is required for sklearn compatibility since our SklearnWrapper
    need to have all parameters of `model_cls` set already during SklearnWrapper definition time.
    This factory function is not needed in case of
    other wrappers since the regressor is already part of the wrapper.

    Parameters
    ----------
    model_cls : class of sklearn compatible regressor
        i.e. LinearRegressor, GradientBoostingRegressor

    Example
    -------
    >>> from hcrystalball.wrappers._sklearn import _get_sklearn_wrapper
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> est = _get_sklearn_wrapper(RandomForestRegressor)(max_depth=6, clip_predictions_lower=0.)
    >>> est
    SklearnWrapper(bootstrap=True, ccp_alpha=0.0, clip_predictions_lower=0.0,
               clip_predictions_upper=None, criterion='mse', fit_params=None,
               lags=3, max_depth=6, max_features='auto', max_leaf_nodes=None,
               max_samples=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
               name='sklearn', oob_score=False, optimize_for_horizon=False,
               random_state=None, verbose=0, warm_start=False)

    Returns
    -------
    SklearnWrapper
    """

    class SklearnWrapper(BaseSklearnWrapper):
        """ "Wrapper for regressors compatible with Sklearn-API.

        This wrapper allows you use Sklearn-API regressors as autoregressive models
        for time series predictions. All model specific parameters will be passed to provided
        regressor class (even thought there is no explicit *model_kwargs).
        One side effect of the current implementation is very quick `fit` method since
        all of the actual model fitting is done in `predict` method resulting
        in longer inference time.

        Parameters
        ----------

        name : str
            Name of the model instance, used also as column name for returned prediction.

        lags: int
            Number of last observations of dependent variable used for modeling (lags = 2,X = yt-1, yt-2).

        fit_params: dict
            Parameters passed to `fit` method of the regressor, i.e. sample_weight.

        optimize_for_horizon: bool
            Whether new model should be fitter for each horizon (i.e. horizon 3 will produce 3 model,
            first for horizon 1, second for horizon 2 and third for horizon 3), this option ensures that
            autoregressive model is using for each horizon the most recent observation possible.

        clip_predictions_lower: float
            Minimal value allowed for predictions - predictions will be clipped to this value.

        clip_predictions_upper: float
            Maximum value allowed for predictions - predictions will be clipped to this value.
        """

        model_class = model_cls

        @tsmodel_wrapper_constructor_factory(model_cls)
        def __init__(
            self,
            lags=3,
            name="sklearn",
            fit_params=None,
            optimize_for_horizon=False,
            clip_predictions_lower=None,
            clip_predictions_upper=None,
        ):
            pass

    return SklearnWrapper


def get_sklearn_wrapper(model_cls, **model_params):
    """Factory function returning the model specific SklearnWrapper with provided `model_cls` parameters.

    This function is required for sklearn compatibility since our SklearnWrapper
    need to have all parameters of `model_cls` set already during SklearnWrapper definition time.
    This factory function is not needed in case of other wrappers since
    the regressor is already part of the wrapper.

    Parameters
    ----------
    model_cls : class of sklearn compatible regressor
        i.e. LinearRegressor, GradientBoostingRegressor

    model_params:
        `model_cls` specific parameters (e.g. max_depth) and/or
        SklearnWrapper specific parameters (e.g. clip_predictions_lower)

    Example
    -------
    >>> from hcrystalball.wrappers._sklearn import _get_sklearn_wrapper
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> est = get_sklearn_wrapper(RandomForestRegressor, max_depth=6, clip_predictions_lower=0.)
    >>> est
    SklearnWrapper(bootstrap=True, ccp_alpha=0.0, clip_predictions_lower=0.0,
               clip_predictions_upper=None, criterion='mse', fit_params=None,
               lags=3, max_depth=6, max_features='auto', max_leaf_nodes=None,
               max_samples=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
               name='sklearn', oob_score=False, optimize_for_horizon=False,
               random_state=None, verbose=0, warm_start=False)

    Returns
    -------
    SklearnWrapper
    """

    return _get_sklearn_wrapper(model_cls)(**model_params)


class _ClassInitializer:
    """Utility class helping with pickling/unpickling SklearnWrapper.

    This helper class is needed because the class definition of
    a wrapped sklearn model is only created at runtime, when the
    'get_sklearn_wrapper' function is invoked. Pickling/unpickilng such a class
    will fail since the object definition cannot be looked up when unpickling.
    This class serves as a dummy for unpickling, which creates an "empty" class
    and then replaces its code with a definition obtained from 'get_sklearn_wrapper'.
    https://stackoverflow.com/questions/19855156/whats-the-exact-usage-of-reduce-in-pickler
    """

    def __call__(self, model_class):
        obj = _ClassInitializer()
        obj.__class__ = _get_sklearn_wrapper(model_class)
        return obj


__all__ = ["get_sklearn_wrapper"]
