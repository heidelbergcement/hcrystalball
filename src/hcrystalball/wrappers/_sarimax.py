import logging
import sys

logging.captureWarnings(True)
sys_out = logging.StreamHandler(sys.__stdout__)
sys_out.setFormatter(logging.Formatter("%(asctime)s - statsmodels - %(levelname)s - %(message)s"))
logging.getLogger("py.warnings").addHandler(sys_out)

import pandas as pd
from pmdarima.arima import AutoARIMA, ARIMA

from hcrystalball.wrappers._base import TSModelWrapper
from hcrystalball.wrappers._base import tsmodel_wrapper_constructor_factory
from hcrystalball.utils import check_X_y, enforce_y_type, check_fit_before_predict


class SarimaxWrapper(TSModelWrapper):
    """Wrapper for `~pmdarima.arima.ARIMA` and `~pmdarima.arima.AutoARIMA`

    Search for optimal order of SARIMAX type model or instantiate one
    in case you provide specific order.

    Parameters
    ----------
    name : str
        Name of the model instance, used also as column name for returned prediction.

    conf_int: bool
        Whether confidence intervals should be also outputed.

    init_with_autoarima: bool
        Whether you want to leverage automated search of pmdarima.arima.AutoARIMA.

    autoarima_dict: dict
        If `init_with_autoarima` is set to True, then `autoarima_dict` is used for instantiation
        of `~pmdarima.arima.AutoARIMA` class, thus it serves as configuration of AutoARIMA search.

    always_search_model: bool
        If `init_with_autoarima` is set to True and `always_search_model` is set to True, then
        the optimal model will be searched for during each `fit` call. On the other hand in most
        cases the desired behaviour is to search for optimal model just for first `fit` call and
        reused this already found model on subsequent `fit` calls (i.e. during cross validation).

    clip_predictions_lower: float
        Minimal value allowed for predictions - predictions will be clipped to this value.

    clip_predictions_upper: float
        Maximum value allowed for predictions - predictions will be clipped to this value.
    """

    @tsmodel_wrapper_constructor_factory(ARIMA)
    def __init__(
        self,
        name="sarimax",
        conf_int=False,
        init_with_autoarima=False,
        autoarima_dict=None,
        always_search_model=False,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):
        """This constructor will be modified at runtime to accept
        all parameters of the ARIMA class on top of the ones defined here!"""
        pass

    @staticmethod
    def _transform_data_to_tsmodel_input_format(X, y=None):
        """Trasnform data into Prophet.model required format

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : array_like, (1d)
            Target vector.

        Returns
        -------
        endog, exog
            endog - pandas.Series with target column if y is not None otherwise None.
            exog - numpy.ndarray with input features
        """
        if y is not None:
            endog = pd.Series(y, index=X.index)
        else:
            endog = None
        exog = X.values
        exog = None if exog.shape[1] == 0 else exog
        return endog, exog

    @staticmethod
    def _adjust_holidays(X):
        """Transform holiday to binary feature.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
        """
        return X.assign(
            **{
                col: X[col] != ""
                for col in X.filter(like="_holiday_").select_dtypes(include="object").columns
            }
        )

    @enforce_y_type
    @check_X_y
    def fit(self, X, y):
        """Transform input data to `pmdarima.arima.ARIMA` required format and fit the model.

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
        if X.filter(like="_holiday_").shape[1] > 0:
            X = self._adjust_holidays(X)
        endog, exog = self._transform_data_to_tsmodel_input_format(X, y)
        if self.init_with_autoarima or self.always_search_model:
            autoarima_params = self.autoarima_dict or {}
            found_params = AutoARIMA(**autoarima_params).fit(y=endog, exogenous=exog).model_.get_params()
            self.set_params(**found_params)
            self.init_with_autoarima = self.always_search_model
        elif self.order is None:
            raise ValueError("Parameter `order` must be set if `init_with_autoarima` is set to False!")
        self.model = self._init_tsmodel(ARIMA)
        self.model.fit(y=endog, exogenous=exog)
        self.fitted = True
        return self

    @check_fit_before_predict
    def predict(self, X):
        """Transform data to required format and provide predictions.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Prediction is stored in column with name being the `name` of the wrapper.
            If `conf_int` attribute is set to True, the returned DataFrame will have three columns,
            with the second and third (named 'name'_lower and 'name'_upper).
        """
        if X.filter(like="_holiday_").shape[1] > 0:
            X = self._adjust_holidays(X)
        _, exog = self._transform_data_to_tsmodel_input_format(X)
        preds, conf_ints = self.model.predict(n_periods=X.shape[0], exogenous=exog, return_conf_int=True)
        preds = pd.DataFrame(preds, index=X.index, columns=[self.name])
        if self.conf_int:
            conf_ints = pd.DataFrame(
                conf_ints,
                columns=[f"{self.name}_lower", f"{self.name}_upper"],
                index=X.index,
            )
            preds = pd.concat([preds, conf_ints], axis=1)

        # TODO make sure we do what we want for confidence intervals
        return self._clip_predictions(preds)


__all__ = ["SarimaxWrapper"]
