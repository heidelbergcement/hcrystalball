import logging
import sys

logging.captureWarnings(True)
sys_out = logging.StreamHandler(sys.__stdout__)
sys_out.setFormatter(logging.Formatter("%(asctime)s - statsmodels - %(levelname)s - %(message)s"))
logging.getLogger("py.warnings").addHandler(sys_out)

from abc import ABCMeta, abstractmethod
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from hcrystalball.wrappers._base import TSModelWrapper
from hcrystalball.wrappers._base import tsmodel_wrapper_constructor_factory
from hcrystalball.utils import check_X_y
from hcrystalball.utils import enforce_y_type
from hcrystalball.utils import check_fit_before_predict


class BaseStatsmodelsForecastingWrapper(TSModelWrapper, metaclass=ABCMeta):
    """BaseWrapper for smoothing models from `~statsmodels.tsa.holtwinters`

    Currently supported ones are `~statsmodels.tsa.holtwinters.ExponentialSmoothing`,
    `~statsmodels.tsa.holtwinters.SimpleExpSmoothing`, `~statsmodels.tsa.holtwinters.Holt`
    """

    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def _transform_data_to_tsmodel_input_format(X, y=None):
        """Trasnform data into `statsmodels.tsa.api` required format

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : array_like, (1d)
            Target vector.

        Returns
        -------
        pandas.Series/int,
            If y is None, length of input to `predict` method is returned
            otherwise series with X.index in index and y in values
        """
        if y is not None:
            return pd.Series(y, index=X.index)
        else:
            return X.shape[0]

    @enforce_y_type
    @check_X_y
    def fit(self, X, y):
        """Transform data to `statsmodels.tsa.api` required format
        and fit the model.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : array_like, (1d)
            Target vector.

        Returns
        -------
        self
            Fitted model
        """
        endog = self._transform_data_to_tsmodel_input_format(X, y)
        self.model = self._init_tsmodel(self.model_cls, endog=endog)
        self.model = self.model.fit(**self.fit_params) if self.fit_params else self.model.fit()
        self.fitted = True
        return self

    @check_fit_before_predict
    def predict(self, X):
        """Transform data to `statsmodels.tsa.api` required format and provide predictions.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Prediction stored in column with name being the `name` of the wrapper.
        """
        horizon = self._transform_data_to_tsmodel_input_format(X)
        preds = self.model.forecast(horizon).to_frame(self.name)
        if hasattr(self.model, "prediction_intervals") and self.conf_int:
            preds = (
                self.model.prediction_intervals(horizon)
                .rename(columns=lambda x: f"{self.name}_" + x)
                .join(preds)
            )
        preds.index = X.index
        return self._clip_predictions(preds)


class ExponentialSmoothingWrapper(BaseStatsmodelsForecastingWrapper):
    """Wrapper for `~statsmodels.tsa.holtwinters.ExponentialSmoothing` (see other parameters there)

    Parameters
    ----------
    name: str
        Name of the model instance, used also as column name for returned prediction

    fit_params: dict
        Parameters passed to `~hcrystalball.wrappers.ExponentialSmoothingWrapper.fit` method of model.
        For more details see `statsmodels.tsa.holtwinters.ExponentialSmoothing.fit`

    clip_predictions_lower: float
        Minimal value allowed for predictions - predictions will be clipped to this value.

    clip_predictions_upper: float
        Maximum value allowed for predictions - predictions will be clipped to this value.
    """

    model_cls = ExponentialSmoothing

    @tsmodel_wrapper_constructor_factory(ExponentialSmoothing)
    def __init__(
        self,
        name="ExponentialSmoothing",
        fit_params=None,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):
        """This constructor will be modified at runtime to accept
        all parameters of the ExponentialSmoothing class on top of the ones defined here!"""
        pass


class SimpleSmoothingWrapper(BaseStatsmodelsForecastingWrapper):
    """Wrapper for `~statsmodels.tsa.holtwinters.SimpleExpSmoothing` (see other parameters there)

    Parameters
    ----------
    name: str
        Name of the model instance, used also as column name for returned prediction

    fit_params: dict
        Parameters passed to `~hcrystalball.wrappers.SimpleSmoothingWrapper.fit` method of model.
        For more details see `statsmodels.tsa.holtwinters.SimpleExpSmoothing.fit`

    clip_predictions_lower: float
        Minimal value allowed for predictions - predictions will be clipped to this value.

    clip_predictions_upper: float
        Maximum value allowed for predictions - predictions will be clipped to this value.
    """

    model_cls = SimpleExpSmoothing

    @tsmodel_wrapper_constructor_factory(SimpleExpSmoothing)
    def __init__(
        self,
        name="SimpleSmoothing",
        fit_params=None,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):
        """This constructor will be modified at runtime to accept
        all parameters of the SimpleExpSmoothing class on top of the ones defined here!"""
        pass


class HoltSmoothingWrapper(BaseStatsmodelsForecastingWrapper):
    """Wrapper for `~statsmodels.tsa.holtwinters.Holt` (see other parameters there)

    Parameters
    ----------
    name: str
        Name of the model instance, used also as column name for returned prediction

    fit_params: dict
        Parameters passed to `~hcrystalball.wrappers.HoltSmoothingWrapper.fit` method of model.
        For more details see `statsmodels.tsa.holtwinters.Holt.fit`

    clip_predictions_lower: float
        Minimal value allowed for predictions - predictions will be clipped to this value.

    clip_predictions_upper: float
        Maximum value allowed for predictions - predictions will be clipped to this value.
    """

    model_cls = Holt

    @tsmodel_wrapper_constructor_factory(Holt)
    def __init__(
        self,
        name="HoltSmoothing",
        fit_params=None,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):
        """This constructor will be modified at runtime to accept
        all parameters of the Holt class on top of the ones defined here!"""
        pass


class ThetaWrapper(BaseStatsmodelsForecastingWrapper):
    """Wrapper for `~statsmodels.tsa.forecasting.theta.ThetaModel` (see other parameters there)

    Parameters
    ----------
    name: str
        Name of the model instance, used also as column name for returned prediction

    conf_int : bool
        Whether confidence intervals should be also outputed.

    fit_params: dict
        Parameters passed to `~hcrystalball.wrappers.ThetaWrapper.fit` method of model.
        For more details see `statsmodels.tsa.forecasting.theta.ThetaModel.fit`

    clip_predictions_lower: float
        Minimal value allowed for predictions - predictions will be clipped to this value.

    clip_predictions_upper: float
        Maximum value allowed for predictions - predictions will be clipped to this value.
    """

    model_cls = ThetaModel

    @tsmodel_wrapper_constructor_factory(ThetaModel)
    def __init__(
        self,
        name="ThetaModel",
        conf_int=False,
        fit_params=None,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):
        """This constructor will be modified at runtime to accept
        all parameters of the Holt class on top of the ones defined here!"""
        pass


__all__ = ["ExponentialSmoothingWrapper", "SimpleSmoothingWrapper", "HoltSmoothingWrapper", "ThetaWrapper"]
