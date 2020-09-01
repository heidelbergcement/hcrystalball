from abc import ABCMeta, abstractmethod
import pandas as pd
from tbats import BATS
from tbats import TBATS
from hcrystalball.wrappers._base import TSModelWrapper
from hcrystalball.wrappers._base import tsmodel_wrapper_constructor_factory
from hcrystalball.utils import check_X_y, enforce_y_type, check_fit_before_predict


class BaseTBATSWrapper(TSModelWrapper, metaclass=ABCMeta):
    """Base Wrapper for models from tbats package

    See more at https://github.com/intive-DataScience/tbats

    Currently supported ones are TBATS, BATS
    """

    @abstractmethod
    def __init__(self):
        pass

    @enforce_y_type
    @check_X_y
    def fit(self, X, y):
        """Fit the model.

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
        self.model = self._init_tsmodel(self.model_cls)
        self.model = self.model.fit(y)
        self.fitted = True
        return self

    @check_fit_before_predict
    def predict(self, X):
        """Transform data to tbats required format and run the predictions.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Prediction stored in column with name being the `name` of the wrapper.
        """
        if self.conf_int:
            preds, conf_ints = self.model.forecast(steps=X.shape[0], confidence_level=self.conf_int_level)

            preds = pd.DataFrame(preds, index=X.index, columns=[self.name])
            preds[f"{self.name}_lower"] = conf_ints["lower_bound"]
            preds[f"{self.name}_upper"] = conf_ints["upper_bound"]
        else:
            preds = pd.DataFrame(
                self.model.forecast(steps=X.shape[0]),
                index=X.index,
                columns=[self.name],
            )
        return self._clip_predictions(preds)


class BATSWrapper(BaseTBATSWrapper):
    """Wrapper for BATS model

    https://github.com/intive-DataScience/tbats

    Brings BATS to sklearn time-series compatible interface and puts fit parameters
    to initialization stage.

    Parameters
    ----------
    name : str
        Name of the model instance, used also as column name for returned prediction.

    fit_params : dict
        Parameters passed to `fit` BATS model.

    conf_int : bool
        Whether confidence intervals should be also outputed.

    conf_int_level : float
        Confidence level of returned confidence interval

    clip_predictions_lower : float
        Minimal value allowed for predictions - predictions will be clipped to this value.

    clip_predictions_upper : float
        Maximum value allowed for predictions - predictions will be clipped to this value.

    Notes
    -----
    Fitting the model might take significant time. You might consider advices from the author
    https://medium.com/p/cf3e4e80cf48/responses/show

    """

    model_cls = BATS

    @tsmodel_wrapper_constructor_factory(BATS)
    def __init__(
        self,
        name="BATS",
        fit_params=None,
        conf_int=False,
        conf_int_level=0.95,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):
        """This constructor will be modified at runtime to accept
        all parameters of the BATS class on top of the ones defined here!"""
        pass


class TBATSWrapper(BaseTBATSWrapper):
    """Wrapper for TBATS model

    https://github.com/intive-DataScience/tbats

    Brings TBATS to sklearn time-series compatible interface and puts fit parameters
    to initialization stage.

    Parameters
    ----------
    name : str
        Name of the model instance, used also as column name for returned prediction.

    fit_params : dict
        Parameters passed to `fit` TBATS model.

    conf_int : bool
        Whether confidence intervals should be also outputed.

    conf_int_level : float
        Confidence level of returned confidence interval

    clip_predictions_lower : float
        Minimal value allowed for predictions - predictions will be clipped to this value.

    clip_predictions_upper : float
        Maximum value allowed for predictions - predictions will be clipped to this value.

    Notes
    -----
    Fitting the model might take significant time. You might consider advices from the author
    https://medium.com/p/cf3e4e80cf48/responses/show

    """

    model_cls = TBATS

    @tsmodel_wrapper_constructor_factory(TBATS)
    def __init__(
        self,
        name="TBATS",
        fit_params=None,
        conf_int=False,
        conf_int_level=0.95,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):
        """This constructor will be modified at runtime to accept
        all parameters of the TBATS class on top of the ones defined here!"""
        pass


__all__ = ["TBATSWrapper", "BATSWrapper"]
