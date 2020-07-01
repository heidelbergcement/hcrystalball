import pandas as pd
from sklearn.base import BaseEstimator

from hcrystalball.utils import check_X_y
from hcrystalball.utils import enforce_y_type
from hcrystalball.utils import check_fit_before_predict
from hcrystalball.utils import get_estimator_name
from hcrystalball.exceptions import DuplicatedModelNameError


class SimpleEnsemble(BaseEstimator):
    """SimpleEnsemble model, which takes a list of any hcrystalball model
    wrapper instance(s) as base learners and aggregates their prediction
    using `ensemble_func`.

    See motivation to average forecasts from different models
    https://otexts.com/fpp2/combinations.html

    Parameters
    ----------
    name: str
        Unique name / identifier of the model instance

    base_learners: list
        List of fully instantiated hcrystalball model wrappers

    ensemble_func: {'mean', 'median', 'min', 'max'}
        Function to aggregate `base_learners` predictions
    """

    def __init__(
        self,
        base_learners,
        ensemble_func="mean",
        name="simple_ensemble",
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):

        self._check_base_learners_names(base_learners)
        self.base_learners = base_learners
        self.name = name
        if ensemble_func not in ("mean", "median", "min", "max"):
            raise ValueError(
                "Invalid ensemble_func passed. Valid choices are: 'mean', 'median', 'min', 'max' "
            )
        self.ensemble_func = ensemble_func
        self.fitted = False
        self.clip_predictions_lower = clip_predictions_lower
        self.clip_predictions_upper = clip_predictions_upper

    @staticmethod
    def _check_base_learners_names(models):
        """Check if the base learner models have all unique names

        Parameters
        ----------
        models: list
            List of instatiated hcrystalball model wrapper instances

        Raises
        ------
        DuplicatedModelNameError
            If multiple models have the same `name` attribute.
        """

        names = [get_estimator_name(model) for model in models]
        if len(names) != len(set(names)):
            raise DuplicatedModelNameError(
                "There seems to be duplicates in model names among SimpleEnsemble base learners."
                "Model names should be unique."
            )

    @enforce_y_type
    @check_X_y
    def fit(self, X, y=None):
        """Fit the stacking ensemble model

        Parameters
        ----------
        X: pandas.DataFrame
            Input features.

        y: numpy.ndarray
            Target vector.

        Returns
        -------
        SimpleEnsemble
            A fitted SimpleEnsemble instance
        """
        self._check_base_learners_names(self.base_learners)

        for model in self.base_learners:
            model.fit(X, y)

        self.fitted = True
        return self

    @check_fit_before_predict
    def predict(self, X):
        """Calculate the prediction of the ensemble for a given set of date / time

        Parameters
        ----------
        X: pandas.DataFrame
            DataFrame container with a single column, named 'date',
            containing the datetimes for which the predictions should be made.

        Returns
        -------
        pandas.DataFrame
            A DataFrame container with the index being the input (date)time vector.
            The single column in the DataFrame contains the prediction and the column
            name is the name of the model (i.e. the `name` parameter passed to the constructor)
        """
        y_pred = pd.DataFrame(index=X.index, columns=[self.name])

        for model in self.base_learners:
            model_name = get_estimator_name(model)
            y_pred[model_name] = model.predict(X)
        y_pred[self.name] = y_pred.drop(columns=[self.name]).apply(self.ensemble_func, axis=1)
        y_pred[self.name] = y_pred[self.name].clip(
            lower=self.clip_predictions_lower, upper=self.clip_predictions_upper
        )
        return y_pred[[self.name]]
