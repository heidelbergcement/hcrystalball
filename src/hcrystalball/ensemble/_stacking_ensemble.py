import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from hcrystalball.model_selection import FinerTimeSplit
from hcrystalball.utils import check_X_y
from hcrystalball.utils import enforce_y_type
from hcrystalball.utils import check_fit_before_predict
from hcrystalball.utils import get_estimator_name
from hcrystalball.exceptions import DuplicatedModelNameError


class StackingEnsemble(BaseEstimator):
    """StackingEnsemble model, which takes a list of any hcrystalball model
    wrapper instance(s) as base learners.

    During fitting the base learners are fitted and prediction(s) will be made for
    the requested horizon, using possibly more than one splits. The predictions for
    each model in all splits are concatenated and serve as the feature matrix for the
    meta model, with the prediction of each model over all splits being a distinct feature.
    Finally the meta model, which is just a regular regressor, will then be fitted to the
    data to determine the relative weights of each base learner in the prediction of the ensemble.

    As a default behaviour the meta model is fitted only the first time the fit()
    method is called, then in each subsequent calls of the fit() method (of a given
    StackingEnsemble instance) omits the fitting of the meta model and fits only the base
    learners. This behaviour can, however be changed using the fit_meta_model_always
    parameter to force the meta model to be refitted every time the fit method is called.
    Note, however, that this latter behaviour can be computationally expensive, as fitting
    the meta model requires fitting the base learners train_n_splits times.

    Parameters
    ----------
    name: str
        Unique name / identifier of the model instance

    base_learners: list
        List of fully instantiated hcrystalball model wrappers

    meta_model: sklearn.base.BaseEstimator
        Scikit-learn compatible regressor

    train_n_splits: int
        Number of splits used for fitting the meta model

    train_horizon: int
        Max. number of steps ahead to be predicted. Ideally this value should not be identical
        to the forecasting horizon in prediction.

    horizons_as_features: bool
        Adds horizon feature for meta model

    weekdays_as_features: bool
        Adds weekdays feature for meta model

    fit_meta_model_always: bool
        If True the meta model will always be re-fitted, each time  the fit() method is called,
        if False the meta model will only be fitted the first time the fit() method is
        called and in subsequent calls of the fit() method only the base learners will be re-fitted.
    """

    def __init__(
        self,
        base_learners,
        meta_model,
        name="stacking_ensemble",
        train_n_splits=1,
        train_horizon=10,
        horizons_as_features=True,
        weekdays_as_features=True,
        fit_meta_model_always=False,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
    ):

        self._check_base_learners_names(base_learners)
        self.name = name
        self.base_learners = base_learners
        self.meta_model = meta_model
        self.train_n_splits = train_n_splits
        self.train_horizon = train_horizon
        self.fit_meta_model_always = fit_meta_model_always
        self.horizons_as_features = horizons_as_features
        self.weekdays_as_features = weekdays_as_features
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

        Returns
        -------
        None

        Raises
        ------
        DuplicatedModelNameError
            If multiple models have the same `name` attribute.
        """

        names = [get_estimator_name(model) for model in models]
        if len(names) != len(set(names)):
            raise DuplicatedModelNameError(
                "There seems to be duplicates in model names among StackingEnsemble base learners. "
                "Model names should be unique."
            )

    def _fit_base_learners(self, X, y=None):
        """Fit the base learners

        Parameters
        ----------
        X: pandas.DataFrame
            Input features.

        y: numpy.ndarray
            Target vector.s

        Returns
        -------
        None
        """

        for model in self.base_learners:
            model.fit(X, y)

    def _predict_features_for_meta_models(self, X):
        """Provide predictions from all base learners

        Parameters
        ----------
        X: pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Container with the X['date'] as index and the names of the base learners
            as column names. Each column should contain the prediction of a base learner
            with a name found in the column name.
        """

        prediction = pd.DataFrame(
            index=X.index,
            columns=[get_estimator_name(model) for model in self.base_learners],
        )

        for model in self.base_learners:
            model_name = get_estimator_name(model)
            prediction.loc[:, model_name] = model.predict(X).values.squeeze()

        return prediction

    @staticmethod
    def _create_horizons_as_features(cross_results_index, horizon, n_splits):
        """DataFrame with dummy columns describing the horizon variable.

        Dummy column is created for each horizon(i.e. horizon 5 == 5 new columns). Column itself
        will be 1 only for it's particular horizon, for the rest will be 0.
        This method is intended for use when 'variable_horizon' is set to True.

        Returns
        -------
        pandas.DataFrame
        A DataFrame container with dummy column for each value in horizon (i.e. horizon 5 == 5 new columns).
        These features should help meta models to model properly weight meta predictions based on horizon
        """

        return pd.get_dummies(pd.Series(list(np.arange(horizon)) * n_splits)).set_index(cross_results_index)

    @staticmethod
    def _create_weekdays_as_features(cross_results_index):
        """DataFrame with dummy columns for each week_day based on provided `cross_results_index`

        Returns
        -------
        pandas.DataFrame
        """
        return pd.get_dummies(pd.to_datetime(cross_results_index).day_name()).set_index(cross_results_index)

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
        StackingEnsemble
            A fitted StackingEnsemble instance
        """
        self._check_base_learners_names(self.base_learners)

        # Fit the base learners and the meta_model
        if (not self.fitted) or self.fit_meta_model_always:
            splitter = FinerTimeSplit(horizon=self.train_horizon, n_splits=self.train_n_splits)

            n_train_meta = self.train_n_splits * self.train_horizon
            X_meta = pd.DataFrame(
                index=X.index[-n_train_meta:],
                columns=[get_estimator_name(bl) for bl in self.base_learners],
            )
            y_meta = y[-n_train_meta:]
            # Get base learners predictions
            for ind_train, ind_pred in splitter.split(X):
                X_train = X.iloc[ind_train, :]
                X_pred = X.iloc[ind_pred, :]
                y_train = y[ind_train]

                self._fit_base_learners(X_train, y_train)
                X_meta.loc[X_pred.index, :] = self._predict_features_for_meta_models(X_pred)
            # Add dummy horizon variable for meta model
            if self.horizons_as_features:
                X_meta = pd.concat(
                    [
                        X_meta,
                        self._create_horizons_as_features(
                            cross_results_index=X_meta.index,
                            horizon=self.train_horizon,
                            n_splits=self.train_n_splits,
                        ),
                    ],
                    axis=1,
                )
            if self.weekdays_as_features:
                X_meta = pd.concat(
                    [X_meta, self._create_weekdays_as_features(cross_results_index=X_meta.index)],
                    axis=1,
                )

            self._fit_columns = X_meta.columns
            self.meta_model.fit(X_meta.values, y_meta)

        # Fit the base learners on the whole training set
        self._fit_base_learners(X, y)
        self.fitted = True

        return self

    def _ensure_pred_and_train_cols_equals(self, X):
        """Returns Pandas dataframe for inference with the same features as during training

        (i.e. Test data could miss some months...). This method is important as most regressors
        expect the same structure of data for training as for inference

        Parameters
        ----------
        data: pandas.DataFrame
            Input features.

        Returns
        -------
        data
            pandas.DataFrame with the same features as train set had
        """
        miss_cols = list(self._fit_columns.difference(X.columns))
        if len(miss_cols) > 0:
            miss_data = pd.DataFrame(
                data=np.zeros((len(X.index), len(miss_cols))),
                columns=miss_cols,
                index=X.index,
            )
            data = X.join(miss_data)

            return data[self._fit_columns]
        else:
            return X[self._fit_columns]

    @check_fit_before_predict
    def predict(self, X):
        """Calculate the prediction of the ensemble for a given set of date / time

        Parameters
        ----------
        X: pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            A DataFrame container with the index being the input (date)time vector.
            The single column in the DataFrame contains the prediction and the column
            name is the name of the model (i.e. the `name` parameter passed to the constructor)
        """

        X_meta = self._predict_features_for_meta_models(X)
        y_pred = pd.DataFrame(index=X.index, columns=[self.name])
        if self.horizons_as_features:
            X_meta = pd.concat(
                [
                    X_meta,
                    self._create_horizons_as_features(
                        cross_results_index=X_meta.index,
                        horizon=len(X_meta),
                        n_splits=1,
                    ),
                ],
                axis=1,
            )
        if self.weekdays_as_features:
            X_meta = pd.concat(
                [X_meta, self._create_weekdays_as_features(cross_results_index=X_meta.index)],
                axis=1,
            )
        X_meta = self._ensure_pred_and_train_cols_equals(X_meta)
        y_pred[self.name] = self.meta_model.predict(X_meta.values)
        y_pred[self.name] = y_pred[self.name].clip(
            lower=self.clip_predictions_lower, upper=self.clip_predictions_upper
        )

        return y_pred
