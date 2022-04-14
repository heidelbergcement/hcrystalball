import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import SCORERS
from sklearn.metrics._scorer import _BaseScorer

from hcrystalball.utils import generate_estimator_hash
from hcrystalball.utils import get_estimator_name
from hcrystalball.utils import get_estimator_repr

logger = logging.getLogger(__name__)


class PersistCVDataMixin:
    def _save_prediction(self, y_pred, estimator_label, y_true):
        """Persist the prediction in cross validation

        Parameters
        ----------
        y_pred: `pandas.DataFrame`
            Predictions. A DataFrame container with a single column and datetime index

        estimator_label: str
            Label of the estimator used to identify the model with a given parameter set in the presisted
            data

        y_true: `pandas.DataFrame`
            True values. A DataFrame container with a single column and the same datetime index as
            y_pred. If not set, the 'y_true' column will be omitted in the data persistence without
            raising any warning or exception.

        Returns
        -------
        None
        """
        self._results[estimator_label].append(
            y_pred.set_axis([estimator_label], axis=1)
            .join(y_true.rename("y_true"))
            .assign(split=self._split_index[estimator_label])
        )
        self._split_index[estimator_label] += 1

    def _upsert_estimator_hash(self, estimator_repr, estimator_hash):
        if estimator_hash not in self._estimator_ids:
            self._estimator_ids[estimator_hash] = estimator_repr


class _TSPredictScorer(_BaseScorer, PersistCVDataMixin):
    def __call__(self, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        return self._score(None, estimator, X, y_true, sample_weight)

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        try:
            y_pred = estimator.predict(X)
        except Exception as e:
            logger.error(
                f"Prediction for estimator {get_estimator_name(estimator)} is set to `np.nan` "
                f"due to its inability to predict on X={X}\n{e}"
            )
            y_pred = pd.DataFrame({get_estimator_name(estimator).split("__")[-1]: np.nan}, index=X.index)

        estimator_repr = get_estimator_repr(estimator)
        estimator_hash = generate_estimator_hash(estimator)
        self._upsert_estimator_hash(estimator_repr, estimator_hash)
        self._save_prediction(
            y_pred=y_pred,
            estimator_label=estimator_hash,
            y_true=y_true if isinstance(y_true, pd.Series) else pd.Series(y_true, index=X.index),
        )

        if y_pred.isna().any().any() or np.isinf(y_pred).any().any():
            return np.nan
        else:
            if sample_weight is not None:
                return self._sign * self._score_func(
                    y_true, y_pred, sample_weight=sample_weight, **self._kwargs
                )
            else:
                return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    def __init__(self, score_func, sign, kwargs):
        """Enhances inherited init with cv data, split index and estimator ids

        Parameters
        ----------
        score_func : callable
            Scoring function
        sign : int
            Whether to minimize or maximize scoring function
        kwargs : dict
            Additional arguments to be passed to inherited init
        """
        super().__init__(score_func, sign, kwargs)

        self._cv_data = pd.DataFrame(columns=["split"])
        self._estimator_ids = dict()
        self._split_index = defaultdict(int)
        self._results = defaultdict(list)

    @property
    def estimator_ids(self):
        return self._estimator_ids

    @property
    def cv_data(self):
        if not self._cv_data.empty:
            return self._cv_data
        elif self._results:
            self._results_to_cv_data()
            return self._cv_data
        else:
            return None

    def _results_to_cv_data(self):
        """
        Transform list like results into the dataframe with expected format.

        This function takes the results of the estimator, that has most predictions (most splits)
        as a blueprint mapping between dates in the predictinos and cv splits.
        This information is later used to correctly label splits for the predictions in other estimators.
        """
        # ensure, that models not successfully fitting on all splits
        # get correctly assigned split numbers
        #
        # 1. take label of the estimator, that has the highest number of splits
        # (how many times predictions were made)
        max_estimator_label = max(self._split_index, key=self._split_index.get)
        # 2. create dictionary lookup for index -> split number
        index_split_lookup = {
            str(result.index): result["split"].unique()[0] for result in self._results[max_estimator_label]
        }
        # 3. for each predictions for each estimator assign split number
        # based on the index for which the predictions were made
        merged = pd.concat(
            [
                split_preds.assign(split=index_split_lookup[str(split_preds.index)])
                for _, estimator_preds in self._results.items()
                for split_preds in estimator_preds
            ]
        )
        # 4. ensure unique combinations of split and index for each estimator while keeping NaNs
        # for failing splits (each estimator will have (exactly one) value for each split)
        self._cv_data = (
            merged.rename_axis([None])
            .reset_index()
            .groupby(["split", "index"])
            .max()
            .reset_index(level=0)
            .astype({"split": pd.Int64Dtype()})
        )


def get_scorer(function="neg_mean_absolute_error"):
    """Get a scorer supporting storing data for gridsearch from string.

    Parameters
    ----------
    function : callable or str
        callable your own function

    Returns
    -------
    sklearn compatible scorer
        Scorer with data and estimator ids storage
    """
    if isinstance(function, str):
        scorer = SCORERS[function]
        greater_is_better = True if scorer._sign == 1 else False
        return make_ts_scorer(scorer._score_func, greater_is_better)
    elif hasattr(function, "_cv_data") and hasattr(function, "_estimator_ids"):
        return function
    else:
        raise ValueError(
            f"Provided scoring function must be instance of `_TSPredictScorer`"
            f"(use make_ts_scorer) or one of {SCORERS.keys()}"
        )


def make_ts_scorer(
    score_func,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.

    This factory function wraps scoring functions for use in `~sklearn.model_selection.GridSearchCV`
    and `~sklearn.model_selection.cross_validate`. It takes a score function, such as ``accuracy_score``,
    ``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
    and returns a callable that scores an estimator's output.
    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    greater_is_better : boolean
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.

    needs_proba : boolean
        Not yet implemented, kept only to be compatible with the scikit-learn API

    needs_threshold : boolean
        Not yet implemented, kept only to be compatible with the scikit-learn API

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    callable
        scorer object that returns a scalar score

    """

    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True," " but not both.")
    if needs_proba:
        raise NotImplementedError("Usage/evaluation of prediction probabilities are not yet implemented.")
    elif needs_threshold:
        raise NotImplementedError("Evaluation of decision function output is not yet implemented.")
    else:
        cls = _TSPredictScorer

    return cls(score_func, sign, kwargs)
