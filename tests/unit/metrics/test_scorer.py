import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from hcrystalball.metrics import get_scorer
from hcrystalball.metrics import make_ts_scorer
from hcrystalball.metrics._scorer import _TSPredictScorer


class ReprMixin:
    def __repr__(self, N_CHAR_MAX=100000):

        txt = ""
        for key, _ in inspect.signature(self.__class__).parameters.items():
            param_repr = "".join([key, "=", str(getattr(self, key))])
            txt = ", ".join([txt, param_repr])
        if len(txt) > 1:
            txt = txt[2:]

        txt = "".join([self.__class__.__name__, "(", txt, ")"])
        return txt[:N_CHAR_MAX]


class MockModel(ReprMixin):
    def __init__(self, factor, shift=0.0):

        self._X = None
        self._y = None
        self.factor = factor
        self.shift = shift

    def fit(self, X, y):

        self._X = X
        if isinstance(y, pd.DataFrame):
            self._y = y.copy()
        elif isinstance(y, pd.Series):
            self._y = y.to_frame()

    def predict(self, X):

        return self._y * self.factor + self.shift


@pytest.mark.parametrize("greater_is_better", [(True,), (False,)])
@pytest.mark.parametrize(
    "needs_proba, needs_threshold, expected_error",
    [
        (False, False, None),
        (False, True, NotImplementedError),
        (True, False, NotImplementedError),
        (True, True, ValueError),
    ],
)
def test_make_ts_scorer(greater_is_better, needs_proba, needs_threshold, expected_error):

    if expected_error is None:
        scorer = make_ts_scorer(
            mean_absolute_error,
            greater_is_better=greater_is_better,
            needs_proba=needs_proba,
            needs_threshold=needs_threshold,
        )
        isinstance(scorer, _TSPredictScorer)
    else:
        with pytest.raises(expected_error):
            _ = make_ts_scorer(
                mean_absolute_error,
                greater_is_better=greater_is_better,
                needs_proba=needs_proba,
                needs_threshold=needs_threshold,
            )


@pytest.mark.parametrize(
    "estimator_and_name",
    [
        ("ts_exp_smoothing_wrapper",),
        ("ts_stacking_ensemble",),
        ("pipeline_ts_exp_smoothing_wrapper",),
        ("pipeline_ts_stacking_ensemble",),
        ("pipeline_in_pipeline_ts_exp_smoothing_wrapper",),
        ("pipeline_in_pipeline_ts_stacking_ensemble",),
    ],
    indirect=["estimator_and_name"],
)
@pytest.fixture(scope="module")
def model4persistence(request):

    if request.param == "shift":
        estimator = [MockModel(factor=1.0, shift=3.0)]
        estimator_repr = "MockModel(factor=1.0,shift=3.0)"
        estimator_hash = hashlib.md5(estimator_repr.encode("utf-8")).hexdigest()
        estimator_ids = {estimator_hash: estimator_repr}
        estimator_index = {estimator_hash: 0}

    elif request.param == "two_estimators":
        estimator = [MockModel(factor=1.0, shift=0.0), MockModel(factor=1.0, shift=3.0)]
        estimator_repr = [
            "MockModel(factor=1.0,shift=0.0)",
            "MockModel(factor=1.0,shift=3.0)",
        ]
        estimator_hash = [hashlib.md5(r.encode("utf-8")).hexdigest() for r in estimator_repr]
        estimator_ids = dict(zip(estimator_hash, estimator_repr))
        estimator_index = {h: i for i, h in enumerate(estimator_hash)}

    else:
        estimator = [MockModel(factor=1.0, shift=0.0)]
        estimator_repr = "MockModel(factor=1.0,shift=0.0)"
        estimator_hash = hashlib.md5(estimator_repr.encode("utf-8")).hexdigest()
        estimator_ids = {estimator_hash: estimator_repr}
        estimator_index = {estimator_hash: 0}

    return estimator, estimator_ids, estimator_index


@pytest.mark.parametrize(
    "X_y_linear_trend, model4persistence",
    [("freq_D", ""), ("freq_D", "shift"), ("freq_D", "two_models")],
    indirect=["X_y_linear_trend", "model4persistence"],
)
def test_save_model_repr_and_hash(X_y_linear_trend, model4persistence):

    X, y = X_y_linear_trend

    test_scorer = make_ts_scorer(
        mean_absolute_error,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )

    estimators, expected_model_ids, _ = model4persistence
    for iestimator in estimators:
        iestimator.fit(X, y)
        _ = test_scorer(iestimator, X, y)

    assert test_scorer.estimator_ids == expected_model_ids


@pytest.mark.parametrize(
    "function, expected_error",
    [
        ("neg_mean_absolute_error", None),
        (make_ts_scorer(mean_absolute_error), None),
        (mean_absolute_error, ValueError),
    ],
)
def test_get_scorer(function, expected_error):

    if expected_error is None:
        result = get_scorer(function)
        assert all(
            [
                hasattr(result, "_cv_data"),
                hasattr(result, "_estimator_ids"),
                hasattr(result, "_sign"),
                hasattr(result, "_score_func"),
            ]
        )
    else:
        with pytest.raises(expected_error):
            get_scorer(function)


@pytest.fixture
def ts_predict_scorer():
    scorer = _TSPredictScorer(score_func=mean_absolute_error, sign=-1, kwargs={})

    model_a_split_1_preds = pd.DataFrame(
        {
            "y_true": [1, 2],
            "pred_ml": [np.nan, np.nan],  # simulate failed .predict()
            "split": [0, 0],
            "updated": [pd.Timestamp("2021-07-31"), pd.Timestamp("2021-08-31")],
        }
    ).set_index(["updated"])
    # model_a_split_2_preds do not exist - simulate failed .fit()
    model_a_split_3_preds = pd.DataFrame(
        {
            "y_true": [3, 4],
            "pred_ml": [2, 3],
            "split": [1, 1],
            "updated": [pd.Timestamp("2021-09-30"), pd.Timestamp("2021-10-30")],
        }
    ).set_index(["updated"])
    # model_a_split_4_preds do not exist - simulate failed .fit()

    scorer._split_index["pred_ml"] = 2
    scorer._results["pred_ml"].append(model_a_split_1_preds)
    scorer._results["pred_ml"].append(model_a_split_3_preds)

    model_b_split_1_preds = pd.DataFrame(
        {
            "y_true": [0, 1],
            "pred_base": [0, 1],
            "split": [0, 0],
            "updated": [pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-31")],
        },
    ).set_index(["updated"])
    model_b_split_2_preds = pd.DataFrame(
        {
            "y_true": [1, 2],
            "pred_base": [1, 2],
            "split": [1, 1],
            "updated": [pd.Timestamp("2021-07-31"), pd.Timestamp("2021-08-31")],
        }
    ).set_index(["updated"])
    model_b_split_3_preds = pd.DataFrame(
        {
            "y_true": [2, 3],
            "pred_base": [2, 3],
            "split": [2, 2],
            "updated": [pd.Timestamp("2021-08-31"), pd.Timestamp("2021-09-30")],
        },
    ).set_index(["updated"])
    model_b_split_4_preds = pd.DataFrame(
        {
            "y_true": [3, 4],
            "pred_base": [3, 4],
            "split": [3, 3],
            "updated": [pd.Timestamp("2021-09-30"), pd.Timestamp("2021-10-30")],
        }
    ).set_index(["updated"])

    scorer._split_index["pred_base"] = 4
    scorer._results["pred_base"].append(model_b_split_1_preds)
    scorer._results["pred_base"].append(model_b_split_2_preds)
    scorer._results["pred_base"].append(model_b_split_3_preds)
    scorer._results["pred_base"].append(model_b_split_4_preds)

    return scorer


def test_splits_in_short_data(ts_predict_scorer):
    expected_df = pd.DataFrame(
        {
            "split": [0, 0, 1, 1, 2, 2, 3, 3],
            "y_true": [0, 1, 1, 2, 2, 3, 3, 4],
            "pred_ml": [np.nan] * 6 + [2, 3],
            "pred_base": [0, 1, 1, 2, 2, 3, 3, 4],
            "index": [
                pd.Timestamp("2021-06-30"),
                pd.Timestamp("2021-07-31"),
                pd.Timestamp("2021-07-31"),
                pd.Timestamp("2021-08-31"),
                pd.Timestamp("2021-08-31"),
                pd.Timestamp("2021-09-30"),
                pd.Timestamp("2021-09-30"),
                pd.Timestamp("2021-10-30"),
            ],
        }
    ).set_index("index")

    pd.testing.assert_frame_equal(
        ts_predict_scorer.cv_data.astype(float),
        expected_df.astype(float),
    )
