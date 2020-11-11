import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pandas.testing import assert_frame_equal

from hcrystalball.ensemble import StackingEnsemble, SimpleEnsemble
from hcrystalball.exceptions import DuplicatedModelNameError


@pytest.fixture(
    scope="module",
    params=["with_duplicates", "no_duplicates", "no_duplicates_with_pipeline"],
)
def base_learners(request):
    class DummyModel:
        def __init__(self, alpha, name):

            self.alpha = alpha
            self.name = name
            self.fitted = False

        def fit(self, X, y):

            self.fitted = True

        def predict(self, X):

            return pd.DataFrame(np.ones(len(X)) * self.alpha, columns=["dummy"], index=X.index)

    if request.param == "with_duplicates":
        return [DummyModel(name="model", alpha=5), DummyModel(name="model", alpha=20)]
    elif request.param == "no_duplicates":
        return [
            DummyModel(name="model_1", alpha=5),
            DummyModel(name="model_2", alpha=20),
        ]
    elif request.param == "no_duplicates_with_pipeline":
        return [
            Pipeline([("model", DummyModel(name="model_1", alpha=5))]),
            DummyModel(name="model_2", alpha=20),
        ]
    elif request.param == "with_duplicates_with_pipeline":
        return [
            Pipeline([("model", DummyModel(name="model_1", alpha=5))]),
            DummyModel(name="model__model_1", alpha=20),
        ]
    else:
        return None


@pytest.mark.parametrize(
    "base_learners, ensemble, kwargs, expected_error",
    [
        ("no_duplicates", StackingEnsemble, {"meta_model": LinearRegression()}, None),
        ("no_duplicates", SimpleEnsemble, {}, None),
        (
            "with_duplicates",
            StackingEnsemble,
            {"meta_model": LinearRegression()},
            DuplicatedModelNameError,
        ),
        ("with_duplicates", SimpleEnsemble, {}, DuplicatedModelNameError),
        (
            "no_duplicates_with_pipeline",
            StackingEnsemble,
            {"meta_model": LinearRegression()},
            None,
        ),
        ("no_duplicates_with_pipeline", SimpleEnsemble, {}, None),
        (
            "with_duplicates_with_pipeline",
            StackingEnsemble,
            {"meta_model": LinearRegression()},
            DuplicatedModelNameError,
        ),
        ("with_duplicates_with_pipeline", SimpleEnsemble, {}, DuplicatedModelNameError),
    ],
    indirect=["base_learners"],
)
def test_check_base_learners_names(base_learners, ensemble, kwargs, expected_error):

    if expected_error is None:
        se = ensemble(base_learners=base_learners, **kwargs)
        assert isinstance(se, ensemble)

    else:
        with pytest.raises(expected_error):
            _ = ensemble(base_learners=base_learners, **kwargs)


@pytest.mark.parametrize(
    "base_learners, ensemble_func, expected_error",
    [
        ("no_duplicates", "mean", None),
        ("no_duplicates", "min", None),
        ("no_duplicates", "max", None),
        ("no_duplicates", "median", None),
        ("no_duplicates", "agg", ValueError),  # pandas available func
        ("no_duplicates", "random_string", ValueError),  # no real func
    ],
    indirect=["base_learners"],
)
def test_ensemble_func(base_learners, ensemble_func, expected_error):

    if expected_error is not None:
        with pytest.raises(expected_error):
            _ = SimpleEnsemble(base_learners=base_learners, ensemble_func=ensemble_func)
    else:
        model = SimpleEnsemble(base_learners=base_learners, ensemble_func=ensemble_func)
        alphas = [bl.alpha for bl in model.base_learners]
        X = pd.DataFrame(index=pd.date_range("2012", "2016", freq="Y"))
        model.fit(X, y=np.ones(len(X)))
        exp_result = pd.DataFrame(
            (
                pd.DataFrame(np.ones(len(X)) * alphas[0])
                .assign(xx=np.ones(len(X)) * alphas[1])
                .apply(ensemble_func, axis=1)
                .values
            ),
            columns=[model.name],
            index=X.index,
        )
        assert_frame_equal(exp_result, model.predict(X))


@pytest.mark.parametrize("base_learners", [("no_duplicates")], indirect=["base_learners"])
def test_ensembles_stackingensemble_create_horizons_as_features(base_learners):

    n_splits = 2
    horizon = 3

    model = StackingEnsemble(
        meta_model=LinearRegression(),
        base_learners=base_learners,
        train_n_splits=n_splits,
        train_horizon=horizon,
    )

    cross_result_index = np.arange(horizon * n_splits, dtype=int)
    df = model._create_horizons_as_features(cross_result_index, horizon=horizon, n_splits=n_splits)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (n_splits * horizon, horizon)


@pytest.mark.parametrize("base_learners", [("no_duplicates")], indirect=["base_learners"])
def test_ensembles_stackingensemble_create_weekdays_as_features(base_learners):

    n_splits = 2
    horizon = 3

    model = StackingEnsemble(
        meta_model=LinearRegression(),
        base_learners=base_learners,
        train_n_splits=n_splits,
        train_horizon=horizon,
    )

    cross_result_index = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
    )
    df = model._create_weekdays_as_features(cross_result_index)
    result = pd.DataFrame(
        {
            "Friday": [0, 0, 1, 0, 0],
            "Saturday": [0, 0, 0, 1, 0],
            "Sunday": [0, 0, 0, 0, 1],
            "Thursday": [0, 1, 0, 0, 0],
            "Wednesday": [1, 0, 0, 0, 0],
        },
        index=cross_result_index,
    ).astype("uint8")

    assert_frame_equal(result, df)
