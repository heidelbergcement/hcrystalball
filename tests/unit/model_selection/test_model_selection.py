import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from hcrystalball.metrics import get_scorer
from hcrystalball.model_selection import FinerTimeSplit
from hcrystalball.model_selection import get_best_not_failing_model
from hcrystalball.model_selection import select_model
from hcrystalball.wrappers import ExponentialSmoothingWrapper
from hcrystalball.wrappers import get_sklearn_wrapper


@pytest.mark.parametrize(
    "train_data, grid_search, parallel_over_dict",
    [("two_regions", "", {"Region": "region_0"}), ("two_regions", "", None)],
    indirect=["train_data", "grid_search"],
)
def test_select_model(train_data, grid_search, parallel_over_dict):

    _train_data = train_data

    if parallel_over_dict:
        col, value = list(parallel_over_dict.items())[0]
        _train_data = train_data[train_data[col] == value].drop(columns="Region")

    partition_columns = ["Region", "Product"]

    results = select_model(
        _train_data,
        target_col_name="Quantity",
        partition_columns=partition_columns,
        parallel_over_dict=parallel_over_dict,
        grid_search=grid_search,
        country_code_column="Holidays_code",
    )
    if parallel_over_dict:
        partitions = (
            train_data.loc[train_data[col] == value, partition_columns]
            .drop_duplicates()
            .to_dict(orient="records")
        )
    else:
        partitions = train_data[partition_columns].drop_duplicates().to_dict(orient="records")

    assert len(results) == len(partitions)

    for result in results:
        assert result.best_model_name == "good_dummy"
        assert result.partition in partitions


@pytest.mark.parametrize(
    "X_y_optional, negative_data, best_model_name, rank, expected_error",
    [
        ("", False, "ExponentialSmoothingWrapper", 1, None),
        ("", True, "SklearnWrapper", 2, None),
        ("", True, "", 2, ValueError),
    ],
    indirect=["X_y_optional"],
)
def test_get_best_not_failing_model(X_y_optional, negative_data, best_model_name, rank, expected_error):
    X, y = X_y_optional
    # data contains 0
    y[y < 1] = 1
    if negative_data:
        y[-1] = -1
    models = [
        ExponentialSmoothingWrapper(freq="D", trend="mul"),
        get_sklearn_wrapper(DummyRegressor, strategy="constant", constant=-5000),
    ]
    models = models if expected_error is None else models[:1]
    grid_search = GridSearchCV(
        estimator=Pipeline([("model", "passthrough")]),
        param_grid=[{"model": models}],
        scoring=get_scorer("neg_mean_absolute_error"),
        cv=FinerTimeSplit(n_splits=1, horizon=5),
        refit=False,
        error_score=np.nan,
    )

    grid_search.fit(X, y)

    if expected_error:
        with pytest.raises(expected_error):
            get_best_not_failing_model(grid_search, X, y)
    else:
        best_param_rank = get_best_not_failing_model(grid_search, X, y)
        assert isinstance(best_param_rank, dict)
        assert best_param_rank["params"]["model"].__class__.__name__ == best_model_name
        assert best_param_rank["rank"] == rank
