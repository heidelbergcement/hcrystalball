import pytest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

from hcrystalball.wrappers import ExponentialSmoothingWrapper
from hcrystalball.compose import TSColumnTransformer
from hcrystalball.feature_extraction import HolidayTransformer


@pytest.mark.parametrize(
    "X_y_linear_trend, wrapper_instance,horizon",
    [
        ("ndarray_freq_D", "sklearn", 10),
        ("ndarray_freq_D", "prophet", 10),
        ("ndarray_freq_D", "smoothing", 10),
        ("ndarray_freq_D", "tbats", 10),
        ("ndarray_freq_D", "stacking_ensemble", 10),
        ("ndarray_freq_D", "sarimax", 10),
    ],
    indirect=["X_y_linear_trend", "wrapper_instance"],
)
def test_model_fit_predict(X_y_linear_trend, wrapper_instance, horizon):
    X, y = X_y_linear_trend
    assert (not hasattr(wrapper_instance, "fitted")) or (wrapper_instance.fitted is False)

    wrapper_instance.fit(X[:-horizon], y[:-horizon])
    assert wrapper_instance.fitted is True
    result = wrapper_instance.predict(X[-horizon:]).astype(float)
    expected_result = pd.DataFrame(y, index=X.index, columns=[wrapper_instance.name])
    # if isinstance(wrapper_instance.cap_predictions_lower, float):
    #     expected_result = expected_result.clip_lower(0.)
    assert_frame_equal(expected_result[-horizon:], result, check_names=False)


@pytest.fixture(scope="module")
def pipeline(request):
    if "passthrough_position" in request.param:
        return TSColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), ["one_hot"]),
                ("raw_cols_1", "passthrough", ["trend"]),
            ]
        )
    if "col_name_clash" in request.param:
        return TSColumnTransformer(
            transformers=[("raw_cols_1", "passthrough", ["trend"]), ("scaler", StandardScaler(), ["trend"])]
        )
    if "more_dimensions_with_get_feature_names" in request.param:
        return TSColumnTransformer(
            transformers=[("raw_cols_1", "passthrough", ["trend"]), ("scaler", OneHotEncoder(), ["one_hot"])]
        )
    if "less_dimensions_without_get_feature_names" in request.param:
        return TSColumnTransformer(
            transformers=[
                ("raw_cols_1", "passthrough", ["trend"]),
                ("pca", PCA(n_components=1), ["one_hot", "trend"]),
            ]
        )
    if "with_model" in request.param:
        return Pipeline(
            [
                (
                    "preproc",
                    TSColumnTransformer(
                        transformers=[
                            ("raw_cols_1", "passthrough", ["trend"]),
                            ("scaler", StandardScaler(), ["trend", "one_hot"]),
                        ]
                    ),
                ),
                ("model", ExponentialSmoothingWrapper(trend="add")),
            ]
        )
    if "more_layers_builtin_transformers" in request.param:
        return Pipeline(
            [
                (
                    "first",
                    TSColumnTransformer(
                        transformers=[
                            ("raw_cols_1", "passthrough", ["trend"]),
                            ("one_hot", OneHotEncoder(sparse=False), ["one_hot"]),
                            ("scaler", StandardScaler(), ["trend"]),
                        ]
                    ),
                ),
                (
                    "second",
                    TSColumnTransformer(
                        transformers=[
                            ("raw_cols_2", "passthrough", ["trend"]),
                            ("one_hot", StandardScaler(), ["x0_1"]),
                        ]
                    ),
                ),
            ]
        )
    if "more_layers_custom_transformers_same_level_country_code_country_col" in request.param:
        return Pipeline(
            [
                (
                    "first",
                    TSColumnTransformer(
                        transformers=[
                            ("raw_cols_1", "passthrough", ["trend", "country"]),
                            ("scaler", StandardScaler(), ["trend", "one_hot"]),
                        ]
                    ),
                ),
                ("holiday", HolidayTransformer(country_code_column="country")),
                (
                    "second",
                    TSColumnTransformer(
                        transformers=[
                            ("one_hot", OneHotEncoder(sparse=False), ["_holiday_country"]),
                            ("raw_cols_2", "passthrough", ["trend"]),
                        ]
                    ),
                ),
            ]
        )

    if "more_layers_custom_transformers_same_level_country_code" in request.param:
        return Pipeline(
            [
                (
                    "first",
                    TSColumnTransformer(
                        transformers=[
                            ("raw_cols_1", "passthrough", ["trend"]),
                            ("scaler", StandardScaler(), ["trend", "one_hot"]),
                        ]
                    ),
                ),
                ("holiday", HolidayTransformer(country_code="DE")),
                (
                    "second",
                    TSColumnTransformer(
                        transformers=[
                            ("one_hot", OneHotEncoder(sparse=False), ["_holiday_DE"]),
                            ("raw_cols_2", "passthrough", ["trend"]),
                        ]
                    ),
                ),
            ]
        )

    if "more_layers_holiday_in_column_transformer" in request.param:
        return Pipeline(
            [
                (
                    "first",
                    TSColumnTransformer(
                        transformers=[
                            ("raw_cols_1", "passthrough", ["trend", "country"]),
                            ("holiday", HolidayTransformer(country_code_column="country"), ["country"]),
                            ("scaler", StandardScaler(), ["trend", "one_hot"]),
                        ]
                    ),
                ),
                (
                    "second",
                    TSColumnTransformer(
                        transformers=[
                            ("one_hot", OneHotEncoder(sparse=False), ["_holiday_country"]),
                            ("raw_cols_2", "passthrough", ["trend", "country"]),
                        ]
                    ),
                ),
            ]
        )


@pytest.mark.parametrize(
    "X_y_linear_trend, pipeline, exp_cols",
    [
        ("more_cols_freq_D", "passthrough_position", ["one_hot", "trend"]),
        ("more_cols_freq_D", "col_name_clash", ["trend", "trend_scaler"]),
        (
            "more_cols_freq_D",
            "more_dimensions_with_get_feature_names",
            ["trend", "x0_1", "x0_2", "x0_3", "x0_4"],
        ),
        (
            "more_cols_freq_D",
            "less_dimensions_without_get_feature_names",
            ["trend", "pca_0"],
        ),
        ("more_cols_freq_D", "with_model", ["ExponentialSmoothing"]),
        ("more_cols_freq_D", "more_layers_builtin_transformers", ["trend", "x0_1"]),
        (
            "more_cols_freq_D",
            "more_layers_custom_transformers_same_level_country_code",
            ["x0_", "x0_New year", "trend"],
        ),
        (
            "more_cols_country_col_freq_D",
            "more_layers_custom_transformers_same_level_country_code_country_col",
            ["x0_", "x0_New year", "trend"],
        ),
        (
            "more_cols_country_col_freq_D",
            "more_layers_holiday_in_column_transformer",
            ["x0_", "x0_New year", "trend", "country"],
        ),
    ],
    indirect=["X_y_linear_trend", "pipeline"],
)
def test_ts_column_transformer_fit_transform(X_y_linear_trend, pipeline, exp_cols):
    X, y = X_y_linear_trend

    if isinstance(pipeline, Pipeline) and hasattr(pipeline.steps[-1][1], "predict"):
        res = pipeline.fit(X, y).predict(X)
        assert list(res.columns) == list(exp_cols)
    else:
        res = pipeline.fit_transform(X, y)
        assert list(res.columns) == list(exp_cols)
        assert res.shape == (len(X), len(exp_cols))

        # check passthrough cols keeps unchanged data
        pass_cols = []
        # wrap col transformer to pipeline to easier iterate over it
        if isinstance(pipeline, TSColumnTransformer):
            pipeline = Pipeline([("transformer", pipeline)])

        last_step = pipeline.steps[-1][1]
        if hasattr(last_step, "_iter"):
            for _, trans, apply_cols, _ in last_step._iter():
                if trans == "passthrough":
                    pass_cols.extend(apply_cols)

        [assert_series_equal(res[col], X[col], check_dtype=False) for col in pass_cols]
