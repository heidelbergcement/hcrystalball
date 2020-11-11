import pytest
from sklearn.preprocessing import StandardScaler
from hcrystalball.compose import TSColumnTransformer
from sklearn.preprocessing import OneHotEncoder


@pytest.fixture
def column_transformer_and_cols(request):
    if "with_duplicated_name" in request.param:
        tran = TSColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), ["trend"]),
                ("raw_cols", "passthrough", ["trend", "one_hot"]),
            ]
        )
        cols = ["trend_scaler", "trend", "one_hot"]

    elif "with_transformer_creating_many_cols" in request.param:
        tran = TSColumnTransformer(
            transformers=[
                ("raw_cols", "passthrough", ["trend", "one_hot"]),
                (
                    "one_hot",
                    OneHotEncoder(),
                    ["one_hot"],
                ),
            ]
        )
        cols = ["trend", "one_hot", "x0_1", "x0_2", "x0_3", "x0_4"]

    elif "passthrough_columns_in_the_middle" in request.param:
        tran = TSColumnTransformer(
            transformers=[
                ("one_hot", OneHotEncoder(), ["one_hot"]),
                ("raw_cols", "passthrough", ["one_hot"]),
                ("scaler", StandardScaler(), ["trend"]),
            ]
        )
        cols = ["x0_1", "x0_2", "x0_3", "x0_4", "one_hot", "trend"]

    return tran, cols


@pytest.mark.parametrize(
    "column_transformer_and_cols, X_y_linear_trend",
    [
        ("with_duplicated_name", "more_cols_freq_D"),
        ("with_transformer_creating_many_cols", "more_cols_freq_D"),
        ("passthrough_columns_in_the_middle", "more_cols_freq_D"),
    ],
    indirect=["column_transformer_and_cols", "X_y_linear_trend"],
)
def test_ts_column_transformer(column_transformer_and_cols, X_y_linear_trend):

    X, y = X_y_linear_trend
    transformer, cols = column_transformer_and_cols
    result = transformer.fit(X, y).transform(X)
    print(result.columns.tolist())
    print(cols)
    assert result.columns.tolist() == cols
