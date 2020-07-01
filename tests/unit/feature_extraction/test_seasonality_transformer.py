import pandas as pd
import numpy as np
import pytest
from hcrystalball.feature_extraction import SeasonalityTransformer
import calendar


weekdays = list(calendar.day_name)
months = list(calendar.month_name)[1:]


@pytest.mark.parametrize(
    "X_start, X_len, weekdays, weeks, months, quarter, year",
    [
        (
            "2019-01-01",
            366,
            weekdays,
            [f"{i}_week" for i in range(1, 53)],
            months,
            [f"{i}_quarter" for i in range(1, 5)],
            [2019, 2020],
        ),
        (
            "2019-01-01",
            365,
            weekdays,
            [f"{i}_week" for i in range(1, 53)],
            months,
            [f"{i}_quarter" for i in range(1, 5)],
            [2019],
        ),
        (
            "2019-01-03",
            5,
            set(weekdays).difference(["Tuesday", "Wednesday"]),
            ["1_week", "2_week"],
            ["January"],
            ["1_quarter"],
            [2019],
        ),
    ],
)
def test_seasonality_transformer(X_start, X_len, weekdays, weeks, months, quarter, year):

    X = pd.DataFrame(index=pd.date_range(X_start, periods=X_len, freq="D"))
    y = pd.Series(np.arange(len(X)), name="values", index=X.index)

    freq = pd.infer_freq(X.index)

    df = pd.concat([X, y, SeasonalityTransformer(freq=freq).fit(X, y).transform(X)], axis=1)

    assert set(weekdays).issubset(df.columns)
    assert set(weeks).issubset(df.columns)
    assert set(months).issubset(df.columns)
    assert set(quarter).issubset(df.columns)
    assert set(year).issubset(df.columns)

    first_row = df.head(1).T
    cols_with_ones = first_row[first_row[first_row.columns[0]] == 1].index

    single_date_cols = (
        SeasonalityTransformer(freq=freq).fit(X.head(1), y.head(1)).transform(X.head(1)).columns
    )

    assert set(cols_with_ones) == set(single_date_cols)


def test_seasonality_transformer_ensure_cols():

    X = pd.DataFrame(index=pd.date_range("2019-01-01", periods=10, freq="D"))
    y = pd.Series(np.arange(len(X)), name="values", index=X.index)

    head_t = SeasonalityTransformer(freq="D").fit(X.head(6), y.head(6))
    head_1 = head_t.transform(X.head(6))
    # names of columns in transformer are accessible over get_feature_names after first transform
    assert set(head_1.columns) == set(head_t.get_feature_names())

    head_2 = head_t.transform(X.tail(6))

    assert set(head_1.columns) == set(head_2.columns)

    tail_t = SeasonalityTransformer(freq="D").fit(X.tail(6), y.tail(6))
    tail_1 = tail_t.transform(X.tail(6))

    # missing column was brought back from first transform
    assert ("Friday" not in tail_1.columns) & ("Friday" in head_2.columns)
    # extra columns from 2nd and other transforms are clipped away
    assert ("2_week" in tail_1.columns) & ("2_week" not in head_2.columns)
