import pytest
import types
import numpy as np
from hcrystalball.model_selection import FinerTimeSplit


@pytest.mark.parametrize(
    "ts_data, expected_error",
    [
        ("series", None),
        ("series_with_NaN", None),
        ("series_with_Inf", None),
        ("series_with_name", None),
        ("series_with_index_name", None),
        ("dataframe", None),
        ("dataframe_with_name", None),
        ("dataframe_with_index_name", None),
        ("dataframe_multicolumn", None),
        ("dataframe_integer_index", None),
        ("empty_dataframe", ValueError),
        ("empty_series", ValueError),
    ],
    indirect=["ts_data"],
)
def test_cv_finertimesplit_split_pandas_container_data(ts_data, expected_error):

    n_splits = 2
    horizon = 3
    fts = FinerTimeSplit(n_splits=n_splits, horizon=horizon)
    if expected_error is None:
        result = fts.split(ts_data)
        assert isinstance(result, types.GeneratorType)
        result = list(result)
        assert len(result) == n_splits
        for i, isplit in enumerate(result):
            assert len(isplit) == 2
            assert len(isplit[0]) == len(ts_data) - (n_splits - i) * horizon
            assert len(isplit[1]) == horizon
            assert np.array_equal(isplit[0], np.arange(len(ts_data) - (n_splits - i) * horizon))
            assert np.array_equal(isplit[1], np.arange(horizon) + len(ts_data) - (n_splits - i) * horizon)
    else:
        with pytest.raises(expected_error):
            _ = list(fts.split(ts_data))


@pytest.mark.parametrize(
    "test_data, expected_error",
    [(np.arange(6), None), ([0, 1, 2, 3, 4, 5], None), ((0, 1, 2, 3, 4, 5), None), (13, TypeError)],
)
def test_cv_finertimesplit_split_input_data_types(test_data, expected_error):

    n_splits = 2
    horizon = 3
    fts = FinerTimeSplit(n_splits=n_splits, horizon=horizon)
    if expected_error is None:
        result = list(fts.split(test_data))
        assert len(result) == n_splits
        for i, isplit in enumerate(result):
            assert len(isplit) == 2
            assert len(isplit[0]) == len(test_data) - (n_splits - i) * horizon
            assert len(isplit[1]) == horizon
            assert np.array_equal(isplit[0], np.arange(len(test_data) - (n_splits - i) * horizon))
            assert np.array_equal(
                isplit[1],
                np.arange(horizon) + len(test_data) - (n_splits - i) * horizon,
            )
    else:
        with pytest.raises(expected_error):
            _ = list(fts.split(test_data))
