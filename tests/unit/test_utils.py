import pytest
import sys
from importlib import import_module

import pandas as pd
import numpy as np
import pandas._testing as tm
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from numpy.testing import assert_array_equal

from hcrystalball.utils import check_X_y
from hcrystalball.utils import check_fit_before_predict
from hcrystalball.utils import get_estimator_repr
from hcrystalball.utils import optional_import
from hcrystalball.utils import deep_dict_update
from hcrystalball.exceptions import InsufficientDataLengthError
from hcrystalball.exceptions import PredictWithoutFitError


@pytest.fixture(scope="module")
def X(request):

    if "series" in request.param:
        return tm.makeTimeSeries(freq="D")
    elif "dataframe" in request.param:
        result = tm.makeTimeDataFrame(freq="D").drop(columns="A")
        if "date_col_str" in request.param:
            return result.assign(index=lambda x: x.index.astype(str)).set_index("index")
        elif "len_<_3" in request.param:
            return result.iloc[:2, :]
        elif "wo_date_col" in request.param:
            result.index.name = "some_other_index_name"
            return result
        else:
            raise ValueError("Invalid X fixture parameter")
    else:
        raise ValueError("Invalid X fixture parameter")


@pytest.fixture(scope="module")
def y(request):

    if request.param is None:
        return None
    elif "dataframe" in request.param:
        return tm.makeTimeDataFrame(freq="D")
    elif "series" in request.param:
        result = tm.makeTimeSeries(freq="D")
        if "wrong_len" in request.param:
            return result[:2]
        elif "ok" in request.param:
            return result
        else:
            raise ValueError("Invalid X fixture parameter")
    elif "ndarray" in request.param:
        result = tm.makeTimeSeries(freq="D").values
        if "wrong_len" in request.param:
            return result[:2]
        elif "wrong_ndim" in request.param:
            return np.array([result, result])
        elif "ok" in request.param:
            return result
        else:
            raise ValueError("Invalid X fixture parameter")
    else:
        raise ValueError("Invalid X fixture parameter")


@pytest.mark.parametrize(
    "X, y, expected_error",
    [
        ("series", None, TypeError),
        ("dataframe_len_<_3", None, InsufficientDataLengthError),
        ("dataframe_date_col_str", None, ValueError),
        ("dataframe_wo_date_col", None, None),
        ("dataframe_wo_date_col", "dataframe", TypeError),
        ("dataframe_wo_date_col", "series_wrong_len", ValueError),
        ("dataframe_wo_date_col", "series_ok", None),
        ("dataframe_wo_date_col", "ndarray_wrong_len", ValueError),
        ("dataframe_wo_date_col", "ndarray_wrong_ndim", ValueError),
        ("dataframe_wo_date_col", "ndarray_ok", None),
    ],
    indirect=["X", "y"],
)
def test_check_X_y(X, y, expected_error):
    @check_X_y
    def pass_func(self, X, y):
        return X, y

    # make sure certain checks raises appropriate errors
    if expected_error is not None:
        with pytest.raises(expected_error):
            pass_func(None, X, y)
    else:
        # make sure X and y stay unchanged after the check
        res_X, res_y = pass_func(None, X, y)

        assert_frame_equal(res_X, X)
        if isinstance(y, pd.Series):
            assert_series_equal(res_y, y)
        else:
            assert_array_equal(res_y, y)


@pytest.mark.parametrize("model_is_fitted, expected_error", [(True, None), (False, PredictWithoutFitError)])
def test_check_fit_before_predict(model_is_fitted, expected_error):
    class DummyModel:
        def __init__(self, name="dummy_model", fitted=False):

            self.name = name
            self.fitted = fitted

        @check_fit_before_predict
        def predict(self, X):

            return X

    x = 3
    dummy_model = DummyModel(fitted=model_is_fitted)

    if expected_error is None:
        assert x == dummy_model.predict(x)

    else:
        with pytest.raises(PredictWithoutFitError):
            _ = dummy_model.predict(x)


@pytest.mark.parametrize(
    "wrapper_instance",
    ["sklearn", "stacking_ensemble", "simple_ensemble", "smoothing", "sarimax", "prophet", "tbats"],
    indirect=["wrapper_instance"],
)
def test_get_model_repr_single_model(wrapper_instance):

    model_repr = get_estimator_repr(wrapper_instance)
    assert model_repr.find("...") == -1
    assert model_repr == wrapper_instance.__repr__(N_CHAR_MAX=10000).replace("\n", "").replace(" ", "")


@pytest.mark.parametrize(
    "pipeline_instance_model_in_pipeline",
    ["sklearn", "stacking_ensemble", "simple_ensemble", "smoothing", "sarimax", "prophet", "tbats"],
    indirect=["pipeline_instance_model_in_pipeline"],
)
def test_get_model_repr_pipeline_instance_model_in_pipeline(
    pipeline_instance_model_in_pipeline,
):

    model_repr = get_estimator_repr(pipeline_instance_model_in_pipeline)
    assert model_repr.find("...") == -1
    assert model_repr == pipeline_instance_model_in_pipeline.__repr__(N_CHAR_MAX=10000).replace(
        "\n", ""
    ).replace(" ", "")


@pytest.mark.parametrize(
    "module_name, class_name, dependency",
    [
        ("_prophet", "ProphetWrapper", "fbprophet"),
        ("_statsmodels", "ExponentialSmoothingWrapper", "statsmodels.tsa.api"),
        ("_statsmodels", "HoltSmoothingWrapper", "statsmodels.tsa.api"),
        ("_statsmodels", "SimpleSmoothingWrapper", "statsmodels.tsa.api"),
        ("_sarimax", "SarimaxWrapper", "pmdarima.arima"),
        ("_tbats", "TBATSWrapper", "tbats"),
        ("_tbats", "BATSWrapper", "tbats"),
    ],
)
def test_optional_import_missing_dependency(module_name, class_name, dependency):
    # store modules for later reference
    hcb_wrappers = sys.modules["hcrystalball.wrappers"]
    hcb_wrappers_module = sys.modules[f"hcrystalball.wrappers.{module_name}"]
    dep = sys.modules[dependency]

    # remove modules
    del sys.modules["hcrystalball.wrappers"]
    del sys.modules[f"hcrystalball.wrappers.{module_name}"]
    sys.modules[dependency] = None

    # import never fails
    res = optional_import(f"hcrystalball.wrappers.{module_name}", class_name, globals())

    assert not res

    # class has an informative docstring
    assert (
        globals()[class_name].__doc__
        == "This is just helper class to inform user about missing dependencies at init time"
    )

    # init of the helper class fails
    with pytest.raises(ModuleNotFoundError):
        globals()[class_name]()

    # restore modules for other tests
    sys.modules["hcrystalball.wrappers"] = hcb_wrappers
    sys.modules[f"hcrystalball.wrappers.{module_name}"] = hcb_wrappers_module
    sys.modules[dependency] = dep


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("_prophet", "ProphetWrapper"),
        ("_statsmodels", "ExponentialSmoothingWrapper"),
        ("_statsmodels", "HoltSmoothingWrapper"),
        ("_statsmodels", "SimpleSmoothingWrapper"),
        ("_sarimax", "SarimaxWrapper"),
        ("_tbats", "TBATSWrapper"),
        ("_tbats", "BATSWrapper"),
    ],
)
def test_optional_import_with_dependency(module_name, class_name):
    # importing from private modules (from hcrystalball.wrappers._prophet import ProphetWrapper)
    # and wrapper module (form hcrystalball.wrappers import ProphetWrapper)
    # returns the same result when having dependencies in place
    ProphetWrapperOrig = getattr(import_module(f"hcrystalball.wrappers.{module_name}"), class_name)

    res = optional_import(f"hcrystalball.wrappers.{module_name}", class_name, globals())

    assert res[0] == class_name
    assert str(globals()[class_name]()) == str(ProphetWrapperOrig())


@pytest.mark.parametrize(
    "source, update, exp_result",
    [
        (
            {"a": {"b": 1, "c": 2}, "x": {"z": 1}},
            {"x": {"c": 0}},
            {"a": {"b": 1, "c": 2}, "x": {"z": 1, "c": 0}},
        ),
        (
            {"x": {"c": 1}},
            {"x": {"c": 0}},
            {"x": {"c": 0}},
        ),
    ],
)
def test_deep_dict_update(source, update, exp_result):
    result = deep_dict_update(source, update)
    assert result == exp_result
    assert source == source
    assert update == update
