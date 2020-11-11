import pandas as pd
import pytest
from hcrystalball.wrappers import ExponentialSmoothingWrapper
from hcrystalball.wrappers import SimpleSmoothingWrapper
from hcrystalball.wrappers import HoltSmoothingWrapper
from hcrystalball.wrappers import ThetaWrapper


@pytest.mark.parametrize("X_y_optional", [("just_X"), ("")], indirect=["X_y_optional"])
@pytest.mark.parametrize(
    "model",
    [ExponentialSmoothingWrapper, HoltSmoothingWrapper, SimpleSmoothingWrapper, ThetaWrapper],
)
def test_statsmodels_transform_data_to_tsmodel_input_format(X_y_optional, model):

    X, y = X_y_optional
    endog = model()._transform_data_to_tsmodel_input_format(X, y)
    if y is not None:
        assert isinstance(endog, pd.Series)
        assert endog.shape[0] == y.shape[0]
    else:
        assert endog == X.shape[0]


@pytest.mark.parametrize("X_y_optional", [("")], indirect=["X_y_optional"])
@pytest.mark.parametrize("conf_int", [True, False])
@pytest.mark.parametrize(
    "model",
    [ExponentialSmoothingWrapper, HoltSmoothingWrapper, SimpleSmoothingWrapper, ThetaWrapper],
)
def test_statsmodels_predict_with_conf_int(X_y_optional, model, conf_int):

    X, y = X_y_optional
    if conf_int and model.__name__ == "ThetaWrapper":
        result = model(name="test", conf_int=conf_int).fit(X[:-10], y[:-10]).predict(X[-10:])
        assert isinstance(result, pd.DataFrame)
        assert all(result.columns == ["test_lower", "test_upper", "test"])

    elif not conf_int:
        result = model(name="test").fit(X[:-10], y[:-10]).predict(X[-10:])
        assert isinstance(result, pd.DataFrame)
        assert all(result.columns == ["test"])

    else:
        with pytest.raises(TypeError):
            result = model(conf_int=conf_int).fit(X[:-10], y[:-10]).predict(X[-10:])
