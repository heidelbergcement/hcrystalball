import pytest
from sklearn.linear_model import ElasticNet
from hcrystalball.wrappers import TBATSWrapper
from hcrystalball.wrappers import ExponentialSmoothingWrapper
from hcrystalball.wrappers import ProphetWrapper


@pytest.mark.parametrize(
    "X_y_linear_trend, wrapper_instance,param, param_value, model_param",
    [
        ("series_with_freq_D", "sklearn", "optimize_for_horizon", True, False),
        ("series_with_freq_D", "sklearn", "fit_intercept", False, True),
        ("series_with_freq_D", "prophet", "extra_holidays", {"Whit Monday": {"lower_window": 2}}, False),
        ("series_with_freq_D", "prophet", "holidays_prior_scale", 20, True),
        ("series_with_freq_D", "smoothing", "fit_params", {"smoothing_level": 0.2}, False),
        ("series_with_freq_D", "smoothing", "trend", "add", True),
        ("series_with_freq_D", "tbats", "conf_int", True, False),
        ("series_with_freq_D", "tbats", "use_box_cox", True, True),
        ("series_with_freq_D", "stacking_ensemble", "meta_model", ElasticNet(), False),
        ("series_with_freq_D", "sarimax", "conf_int", True, False),
        ("series_with_freq_D", "sarimax", "order", (1, 1, 0), True),
    ],
    indirect=["X_y_linear_trend", "wrapper_instance"],
)
def test_set_params(X_y_linear_trend, wrapper_instance, param, param_value, model_param):
    X, y = X_y_linear_trend
    wrapper_instance.set_params(**{param: param_value})
    assert wrapper_instance.get_params()[param] == param_value
    if model_param:
        wrapper_instance.fit(X, y)
        if isinstance(wrapper_instance, ExponentialSmoothingWrapper):
            assert wrapper_instance.model.model.__getattribute__(param) == param_value
        elif isinstance(wrapper_instance, ProphetWrapper):
            assert wrapper_instance.model.__getattribute__(param) == param_value
        elif isinstance(wrapper_instance, TBATSWrapper):
            assert vars(wrapper_instance.model.params.components)[param] == param_value
        else:
            assert wrapper_instance.model.get_params()[param] == param_value
