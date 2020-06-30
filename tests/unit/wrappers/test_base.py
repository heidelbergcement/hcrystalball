import pytest


@pytest.mark.parametrize(
    "X_y_linear_trend, wrapper_instance_capped",
    [
        ("freq_D", "sklearn;93;99"),
        ("freq_D", "stacking_ensemble;93;99"),
        ("freq_D", "simple_ensemble;93;99"),
        ("freq_D", "smoothing;93;99"),
        ("freq_D", "sarimax;93;99"),
        ("freq_D", "prophet;93;99"),
        ("freq_D", "tbats;93;99"),
    ],
    indirect=["X_y_linear_trend", "wrapper_instance_capped"],
)
def test_clip_predictions(X_y_linear_trend, wrapper_instance_capped):

    X, y = X_y_linear_trend

    LOWER = 93.0
    UPPER = 99.0
    y_pred = wrapper_instance_capped.fit(X[:-10], y[:-10]).predict(X[-10:])
    assert y_pred[wrapper_instance_capped.name].between(LOWER, UPPER).all()
