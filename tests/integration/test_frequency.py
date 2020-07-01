import pytest
import pandas as pd


@pytest.mark.parametrize(
    "X_y_with_freq, freq",
    [
        ("series_with_freq_D", "D"),
        ("series_with_freq_M", "M"),
        ("series_with_freq_Q", "Q-DEC"),
        ("series_with_freq_Y", "A-DEC"),
    ],
    indirect=["X_y_with_freq"],
)
@pytest.mark.parametrize(
    "wrapper_instance",
    [
        ("sklearn"),
        ("stacking_ensemble"),
        ("simple_ensemble"),
        ("smoothing"),
        ("sarimax"),
        ("prophet"),
        ("tbats"),
    ],
    indirect=["wrapper_instance"],
)
def test_model_frequencies(X_y_with_freq, freq, wrapper_instance):
    X, y = X_y_with_freq

    predicted_index = wrapper_instance.fit(X[:-10], y[:-10]).predict(X[-10:]).index
    assert pd.infer_freq(predicted_index) == freq
    assert len(predicted_index) == len(X[-10:])
