import pandas as pd
import pytest
from hcrystalball.wrappers import TBATSWrapper
from tbats import TBATS
from pandas.testing import assert_frame_equal


@pytest.mark.parametrize("X_y_linear_trend", [("freq_D")], indirect=["X_y_linear_trend"])
def test_conf_int(X_y_linear_trend):

    HORIZON = 5
    X, y = X_y_linear_trend

    model = TBATS(use_arma_errors=False, use_box_cox=False)
    model_wrapped = TBATSWrapper(use_arma_errors=False, use_box_cox=False, conf_int=True, conf_int_level=0.95)
    model = model.fit(y[:-HORIZON])
    model_wrapped = model_wrapped.fit(X[:-HORIZON], y[:-HORIZON])

    preds_orig, conf_int = model.forecast(steps=HORIZON, confidence_level=0.95)
    preds = model_wrapped.predict(X[-HORIZON:])

    expected_result = (
        pd.DataFrame(preds_orig, index=X.index[-HORIZON:], columns=["TBATS"])
        .assign(TBATS_lower=conf_int["lower_bound"])
        .assign(TBATS_upper=conf_int["upper_bound"])
    )
    print("expected_result", expected_result)

    print("preds", preds)
    print("preds_orig", preds_orig)

    assert_frame_equal(preds, expected_result)
