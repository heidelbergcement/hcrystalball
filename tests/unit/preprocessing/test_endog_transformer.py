import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from hcrystalball.compose import TSColumnTransformer
from hcrystalball.wrappers import get_sklearn_wrapper
from hcrystalball.preprocessing import TargetTransformer
from hcrystalball.utils import generate_tsdata


def test_target_transformer():
    X, y = generate_tsdata(n_dates=365 * 2)
    X["trend"] = np.arange(len(X))

    preprocessing = TSColumnTransformer(transformers=[("scaler", StandardScaler(), ["trend"])])
    # define random forest model
    rf_model = get_sklearn_wrapper(RandomForestRegressor)
    # glue it together
    sklearn_model_pipeline = Pipeline([("preprocessing", preprocessing), ("model", rf_model)])

    scaled_pipeline = TargetTransformer(sklearn_model_pipeline, StandardScaler())

    preds = scaled_pipeline.fit(X[:-10], y[:-10]).predict(X[-10:])

    assert hasattr(scaled_pipeline, "named_steps")
    assert isinstance(scaled_pipeline.y_transformer, StandardScaler)
    assert isinstance(preds, pd.DataFrame)
