import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# TODO adding possibility to infer frequency from the data
class SeasonalityTransformer(BaseEstimator, TransformerMixin):
    """Generate seasonal feature columns using one-hot encoding.

    Parameters
    ----------
    auto : bool
        Automatically generate week_day, monthly, quarterly, yearly, weekly if
        it makes sense given the data frequency
    freq : str
        Frequency of data
    week_day : bool
        Whether to add day name as a feature
    monthly : bool
        Whether to add month as a feature
    quarterly : bool
        Whether to add quarter as a feature
    yearly : bool
        Whether to add year as a feature
    weekly : bool
        Whether to add week number as a feature

    Raises
    ------
    ValueError
        Error is raised if freq is not in ['D', 'W', 'M','Q', 'Y', None]
    ValueError
        Error is raised if freq is not provided when using auto=True
    """

    def __init__(
        self,
        auto=True,
        freq=None,
        week_day=None,
        monthly=None,
        quarterly=None,
        yearly=None,
        weekly=None,
        month_start=False,
        month_end=False,
        quarter_start=False,
        quarter_end=False,
        year_start=False,
        year_end=False,
    ):
        self.auto = auto
        self.freq = freq
        if self.freq is not None and self.freq not in ["D", "W", "M", "Q", "Y"]:
            raise ValueError("`freq` needs to be one of 'D', 'W', 'M', 'Q', 'Y', None")
        if self.auto is True and self.freq is None:
            raise ValueError("`freq` needs to be provided if `auto` is set to True")
        self.week_day = week_day
        self.monthly = monthly
        self.quarterly = quarterly
        self.yearly = yearly
        self.weekly = weekly
        self.month_start = month_start
        self.month_end = month_end
        self.quarter_start = quarter_start
        self.quarter_end = quarter_end
        self.year_start = year_start
        self.year_end = year_end
        self._fit_columns = None

    def get_feature_names(self):
        """Provide handle to get column names for created data

        Returns
        -------
        list :
            Name of the generated feature vectors when the transformer is fitted.
        """
        return self._fit_columns

    def fit(self, X, y):
        """Set fit columns to None

        Parameters
        ----------
        X : pandas.DataFrame
            Ignored.
        y : numpy.ndarray
            Ignored.

        Returns
        -------
        SeasonalityTransformer
            self
        """
        self._fit_columns = None
        return self

    def _ensure_pred_and_train_cols_equals(self, X):
        """Ensure match between fit and transform columns

        Returns Pandas dataframe for inference with the same features as during training
        (i.e. Test data could miss some months...). This method is important as most
        regressors expect the same structure of data for training as for inference

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Data with the same features as train set had
        """
        miss_cols = list(self._fit_columns.difference(X.columns))
        if len(miss_cols) > 0:
            miss_data = pd.DataFrame(
                data=np.zeros((len(X.index), len(miss_cols)), dtype=int),
                columns=miss_cols,
                index=X.index,
            )
            data = X.join(miss_data)

            return data[self._fit_columns]
        else:
            return X[self._fit_columns]

    def transform(self, X):
        """Create seasonal columns from datetime index

        Parameters
        ----------
        X: pandas.DataFrame
            Input features.

        Returns
        -------
        pandas.DataFrame
            Contains the generated feature vector(s)
        """
        date = pd.to_datetime(X.index)

        season_feat = []
        if (self.week_day or (self.auto and self.freq in ["D"])) and self.week_day is not False:
            season_feat.append(pd.get_dummies(date.day_name()))
        if self.weekly or (self.auto and self.freq in ["D", "W"]) and self.weekly is not False:
            season_feat.append(pd.get_dummies(date.week).rename(columns=lambda x: f"{x}_week"))
        if self.monthly or (self.auto and self.freq in ["D", "W", "M"]) and self.monthly is not False:
            season_feat.append(pd.get_dummies(date.month_name()))
        if (
            self.quarterly
            or (self.auto and self.freq in ["D", "W", "M", "Q"])
            and self.quarterly is not False
        ):
            season_feat.append(pd.get_dummies(date.quarter).rename(columns=lambda x: f"{x}_quarter"))
        if self.yearly or (self.auto and self.freq in ["D", "W", "M", "Q", "Y"]) and self.yearly is not False:
            season_feat.append(pd.get_dummies(date.year))

        _X = pd.concat(season_feat, axis=1)

        if self.month_start:
            _X["month_start"] = date.is_month_start
        if self.month_end:
            _X["month_end"] = date.is_month_end
        if self.quarter_start:
            _X["quarter_start"] = date.is_quarter_start
        if self.quarter_end:
            _X["quarter_end"] = date.is_quarter_end
        if self.year_start:
            _X["year_start"] = date.is_year_start
        if self.year_end:
            _X["year_end"] = date.is_year_end

        _X.columns = [f"_{col}" for col in _X.columns]

        if self._fit_columns is not None:
            _X = self._ensure_pred_and_train_cols_equals(_X)
        else:
            self._fit_columns = _X.columns

        _X.index = date

        return pd.merge(X, _X, left_index=True, right_index=True, how="left")
