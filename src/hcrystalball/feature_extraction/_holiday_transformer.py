import pandas as pd
from workalendar.registry import registry
from sklearn.base import BaseEstimator, TransformerMixin


class HolidayTransformer(TransformerMixin, BaseEstimator):
    """Generate holiday feature based on provided ISO code

    Parameters
    ----------
    country_code: str
        ISO code if country/region

    country_code_column: str
        name of the column which have the ISO code of the country/region

    days_before: int
        number of days before the holiday which will be taken into account
        (i.e. 2 means that new bool column will be created and will be True for 2 days before holidays,
        otherwise False)

    days_after: int
        number of days after the holiday which will be taken into account
        (i.e. 2 means that new bool column will be created and will be True for 2 days after holidays,
        otherwise False)

    bridge_days: bool
        overlaping `days_before` and `days_after` feature which serves for modeling between
        holidays working days

    Please be aware that you cannot provide both country_code and country_code_column
    during initialization since this would be ambuguious. If you provide `country_code_column`
    instead of `country_code` the ISO code found in the column will be assigned into `country_code` column.
    """

    def __init__(
        self,
        country_code=None,
        country_code_column=None,
        days_before=0,
        days_after=0,
        bridge_days=False,
    ):
        self.country_code = country_code
        self.unified_country_code = country_code
        self.country_code_column = country_code_column
        self.days_before = days_before
        self.days_after = days_after
        self.bridge_days = bridge_days

        if self.country_code is None and self.country_code_column is None:
            raise ValueError("You need to provide `country_code` or `country_code_column`")
        if self.country_code and self.country_code_column:
            raise ValueError("Provide `country_code` or `country_code_column`, not both")
        self._col_name = f"_holiday_{self.country_code or self.country_code_column}"

    @property
    def unified_country_code(self):
        """Utility storing country code or unique value from country_code_column"""
        return self._unified_country_code

    def get_feature_names(self):
        """Return list with features which the transformer generates"""
        return [self._col_name]

    @unified_country_code.setter
    def unified_country_code(self, value):
        if value is not None and value not in list(registry.region_registry.keys()):
            raise ValueError("Unknown `country_code`. For list of valid codes please look at workalendar.")
        self._unified_country_code = value

    def fit(self, X, y=None):
        """Check if `date_col` has daily frequency

        This check is in `fit` method since pandas.infer_freq is used which requires at least 3 observations.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : Any
            Ignored

        Returns
        -------
        HolidayTransformer
            self

        Raises
        ------
        ValueError
            in case daily frequency is not used or very few datapoints are provided in X
        """
        if pd.infer_freq(X.index) != "D":
            raise ValueError(
                f"HolidayTransformer can be used only with daily frequency in index. "
                f"Your index is of type {type(X.index)} with frequency {pd.infer_freq(X.index)}"
            )

        return self

    def transform(self, X, y=None):
        """Create data with 'holiday' colummn

        Columns contains names of the holidays based on provided 'date' column

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : numpy.ndarray
            iIgnored.

        Returns
        -------
        pandas.DataFrame
            DataFrame with `self._col_name` column including names of holidays for each of the date

        Raises
        ------
        KeyError
            if 'country_code_column' is not found in X
        ValueError
            if country_code_column has more than 1 value in X
        """

        if self.country_code_column:
            if self.country_code_column not in X.columns:
                raise KeyError(
                    f"Column {self.country_code_column} provided as "
                    f"`country_code_column` is not in dataframe!"
                )
            if X[self.country_code_column].nunique() != 1:
                raise ValueError(
                    f"There needs to be only one unique value in entire `country_code_column` column. "
                    f"These values were found {X[self.country_code_column].unique()}!"
                )
            self.unified_country_code = X[self.country_code_column].unique()[0]

        years = X.index.year.unique().tolist() + [max(X.index.year)]
        cal = registry.region_registry[self.unified_country_code]()
        holidays = (
            pd.concat(
                [pd.DataFrame(data=cal.holidays(year), columns=["date", self._col_name]) for year in years]
            )
            # one day could have multiple public holidays
            .drop_duplicates(subset="date").set_index("date")
        )

        df = (
            pd.merge(X, holidays, left_index=True, right_index=True, how="left")
            .fillna({self._col_name: ""})
            .drop(columns=[self.country_code_column], errors="ignore")
        )

        df = self._get_day_around_holiday_feature(
            df, f"_{self.days_before}_before{self._col_name}", -self.days_before
        )
        df = self._get_day_around_holiday_feature(
            df, f"_{self.days_after}_after{self._col_name}", self.days_after
        )
        if self.bridge_days:
            if self.days_before == 0 or self.days_before == 0:
                raise ValueError(
                    """`bridge_days` feature is created only if `days_before`
                                and `days_before` are both greater than 0 """
                )
            else:
                df = df.assign(
                    **{
                        f"_bridge{self._col_name}": lambda df: (
                            df[
                                [
                                    f"_{self.days_before}_before{self._col_name}",
                                    f"_{self.days_after}_after{self._col_name}",
                                ]
                            ].all(axis=1)
                        )
                    }
                )

        return df

    def _get_day_around_holiday_feature(self, df, col_name, days):
        """
        Add new boolean column into pandas.DataFrame with number of `days` being True
        before or after the public holiday depending on `when` parameter.

        Parameters
        ----------
        df : pandas.DataFrame
            column with `self._col_name` and datetime index
        col_name : str
            column name of new feature
        days : int
            number of days taken into account (with 0 doesn't create any column)

        Returns
        -------
        pandas.DataFrame
            DataFrame with `self._col_name` column and additional column `col_name` if `days` > 0
        """
        for day in range(1, abs(days) + 1):
            day = day if days > 0 else -day
            df = df.assign(**{f"_{col_name}_{day}": lambda df: ((df[self._col_name] != "").shift(day))})
        cols = df.filter(like=f"_{col_name}_").columns
        if days != 0:
            # all intermediate columns called (i.e. _{`col_name`}_) are combined into one as `col_name`
            df = df.assign(**{f"{col_name}": lambda df: df[cols].any(axis=1)})

        return df.drop(cols, axis=1)
