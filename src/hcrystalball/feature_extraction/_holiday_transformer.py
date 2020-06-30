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

    Please be aware that you cannot provide both country_code and country_code_column
    during initialization since this would be ambuguious. If you provide `country_code_column`
    instead of `country_code` the ISO code found in the column will be assigned into `country_code` column.
    """

    def __init__(self, country_code=None, country_code_column=None):
        self.country_code = country_code
        self.unified_country_code = country_code
        self.country_code_column = country_code_column
        if self.country_code is None and self.country_code_column is None:
            raise ValueError("You need to provide `country_code` or `country_code_column`")
        if self.country_code and self.country_code_column:
            raise ValueError("Provide `country_code` or `country_code_column`, not both")

    @property
    def unified_country_code(self):
        """Utility storing country code or unique value from country_code_column"""
        return self._unified_country_code

    def get_feature_names(self):
        """Return list with features which the transformer generates - `holiday`"""
        return ["holiday"]

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
            DataFrame with 'holiday' column including names of holidays for each of the date

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
            pd.concat([pd.DataFrame(data=cal.holidays(year), columns=["date", "holiday"]) for year in years])
            .assign(date=lambda x: x["date"].astype(str))
            # one day could have multiple public holidays
            .drop_duplicates(subset="date")
            .set_index("date")
        )

        result = (
            pd.merge(X, holidays, left_index=True, right_index=True, how="left")
            .fillna({"holiday": ""})
            .drop(columns=[self.country_code_column], errors="ignore")
        )

        return result
