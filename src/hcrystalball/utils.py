import hashlib
import functools
import collections
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from hcrystalball.exceptions import InsufficientDataLengthError
from hcrystalball.exceptions import PredictWithoutFitError


def deep_dict_update(source, overrides):
    """
    Update a nested dictionary.
    """
    if overrides is None:
        overrides = {}
    if source is None:
        source = {}

    result = source.copy()

    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_dict_update(result.get(key, {}), value)
            result[key] = returned
        else:
            result[key] = overrides[key]
    return result


def optional_import(module_path, class_name, caller_namespace):
    """Imports optional dependencies.

    Imports class from module if importable,
    otherwise creates dummy class with the same name as requested class.
    Dummy class fails on init with ModuleNotFound error for the missing
    dependencies.

    Parameters
    ----------
    module_name : str
        Path to the module
    class_name : str
        Name of the class to import/mock
    caller_namespace : dict, str
        Namespace of the caller

    Returns
    -------
    list
        If importable [class_name] to extend __all__ of the calling module
        otherwise empty list
    """
    dunder_all_extend = []
    try:
        exec(f"from {module_path} import {class_name}", {}, caller_namespace)
        dunder_all_extend.append(class_name)
    except Exception:
        exec(
            f"class {class_name}:\n"
            f"    'This is just helper class to inform user about missing dependencies at init time'\n"
            f"    def __init__(self, **kwargs):\n"
            f"    # init always fails\n"
            f"        from {module_path} import {class_name}\n",
            {},
            caller_namespace,
        )
    return dunder_all_extend


def get_estimator_repr(model, n_char_max=10000):
    """Get the string representation of a model

    Parameters
    ----------
    model : hcrystalball.base.TSModelWrapper, sklearn.base.BaseEstimator
        instance of model

    n_char_max : int
        max. number of characters to be used for the string

    Returns
    -------
    str:
        String representation of the model
    """
    return model.__repr__(N_CHAR_MAX=n_char_max).replace("\n", "").replace(" ", "")


def generate_estimator_hash(model):
    """Generate a unique hash for a model using its string representation

    Parameters
    ----------
    model: hcrystalball.base.TSModelWrapper, sklearn.base.BaseEstimator
        instance of model

    Returns
    -------
    str:
        String containing the MD5 hash generated from the string representation of the model
    """
    model_str = get_estimator_repr(model)
    model_label_hash = hashlib.md5(model_str.encode("utf-8")).hexdigest()

    return model_label_hash


def generate_partition_hash(partition_label):
    """Generate a unique hash for data partition

    Parameters
    ----------
    partition_label : dict
        label in form {"column_1":"value_a", "column_2":"value_b"}

    Returns
    -------
    str
        String containing the MD5 hash generated from the string representation of partition label
    """
    h = hashlib.md5()
    sorted_keys = sorted(partition_label, key=str.lower)
    [h.update((str(key) + str(partition_label[key])).encode("utf-8")) for key in sorted_keys]
    return h.hexdigest()


def check_X_y(func):
    """Check that X and y have expected shapes and types"""

    @functools.wraps(func)
    def _check_X_y(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("`X` must be a pandas dataframe.")

        if len(X) < 3:
            raise InsufficientDataLengthError(
                "`X` must have at least 3 observations. " "Please provide valid data."
            )

        if not pd.api.types.is_datetime64_dtype(X.index):
            raise ValueError(f"`X` must contain index of type datetime. Your index is {X.index}")

        if y is not None:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise TypeError(
                    f"`y` must be either pandas series or numpy ndarray. " f"You provided `{type(y)}`"
                )

            if len(X) != len(y):
                raise ValueError(
                    f"`X` and `y` must have same length. " f"`len(X)={len(X)}` and `len(y)={len(y)}`"
                )

            if y.ndim != 1:
                raise ValueError(f"`y` must have 1 dimension. " f"You provided y with ndim={y.ndim}")

        return func(self, X, y)

    return _check_X_y


def enforce_y_type(func):
    """Enforce y is 1-dimensional numpy array when pd.Series passed"""

    @functools.wraps(func)
    def _enforce_y_type(self, X, y):
        if isinstance(y, pd.Series):
            y = y.values
        return func(self, X, y)

    return _enforce_y_type


def check_fit_before_predict(func):
    """Check if the model has been fitted first before calling the predict method"""

    @functools.wraps(func)
    def _check_fit_before_predict(self, X):
        if not self.fitted:
            raise PredictWithoutFitError(model_name=self.name)
        return func(self, X)

    return _check_fit_before_predict


def get_estimator_name(estimator):
    """Get the name of the estimator

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator, sklearn.pipeline.Pipeline
        instance of model

    Returns
    -------
    str
        Name of the estimator. For TS Framework models, the name attribute
        of the estimator is returned, while for other model instances
        the class name will be returned. In case the input estimator is a pipeline
        the name of the last step in the pipeline, which is supposed
        to be a regressor/classifier will be returned.
    """

    def __get_estimator_name(estimator, name):

        if isinstance(estimator, Pipeline):
            if name.strip() == "":
                name_tmp = estimator.steps[-1][0]
            else:
                name_tmp = "__".join([name, estimator.steps[-1][0]])
            name = __get_estimator_name(estimator.steps[-1][1], name_tmp)
            return name
        else:
            if hasattr(estimator, "name"):
                if name.strip() == "":
                    return estimator.name
                else:
                    return "__".join([name, estimator.name])
            else:
                if name.strip() == "":
                    return estimator.__class__.__name__
                else:
                    return "__".join([name, estimator.__class__.__name__])

    name = __get_estimator_name(estimator, "")
    return name


def get_sales_data(n_dates=100, n_assortments=2, n_states=3, n_stores=3):
    """Load subset of Rossmann store sales dataset.

    This function loads a subset of the Rossmann store sales dataset
    from https://www.kaggle.com/c/rossmann-store-sales
    with the 100 stores with the highest sales overall.
    The data is for stores in Germany, in the date range ``2015-04-23`` to ``2015-07-31``.

    The data is returned as a `pandas.DataFrame`:

    - ``Date`` - DataFrame index, date of recorded sales numbers
    - ``Store`` - a unique Id for each store
    - ``Sales`` - the turnover for any given day (this is what you are predicting)
    - ``Open`` - an indicator for whether the store was open: 0 = closed, 1 = open
    - ``Promo`` - indicates whether a store is running a promo on that day
    - ``SchoolHoliday`` - indicates if the (Store, Date) was affected by the closure of public schools
    - ``StoreType`` - differentiates between 4 different store models: a, b, c, d
    - ``Assortment`` - describes an assortment level: a = basic, b = extra, c = extended
    - ``Promo2`` - Promo2 is a continuing and consecutive promotion for some stores:
      0 = store is not participating, 1 = store is participating
    - ``State`` - String code for state in Germany that the store is in
      (see https://en.wikipedia.org/wiki/States_of_Germany)
    - ``HolidayCode`` - the ``State`` prefixed with ``DE-``.

    The ``Assortment``, ``State`` and ``Store`` serve as data partitioning columns.
    ``HolidayCode`` will provide country specific holidays for the given ``Date``.
    ``Open``, ``Promo``, ``Promo2`` and ``SchoolHoliday`` serve as exogenous variables.
    ``Sales`` is the target column we will predict.

    Parameters
    ----------
    n_dates : int
        Number of days to be included for each series

    n_assortments : int
        Number of assortments to included

    n_states : int
        Number of states to included

    n_stores : int
        Number of stores to included


    Example
    -------
    >>> get_sales_data()
                Store  Sales  Open  Promo  SchoolHoliday StoreType Assortment  Promo2 State HolidayCode
    Date
    2015-04-23    906   8162  True  False          False         a          a   False    HE       DE-HE
    2015-04-23    251  16573  True  False          False         a          c   False    NW       DE-NW
    2015-04-23    320  13114  True  False          False         a          c   False    SH       DE-SH
    2015-04-23    335  11189  True  False          False         b          a    True    NW       DE-NW
    2015-04-23    336  10184  True  False          False         a          a   False    HE       DE-HE
    ...           ...    ...   ...    ...            ...       ...        ...     ...   ...         ...
    2015-07-31    817  23093  True   True           True         a          a   False    BE       DE-BE
    2015-07-31    831  15152  True   True           True         a          a   False    NW       DE-NW
    2015-07-31    906  15131  True   True           True         a          a   False    HE       DE-HE
    2015-07-31    586  17879  True   True           True         a          c   False    NW       DE-NW
    2015-07-31    251  22205  True   True           True         a          c   False    NW       DE-NW

    Returns
    -------
    pandas.DataFrame
        Rossmann store sales subset, see description above.

    Raises
    ------
    ValueError
        Error is raised if the number of assortments is higher than what dataset holds,
        if there are less than requested number of states within any assortment,
        or if there are not enough valid combinations of number of assortments, states and stores.
    """
    data_path = Path(__file__).parent / "data/rossmann_train_rich_top_100.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    valid_n_stores = (
        (df.groupby(["Assortment", "State"])["Store"].nunique()).loc[lambda x: x > n_stores].reset_index()
    )

    if valid_n_stores["Assortment"].nunique() < n_assortments:
        raise ValueError(
            f"There aren't enough Assortments ({valid_n_stores['Assortment'].nunique()} "
            f"from requested {n_assortments}) "
            f"complying with having {n_stores} stores in each of {n_states} states"
        )
    if any(valid_n_stores.groupby("Assortment")["State"].nunique() < n_states):
        raise ValueError(
            f"There aren't enough States from requested {n_states} within one Assortment "
            f"complying with having at least {n_stores} stores"
            f"{valid_n_stores.groupby('Assortment')['State'].nunique()}"
        )

    valid_assortment_states = (
        valid_n_stores.set_index(["Assortment", "State"])
        .groupby(["Assortment"])["Store"]
        .nlargest(n_states)
        .reset_index(level=2)
        .reset_index(level=1)
        .set_index(["Assortment", "State"])
        .index
    )

    if len(valid_assortment_states) < n_assortments * n_states:
        raise ValueError(
            f"There are not {n_assortments} store types all having at least {n_states} "
            f"unique states with at least {n_stores} stores in within the data."
            f"Try lowering `n_assortments`, `n_states` or `n_stores`"
            f"{valid_assortment_states}"
        )

    valid_stores = (
        df.set_index(["Assortment", "State"])
        .loc[lambda x: x.index.isin(valid_assortment_states[: n_assortments * n_states])]
        .reset_index()
        .groupby(["Assortment", "State", "Store"], as_index=False)["Sales"]
        .sum()
        .set_index("Store")
        .groupby(["Assortment", "State"])["Sales"]
        .nlargest(n_stores)
        .reset_index()["Store"]
        .values
    )

    data = df.loc[
        lambda x: (x["Store"].isin(valid_stores)) & (x.index.isin(x.index.drop_duplicates()[:n_dates]))
    ].sort_index()

    if len(data) != n_assortments * n_states * n_stores * n_dates:
        raise ValueError(
            f"There are not enough data for valid combinations of stores, store_types and states {data}"
        )

    return data


def generate_tsdata(n_dates=365, random_state=None):
    """Generate dummy daily time series data compatible with hcrystalball API.

    Parameters
    ----------
    n_dates : int
        Number of days to be included in the data

    random_state : int or RandomState instance
        Control random number generation following the scikit-learn pattern.
        Default is to use the global Numpy RNG.
        For reproducible results pass an int.
        To draw from a given RNG pass a RandomState.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame with `pandas.DatetimeIndex`
    y : pandas.Series
        Target values with `pandas.DatetimeIndex`
    """
    random_state = check_random_state(random_state)
    trend = np.arange(n_dates)
    noise = random_state.uniform(0, 1, len(trend))
    yearly_seasonality = 4 + 3 * np.sin(trend / 365 * 1 * np.pi)
    monthly_seasonality = 4 * np.sin(trend / 365 * 12 * np.pi + 365 / 2)
    weekly_seasonality = 4 * np.sin(trend / 365 * 52 * np.pi)
    trend = np.linspace(0, 5, n_dates)
    date_index = pd.date_range(start="2017-01-01", periods=len(trend), freq="D")
    y = yearly_seasonality + monthly_seasonality + weekly_seasonality + trend + noise
    y = pd.Series(y, index=date_index, name="target")
    X = pd.DataFrame(index=date_index)

    return X, y


def generate_multiple_tsdata(
    n_dates=10, n_regions=2, n_plants=3, n_products=4, country="DE", random_state=None
):
    """Provide easy way how to generate dummy data for tests or tutorial purposes

    Parameters
    ----------
    n_dates : int
        lenght of one time-series

    n_regions : int
        number of regions within column 'Region'

    n_plants : int
        number of plants within colum 'Plant'

    n_products : int
        number of products within colum 'Product'

    country : str
        ISO code of country or country-region

    random_state : int or RandomState instance,
        Control random number generation following the scikit-learn pattern.
        Default is to use the global Numpy RNG.
        For reproducible results pass an int.
        To draw from a given RNG pass a RandomState.

    Returns
    -------
    pandas.DataFrame
        Data with datetime index and following columns
        - "Region", "Plant", "Product" - for partitioning the data
        - "Country" - for holidays code
        - "Raining" - as bool exogenous variable
        - "Quantity" - as target variable
    """
    random_state = check_random_state(random_state)
    str_index = pd.date_range("2018-01-01", periods=n_dates, freq="D")
    regions = [f"region_{i}" for i in range(n_regions)]
    plants = [f"plant_{i}" for i in range(n_plants)]
    products = [f"product_{i}" for i in range(n_products)]
    dfs = []
    for region in regions:
        df_tmp = pd.DataFrame(
            columns=["Date", "Region", "Plant", "Product", "Country", "Raining", "Quantity"],
            index=range(len(str_index)),
        )
        df_tmp.loc[:, "Region"] = region
        df_tmp.loc[:, "Country"] = country

        for plant in plants:
            df_tmp.loc[:, "Plant"] = plant
            for product in products:
                df_tmp.loc[:, "Date"] = str_index
                df_tmp.loc[:, "Product"] = product
                _, y = generate_tsdata(n_dates=n_dates)
                df_tmp.loc[:, "Quantity"] = y.values
                df_tmp.loc[:, "Raining"] = random_state.choice(a=[False, True], size=(n_dates,))
                dfs.append(df_tmp.copy())

    return pd.concat(dfs).set_index("Date")


__all__ = [
    "get_estimator_repr",
    "get_sales_data",
    "generate_estimator_hash",
    "generate_multiple_tsdata",
    "generate_partition_hash",
    "generate_tsdata",
    "optional_import",
]
