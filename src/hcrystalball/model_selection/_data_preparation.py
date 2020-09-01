import pandas as pd
import numpy as np


def partition_data(df, partition_by):
    """Partition data by values found in one or more columns.

    For each of the selected columns the unique values will
    be determined and a selection will be made for each element
    in the cross product of the unique values.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be partitioned

    partition_by : list
        Column names to partition by

    Returns
    -------
    dict
        Partition dictionary with keys:

        * labels : Tuple of dictionaries whose keys are the column names
                   and values are the actual values in the column
        * data   : Tuple of pandas.DataFrame objects holding the subset of the data with
    """
    labels = []
    data = []

    for key, group in df.groupby(partition_by):
        if not isinstance(key, tuple):
            key = (key,)
        labels.append(dict(zip(partition_by, key)))
        data.append(group.drop(partition_by, axis=1))
    return {"labels": tuple(labels), "data": tuple(data)}


def partition_data_by_values(df, column, partition_values, default_df=None):
    """Partition data by one column and a fixed set ov values within that column.

    If a value is not present, optionally provide default data for the partition.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be partitioned

    column : str
        column with values to partition by

    partition_values: list
        values to partition by

    default_df : pandas.DataFrame
        data to be used as default in case value is not present

    Returns
    -------
    dict
        Partition dictionary with keys:

        * labels : Tuple of dictionaries whose keys are the column names
                   and values are the actual values in the column
        * data   : Tuple of pandas.DataFrame objects holding the subset of the data with
    """
    labels = []
    data = []
    for v in partition_values:
        part = df[df[column] == v]
        if part.empty:
            if default_df is None:
                continue
            part = default_df.copy()
        labels.append({column: v})
        data.append(part.drop(column, axis=1))
    return {"labels": tuple(labels), "data": tuple(data)}


def filter_data(df, include_rules=None, exclude_rules=None):
    """Filter provided dataframe by {column:value} rules.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be filtered

    include_rules : dict
        Rules for what to include. The keys of the dictionaries should be the name of the filtered
        columns, while the values of the dictionary should be list of values to include.

    exclude_rules : dict
        Rules for what to include. The keys of the dictionaries should be the name of the filtered
        columns, while the values of the dictionary should be list of values to exclude.

    Returns
    -------
    pandas.DataFrame:
        Data of the same type / format as the output with the filters applied.
    """
    df = df.copy()

    # Check for overlap between include and exclude rules and raise an exception if there is any overlap
    if (include_rules is not None) & (exclude_rules is not None):
        common_keys = set(include_rules.keys()) & set(exclude_rules.keys())
        for key in common_keys:
            common_values = set(include_rules[key]) & set(exclude_rules[key])
            if len(common_values) > 0:
                raise ValueError(
                    f"Overlap is found in include_rules and exclude_rules in key `{key}`."
                    f"This is not allowed"
                )

    if include_rules is not None:
        if not isinstance(include_rules, dict):
            raise TypeError("`include_rules` is not a dictionary.")

        mask = pd.Series(index=df.index)
        mask[:] = True
        for key, value in include_rules.items():
            mask = mask & (df[key].isin(value))

        df = df.loc[mask, :]

    if exclude_rules is not None:
        if not isinstance(exclude_rules, dict):
            raise TypeError("`exclude_rules` is not a dictionary.")

        mask = pd.Series(index=df.index)
        mask[:] = True
        for key, value in exclude_rules.items():
            mask = mask & (~df[key].isin(value))

        df = df.loc[mask, :]

    return df


def prepare_data_for_training(
    df,
    frequency,
    partition_columns,
    parallel_over_columns=None,
    country_code_column=None,
):
    """Prepare data for model selection.

    Transforms data to a form handled by model selection / training,
    ensuring correct frequency and filling NaN

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be transformed, must have a date column of type `str`

    frequency : str
        frequency identifier ('D', 'M' etc.)

    parallel_over_columns: list
        column(s) which define logical segmentation of data for training

    country_code_column : str
        name of columns from which to take holiday ISO information

    Returns
    -------
    pandas.DataFrame
        Resampled, aggregated data
    """
    parallel_over_columns = parallel_over_columns or {}
    partition_columns = list(set(partition_columns).difference(parallel_over_columns))

    # TODO this check should go into separate function for check of exogeneous variables
    country_code_columns = (
        [country_code_column] if isinstance(country_code_column, str) else country_code_column
    )
    if country_code_column and not set(country_code_columns).issubset(set(df.columns)):
        raise KeyError(
            f"Column(s) {country_code_column} provided as `country_code_column` is not in dataframe!"
        )

    df = df.astype({col: "category" for col in partition_columns})

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.columns.difference(num_cols + partition_columns)

    if len(partition_columns) != 0:
        df = df.groupby(partition_columns)

    df = (
        df.resample(frequency)
        .agg({**{col: "sum" for col in num_cols}, **{col: "last" for col in cat_cols}})
        .fillna(
            {
                **{col: 0 for col in num_cols},
                **{col: "" for col in cat_cols.difference(country_code_columns or set())},
            }
        )
        .reset_index(partition_columns)
    )

    if country_code_columns:
        df[country_code_columns] = df[country_code_columns].fillna(method="ffill")

    return df
