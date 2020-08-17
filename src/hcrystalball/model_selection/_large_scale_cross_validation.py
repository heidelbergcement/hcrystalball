import itertools
from copy import deepcopy
import pandas as pd
import sys
import io
from collections import OrderedDict

from ._data_preparation import partition_data
from ._data_preparation import filter_data
from ._data_preparation import prepare_data_for_training
from ._model_selector_result import ModelSelectorResult
from .utils import persist_experts_in_physical_partition

# redirect prefect to the console
import logging

logger = logging.getLogger(__name__)


# ensure correct usage/skip of progress bar
def make_progress_bar(*args, **kwargs):
    """Create iterable as progress bar if available.

    Ensure simple loop is returned or tqdm_notebook progress bar when prerequisities met

    Returns
    -------
    iterable or tqdm_notebook
        tqdm_notebook based progress bar or simple iterable
    """
    pbar = args[0]
    try:
        from tqdm.auto import tqdm as tqdm_auto

        pbar = tqdm_auto(*args, **kwargs)
    except Exception as e:
        logging.warning(
            "No prerequisites (tqdm) installed for interactive progress bar, "
            "continuing without one. See the output in the console "
            "or check installation instructions."
            f"{e}"
        )
    return pbar


def select_model(
    df,
    target_col_name,
    partition_columns,
    grid_search,
    parallel_over_dict=None,
    frequency=None,
    country_code_column=None,
):
    """Find the best model from grid_search for each partition

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be used for model selection

    target_col_name : str
        name of target column

    partition_columns : list
        List of column names in the input dataframe to be used for partitioning the data.

    grid_search : sklearn.model_selection.GridSearchCV
        Instance compatible with Sklearn GridSearchCV

    parallel_over_dict : dict
        Dictionary with the partition label used for parallelization
        (e.g. {'Region':'Africa'})

    frequency : str
        Frequency at which model selection was done.
        It is mainly for bookeeping purposes and later used to instantiate result objects

    country_code_column : str
        Name of the column with ISO code of country/region, which can be used for supplying holiday.
        e.g. 'State' with values like 'DE', 'CZ' or 'Region' with values like 'DE-NW', 'DE-HE', etc.

    Returns
    -------
    list:
        List of ModelSelectorResults containing all important information for each partition
    """
    parallel_partition = parallel_over_dict or {}
    results = []

    if len(partition_columns) == 0:
        df_cv = {
            "labels": tuple([{"no_partition_label": ""}]),
            "data": tuple([df]),
        }
    else:
        df_cv = partition_data(df, list(set(partition_columns).difference(parallel_partition)))

    # will define progress bar if dependencies installed
    pbar = make_progress_bar(
        zip(df_cv["labels"], df_cv["data"]),
        total=len(df_cv["labels"]),
        leave=True,
        desc="Select Models"
        if parallel_over_dict is None
        else (", ").join([f"{k}: {v}" for k, v in parallel_over_dict.items()]),
    )

    # run model selection and create result object for each data partition
    for non_parallel_partition, data_part in pbar:
        # for non_parallel_partition, data_part in zip(df_cv["labels"], df_cv["data"]):
        grid_search_tmp = deepcopy(grid_search)
        X, y = data_part.drop(target_col_name, axis=1), data_part[target_col_name]
        if hasattr(grid_search_tmp, "autosarimax"):
            # ensures uniformly returned model for autosarimax within n_splits
            # by fitting the whole `pipeline` on first split (with exog cols and holiday transformer)
            # and appends the best found `model` to current grid_search param grid
            ind_last_split, _ = list(grid_search_tmp.cv.split(X))[0]
            best_sarimax = grid_search_tmp.autosarimax.fit(X.iloc[ind_last_split, :], y.iloc[ind_last_split])
            grid_search_tmp.param_grid.append({"model": [best_sarimax["model"]]})

        # searches among all models with found best Sarimax on last split (as opose to using AutoSarimax)
        grid_search_tmp.fit(X, y)
        # add convenient result object
        results.append(
            ModelSelectorResult(
                # more robust refit=True that makes sure, that models failing during fit
                # are ommited even if exception happens in wrapper's model's __init__
                best_model=grid_search_tmp.estimator.set_params(**grid_search_tmp.best_params_).fit(X, y),
                cv_results=pd.DataFrame(grid_search_tmp.cv_results_),
                cv_data=grid_search_tmp.scorer_.cv_data,
                model_reprs=grid_search_tmp.scorer_.estimator_ids,
                partition=OrderedDict(sorted({**parallel_partition, **non_parallel_partition}.items())),
                X_train=X,
                y_train=y,
                frequency=frequency,
                horizon=grid_search.cv.horizon,
                country_code_column=country_code_column,
            )
        )

    return results


def _define_model_selection_flow():
    """Define flow that runs model selection.

    Specifically data filtering, partitioning and model selection
    and optional persistence on a given dataset

    Returns
    -------
    prefect.Flow
    """

    from prefect import task, Flow, Parameter, unmapped

    with Flow("model selection") as flow:
        df = Parameter("data")
        grid_search = Parameter("grid_search")
        target_col_name = Parameter("target_col_name")
        country_code_column = Parameter("country_code_column")
        include_rules = Parameter("include_rules")
        exclude_rules = Parameter("exclude_rules")
        parallel_over_columns = Parameter("parallel_over_columns")
        partition_columns = Parameter("partition_columns")
        frequency = Parameter("frequency")
        output_path = Parameter("output_path")
        persist_cv_data = Parameter("persist_cv_data")
        persist_cv_results = Parameter("persist_cv_results")
        persist_model_reprs = Parameter("persist_model_reprs")
        persist_best_model = Parameter("persist_best_model")
        persist_partition = Parameter("persist_partition")
        persist_model_selector_results = Parameter("persist_model_selector_results")
        df_filtered = task(filter_data)(df=df, include_rules=include_rules, exclude_rules=exclude_rules)

        partitions = task(partition_data)(df=df_filtered, partition_by=parallel_over_columns)

        parallel_over_dicts, partition_dfs = partitions["labels"], partitions["data"]

        train_data = task(prepare_data_for_training).map(
            df=partition_dfs,
            frequency=unmapped(frequency),
            partition_columns=unmapped(partition_columns),
            parallel_over_columns=unmapped(parallel_over_columns),
            country_code_column=unmapped(country_code_column),
        )
        results = task(select_model).map(
            df=train_data,
            target_col_name=unmapped(target_col_name),
            grid_search=unmapped(grid_search),
            partition_columns=unmapped(partition_columns),
            parallel_over_dict=parallel_over_dicts,
            frequency=unmapped(frequency),
            country_code_column=unmapped(country_code_column),
        )

        write_ok = task(persist_experts_in_physical_partition).map(
            results=results,
            folder_path=unmapped(output_path),
            persist_cv_results=unmapped(persist_cv_results),
            persist_cv_data=unmapped(persist_cv_data),
            persist_model_reprs=unmapped(persist_model_reprs),
            persist_best_model=unmapped(persist_best_model),
            persist_partition=unmapped(persist_partition),
            persist_model_selector_results=unmapped(persist_model_selector_results),
        )

    flow.set_reference_tasks([write_ok])

    return flow


def run_model_selection(
    df,
    grid_search,
    target_col_name,
    frequency,
    partition_columns,
    parallel_over_columns,
    include_rules=None,
    exclude_rules=None,
    country_code_column=None,
    output_path="",
    persist_cv_results=False,
    persist_cv_data=False,
    persist_model_reprs=False,
    persist_best_model=False,
    persist_partition=False,
    persist_model_selector_results=True,
    visualize_success=False,
    executor=None,
):
    """Run parallel cross validation on data and select best model

    Best models are selected for each timeseries and if wanted persisted.'

    Parameters
    ----------
    df : pandas.DataFrame
        Container holding historical data for training

    grid_search : sklearn.model_selection.GridSearchCV
        Preconfigured grid search definition which determines which models
        and parameters will be tried

    target_col_name : str
        Name of target column

    frequency : str
        Temporal frequency of data.
        Data with different frequency will be resampled to this frequency.

    partition_columns : list, tuple
        Column names based on which the data should be split up / partitioned

    parallel_over_columns : list, tuple
        Subset of partition_columns, that are used to parallel split.

    include_rules : dict
        Dictionary with keys being column names and values being list of values to include in
        the output.

    exclude_rules : dict
        Dictionary with keys being column names and values being list of values to exclude
        from the output.

    country_code_column : str
        Name of the column with country code, which can be used for supplying holiday
        (i.e. having gridsearch with HolidayTransformer with argument `country_code_column`
        set to this one)

    output_path : str
        Path to directory for storing the output, default is cwd

    persist_cv_results  : bool
        If True cv_results of sklearn.model_selection.GridSearchCV
        as pandas df will be saved as pickle for each partition

    persist_cv_data : bool
        If True the pandas df detail cv data
        will be saved as pickle for each partition

    persist_model_reprs : bool
        If True model reprs
        will be saved as json for each partition

    persist_best_model : bool
        If True best model
        will be saved as pickle for each partition

    persist_partition : bool
        If True dictionary of partition label
        will be saved as json for each partition

    persist_model_selector_results : bool
        If True ModelSelectoResults with all important information
        will be saved as pickle for each partition

    visualize_success : bool
        If True, generate graph of task completion

    executor : prefect.engine.executors
        Provide prefect's executor.
        For more information see https://docs.prefect.io/api/latest/engine/executors.html

    Returns
    -------
    prefect.Flow, prefect.engine.state.State
        - flow itself
        - state of computations
    """
    from prefect.utilities.debug import raise_on_exception

    flow = _define_model_selection_flow()

    with raise_on_exception():
        state = flow.run(
            executor=executor,
            data=df,
            grid_search=grid_search,
            target_col_name=target_col_name,
            country_code_column=country_code_column,
            include_rules=include_rules,
            exclude_rules=exclude_rules,
            parallel_over_columns=parallel_over_columns,
            partition_columns=partition_columns,
            frequency=frequency,
            persist_cv_results=persist_cv_results,
            persist_cv_data=persist_cv_data,
            persist_model_reprs=persist_model_reprs,
            persist_best_model=persist_best_model,
            persist_partition=persist_partition,
            persist_model_selector_results=persist_model_selector_results,
            output_path=output_path,
        )

    if visualize_success:
        flow.visualize(flow_state=state)
    return flow, state


def select_model_general(
    df,
    grid_search,
    target_col_name,
    frequency,
    partition_columns=None,
    parallel_over_columns=None,
    executor=None,
    include_rules=None,
    exclude_rules=None,
    country_code_column=None,
    output_path="",
    persist_cv_results=False,
    persist_cv_data=False,
    persist_model_reprs=False,
    persist_best_model=False,
    persist_partition=False,
    persist_model_selector_results=False,
):
    """Run cross validation on data and select best model

    Best models are selected for each timeseries and if wanted persisted.

    Parameters
    ----------
    df : pandas.DataFrame
        Container holding historical data for training

    grid_search : sklearn.model_selection.GridSearchCV
        Preconfigured grid search definition which determines which models
        and parameters will be tried

    target_col_name : str
        Name of target column

    frequency : str
        Temporal frequency of data.
        Data with different frequency will be resampled to this frequency.

    partition_columns : list, tuple
        Column names based on which the data should be split up / partitioned

    parallel_over_columns : list, tuple
        Subset of partition_columns, that are used to parallel split.

    executor : prefect.engine.executors
        Provide prefect's executor. Only valid when `parallel_over_columns` is set.
        For more information see https://docs.prefect.io/api/latest/engine/executors.html

    include_rules : dict
        Dictionary with keys being column names and values being list of values to include in
        the output.

    exclude_rules : dict
        Dictionary with keys being column names and values being list of values to exclude
        from the output.

    country_code_column : str
        Name of the column with country code, which can be used for supplying holiday
        (i.e. having gridsearch with HolidayTransformer with argument `country_code_column`
        set to this one).

    output_path : str
        Path to directory for storing the output, default behavior is current working directory

    persist_cv_results  : bool
        If True cv_results of sklearn.model_selection.GridSearchCV as pandas df
        will be saved as pickle for each partition

    persist_cv_data : bool
        If True the pandas df detail cv data
        will be saved as pickle for each partition

    persist_model_reprs : bool
        If True model reprs will be saved as json for each partition

    persist_best_model : bool
        If True best model will be saved as pickle for each partition

    persist_partition : bool
        If True dictionary of partition label will be saved as json for each partition

    persist_model_selector_results : bool
        If True ModelSelectoResults with all important information
        will be saved as pickle for each partition

    Returns
    -------
    list
        List of ModelSelectorResult
    """
    if parallel_over_columns is not None:
        # run prefect flow with paralellism
        flow_result = run_model_selection(**locals())
        # access result of select_model and flatten it
        result = flow_result[1].result[flow_result[0].get_tasks("select_model")[0]].result
        flat_list = list(itertools.chain.from_iterable(result))

        return flat_list

    else:
        partition_columns = partition_columns or []
        # run without prefect
        df_prep = df.pipe(filter_data, include_rules=include_rules, exclude_rules=exclude_rules).pipe(
            prepare_data_for_training,
            frequency=frequency,
            partition_columns=partition_columns,
            country_code_column=country_code_column,
        )

        result = select_model(
            df=df_prep,
            target_col_name=target_col_name,
            partition_columns=partition_columns,
            grid_search=grid_search,
            parallel_over_dict=None,
            frequency=frequency,
            country_code_column=country_code_column,
        )

        if any(
            [
                persist_cv_results,
                persist_cv_data,
                persist_model_reprs,
                persist_partition,
                persist_best_model,
                persist_model_selector_results,
            ]
        ):
            persist_experts_in_physical_partition(
                results=result,
                folder_path=output_path,
                persist_cv_results=persist_cv_results,
                persist_cv_data=persist_cv_data,
                persist_model_reprs=persist_model_reprs,
                persist_partition=persist_partition,
                persist_best_model=persist_best_model,
                persist_model_selector_results=persist_model_selector_results,
            )

        return result
