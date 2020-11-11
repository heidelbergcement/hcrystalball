import pandas as pd
from pathlib import Path
from collections import Counter

from ._configuration import get_gridsearch
from ._configuration import add_model_to_gridsearch
from ._large_scale_cross_validation import select_model_general
from .utils import persist_experts_in_physical_partition
from ._model_selector_result import load_model_selector_result


def load_model_selector(folder_path):
    """Load information about stored model selection

    Parameters
    ----------
    folder_path : str
        path where .model_selector_result files are stored

    Returns
    -------
    ModelSelector
        Information about model selection for each partition
    """
    results = [
        load_model_selector_result(path=r.parent, partition_hash=r.stem)
        for r in Path(folder_path).glob("*.model_selector_result")
    ]
    model_selector = ModelSelector(horizon=results[0].horizon, frequency=results[0].frequency)
    model_selector.results = results
    return model_selector


class ModelSelector:
    """Enable large scale cross validation easily accessible.

    Through `create_gridsearch` and `select_model` methods
    run cross validation and present / persist all relevant information.

    Parameters
    ----------
    horizon : int
        How many steps ahead the predictions during model selection are to be made

    frequency : str
        Temporal frequency of the data which is to be used in model selection
        Data with different frequency will be resampled to this frequency.

    country_code_column : str, list
        Name of the column(s) with ISO code of country/region, which can be used for supplying holiday.
        If used, later provided data must have daily frequency.
        e.g. 'State' with values like 'DE', 'CZ' or 'Region' with values like 'DE-NW', 'DE-HE', etc.
    """

    def __init__(self, horizon, frequency, country_code_column=None):

        self.horizon = horizon
        self.frequency = frequency
        self.country_code_column = country_code_column

        self._partitions = None
        self._results = None
        self._stored_path = None

    def select_model(
        self,
        df,
        target_col_name,
        partition_columns=None,
        parallel_over_columns=None,
        executor=None,
        include_rules=None,
        exclude_rules=None,
        output_path="",
        persist_cv_results=False,
        persist_cv_data=False,
        persist_model_reprs=False,
        persist_best_model=False,
        persist_partition=False,
        persist_model_selector_results=False,
    ):
        """Run cross validation on data and selects best model.

        Best models are selected for each timeseries, stored in attribute `self.results`
        and if wanted also persisted.

        Parameters
        ----------
        df : pandas.DataFrame
            Container holding historical data for training

        target_col_name : str
            Name of target column

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

        output_path : str
            Path to directory for storing the output, default behavior is current working directory

        persist_cv_results  : bool
            If True cv_results of sklearn.model_selection.GridSearchCV as pandas df
            will be saved as pickle for each partition

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

        """

        params = {k: v for k, v in locals().items() if k != "self"}
        self.results = select_model_general(
            frequency=self.frequency,
            grid_search=self.grid_search,
            country_code_column=self.country_code_column,
            **params,
        )

    def create_gridsearch(
        self,
        n_splits=5,
        between_split_lag=None,
        scoring="neg_mean_absolute_error",
        country_code=None,
        holidays_days_before=0,
        holidays_days_after=0,
        holidays_bridge_days=False,
        sklearn_models=True,
        sklearn_models_optimize_for_horizon=False,
        autosarimax_models=False,
        autoarima_dict=None,
        prophet_models=False,
        tbats_models=False,
        exp_smooth_models=False,
        theta_models=False,
        average_ensembles=False,
        stacking_ensembles=False,
        stacking_ensembles_train_horizon=10,
        stacking_ensembles_train_n_splits=20,
        clip_predictions_lower=None,
        clip_predictions_upper=None,
        exog_cols=None,
    ):
        """Create grid_search attribute (`sklearn.model_selection.GridSearchCV`) based on selection criteria

        Parameters
        ----------
        n_splits : int
            How many cross-validation folds should be used in model selection

        between_split_lag : int
            How big lag of observations should cv_splits have
            If kept as None, horizon is used resulting in non-overlaping cv_splits

        scoring : str, callable
            String of sklearn regression metric name, or hcrystalball compatible scorer. For creation
            of hcrystalball compatible scorer use `make_ts_scorer` function.

        country_code : str
            Country code in str (e.g. 'DE'). Used in holiday transformer.
            Only one of `country_code_column` or `country_code` can be set.

        holidays_days_before : int
            Number of days before the holiday which will be taken into account
            (i.e. 2 means that new bool column will be created and will be True for 2 days before holidays,
            otherwise False)

        holidays_days_after : int
            Number of days after the holiday which will be taken into account
            (i.e. 2 means that new bool column will be created and will be True for 2 days after holidays,
            otherwise False)

        holidays_bridge_days : bool
            Overlaping `holidays_days_before` and `holidays_days_after` feature which serves for modeling
            between holidays working days

        sklearn_models : bool
            Whether to consider sklearn models

        sklearn_optimize_for_horizon: bool
            Whether to add to default sklearn behavior also models, that optimize predictions for each horizon

        autosarimax_models : bool
            Whether to consider auto sarimax models

        autoarima_dict : dict
            Specification of pmdautoarima search space

        prophet_models : bool
            Whether to consider FB prophet models

        exp_smooth_models : bool
            Whether to consider exponential smoothing models

        average_ensembles : bool
            Whether to consider average ensemble models

        stacking_ensembles : bool
            Whether to consider stacking ensemble models

        stacking_ensembles_train_horizon : int
            Which horizon should be used in meta model in stacking ensebmles

        stacking_ensembles_train_n_splits : int
            Number of splits used in meta model in stacking ensebmles

        clip_predictions_lower: float, int
            Minimal number allowed in the predictions

        clip_predictions_upper: float, int
            Maximal number allowed in the predictions

        exog_cols: list
            List of columns to be used as exogenous variables
        """
        if self.country_code_column is not None and country_code is not None:
            raise ValueError(
                "You can use either `country_code_column` in ModelSelector constructor "
                "or `country_code` here, not both."
            )

        params = {k: v for k, v in locals().items() if k not in ["self"]}

        self.grid_search = get_gridsearch(
            frequency=self.frequency,
            horizon=self.horizon,
            country_code_column=self.country_code_column,
            **params,
        )

    def add_model_to_gridsearch(self, model):
        """Extend `self.grid_search` parameter grid with provided model.

        Adds given model or list of models to the gridsearch under 'model' step

        Parameters
        ----------
        model : sklearn compatible model or list of sklearn compatible models
            model(s) to be added to provided grid search
        """
        self.grid_search = add_model_to_gridsearch(model, self.grid_search)

    def persist_results(
        self,
        folder_path="results",
        persist_cv_results=False,
        persist_cv_data=False,
        persist_model_reprs=False,
        persist_best_model=False,
        persist_partition=False,
        persist_model_selector_results=True,
    ):
        """Store expert files for each partition.

        The file names follow {partition_hash}.{expert_type} e.g. 795dab1813f05b1abe9ae6ded93e1ec4.cv_data

        Stores value of folder_path argument to `self.stored_path`

        Parameters
        ----------
        folder_path : str
            Path to the directory, where expert files are stored,
            by default '' resulting in current working directory

        persist_cv_results : bool
            If True `cv_results` of sklearn.model_selection.GridSearchCV as pandas df will be saved as pickle
            for each partition

        persist_cv_data : bool
            If True the pandas df detail cv data will be saved as pickle for each partition

        persist_model_reprs : bool
            If True model reprs will be saved as json for each partition

        persist_best_model : bool
            If True best model will be saved as pickle for each partition

        persist_partition : bool
            If True dictionary of partition label will be saved
            as json for each partition

        persist_model_selector_results : bool
            If True ModelSelectoResults with all important information will be saved
            as pickle for each partition
        """
        params = {k: v for k, v in locals().items() if k != "self"}
        self.stored_path = persist_experts_in_physical_partition(results=self.results, **params)

    @property
    def results(self):
        """Results for each partition

        Returns
        -------
        list
            List of `ModelSelectorResult` objects

        Raises
        ------
        ValueError
            If `select_model` was not called before
        """
        if self._results is None:
            raise ValueError("You need to run `select_model` first to obtain the `results`!")
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def partitions(self):
        """List of partitions the model selection was ran on.

        Created only after calling `select_model`.

        Returns
        -------
        list
            List of dictionaries of partitions

        Raises
        ------
        ValueError
            If `select_model` was not called before
        """
        if self._partitions is None:
            if self._results is None:
                raise ValueError("You need to run `select_model` first to obtain the `partitions`!")
            else:
                self._partitions = self.get_partitions()
        return self._partitions

    @partitions.setter
    def partitions(self, value):
        self._partitions = value

    @property
    def stored_path(self):
        """Path where `ModelSelector` object was stored

        Created only after calling `persist_results`.

        Returns
        -------
        str
            Pathlike string to the folder containing stored ModelSelector object

        Raises
        ------
        ValueError
            If `presist_results` was not called before
        """
        if self._stored_path is None:
            raise ValueError("You need to run `persist_results` first to obtain the `stored_path`!")
        return self._stored_path

    @stored_path.setter
    def stored_path(self, value):
        self._stored_path = value

    def get_result_for_partition(self, partition=None):
        """Provide result for given partition

        Parameters
        ----------
        partition : str, dict
            partition_hash or partition_dict of data to which result is tied to

        Returns
        -------
        ModelSelectorResult
            result of model selection for given partition

        Raises
        ------
        ValueError
            if partition is not present in the results
        """
        if isinstance(partition, dict):
            result = [result for result in self.results if result.partition == partition]
        else:
            result = [result for result in self.results if result.partition_hash == partition]

        if not result:
            raise ValueError(
                f"Partition {partition} does not exist. Run 'get_partitions()' to see available options."
            )
        return result[0]

    def get_partitions(self, as_dataframe=False):
        """Provide overview of partitions for which results are available

        Parameters
        ----------
        as_dataframe : bool
            Whether to return partitions as pandas.DataFrame -> returns list of dicts

        Returns
        -------
        pandas.DataFrame, list[dict, dict, ...]
            partitions available in model selector results
        """
        partitions = [result.partition for result in self.results]
        if as_dataframe:
            return pd.DataFrame(partitions)
        return partitions

    def plot_best_wrapper_classes(self, title="Most often selected classes", **plot_params):
        """Plot number of selected wrapper classes that were picked as best models

        Parameters
        ----------
        title : str
            Title of the plot

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot of most selected wrapper classes
        """
        no_of_best_wrapper_classes = Counter([res.best_model_name for res in self.results])
        return pd.Series(no_of_best_wrapper_classes).plot(kind="barh", title=title, **plot_params)

    def plot_results(self, partitions=None, plot_from=None, **plot_params):
        """Plot training data and cv forecasts for each of the partition

        Parameters
        ----------
        partitions : list
            List of partitions to plot results for

        plot_from : str
            Date from which to show the plot
            e.g. '2019-12-31', '2019', or '2019-12'

        Returns
        -------
        list
            List of `matplotlib.axes._subplots.AxesSubplot` for each partition
        """
        partitions = partitions or self.partitions
        plts = []

        for partition in partitions:
            partition_result = self.get_result_for_partition(partition)
            plts.append(
                partition_result.plot_result(
                    plot_from=plot_from,
                    title=(" ").join([f"{k}={v}" for k, v in partition.items()]),
                    **plot_params,
                )
            )

        return plts

    def __repr__(self):
        r = "ModelSelector\n"
        r += "-------------\n"
        r += f"  frequency: {self.frequency}\n"
        r += f"  horizon: {self.horizon}\n"
        r += f"  country_code_column: {self.country_code_column}\n"
        if self._results is not None:
            r += f"  results: List of {len(self.results)} ModelSelectorResults\n"
            r += f"  paritions: List of {len(self.partitions)} partitions\n"
            for partition in self.partitions:
                r += f"     {dict(partition)}\n"
        r += "-------------\n"
        return r
