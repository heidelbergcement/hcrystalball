import functools
import pandas as pd

from hcrystalball.utils import generate_estimator_hash
from hcrystalball.utils import generate_partition_hash
from hcrystalball.utils import get_estimator_name
from .utils import _persist_to_file
from .utils import _load_file

load_model_selector_result = functools.partial(_load_file, expert_type="model_selector_result")


class ModelSelectorResult:
    """Consolidate infromation/methods from cross validation for 1 time series

    Store all relevant information about model selection and provide
    utility methods (e.g. plot_model_performance) and data (e.g. df_plot)
    for easier access to further insights.

    Parameters
    ----------
    best_model : sklearn compatible estimator
        best model found during model selection

    cv_results : pandas.DataFrame
        cv_results of sklearn.model_selection.GridSearchCV in form of DataFrame

    cv_data : pandas.DataFrame
        data with models predictions, cv split indication and true target values

    model_reprs : dict
        dictionary of model representations used in model selection
        in form of {model_hash : model_repr}

    partition : dict
        dictionary indicating for which part of the data the model selection results belong to
        e.g. {"Region":"Canada", "Product":"Chips"}

    X_train : pandas.DataFrame
        training data features

    y_train : pandas.Series
        training data target

    frequency : str
        temporal frequency of data on which the model was trained / selected

    horizon : int
        how many steps ahead predictions were made

    country_code_column : str
        Name of the column with ISO code of country/region, which can be used for supplying holiday.
        e.g. 'State' with values like 'DE', 'CZ' or 'Region' with values like 'DE-NW', 'DE-HE', etc.
    """

    def __init__(
        self,
        best_model,
        cv_results,
        cv_data,
        model_reprs,
        partition,
        X_train,
        y_train,
        frequency,
        horizon,
        country_code_column,
    ):
        self.best_model = best_model
        self.cv_results = cv_results
        self.cv_data = cv_data
        self.model_reprs = model_reprs
        self.partition = partition
        self.X_train = X_train
        self.y_train = y_train
        self.frequency = frequency
        self.horizon = horizon
        self.country_code_column = country_code_column

        self.best_model_hash = generate_estimator_hash(best_model)
        self.best_model_cv_data = self.cv_data.rename({self.best_model_hash: "best_model"}, axis=1)[
            ["split", "y_true", "best_model"]
        ]
        self.best_model_name = get_estimator_name(best_model).replace("model__", "")
        self.best_model_cv_results = self.cv_results[self.cv_results["rank_test_score"] == 1].iloc[0]
        self.best_model_repr = self.model_reprs[self.best_model_hash]
        self.partition_hash = generate_partition_hash(self.partition)

        self._persist_attrs = sorted(set(self.__dict__.keys()).difference(["self"]))
        self._df_plot = None

    def persist(self, attribute_name=None, path=""):
        """Persist whole object or particular object attributes

        Parameters
        ----------
        attribute_name : str
            Name of the attribute to be stored - stores whole object

        path: str
            Where to store the object or object attribute
            Creates file named as {partition_hash}.{attribute_name} by default at current working directory

        Raises
        ------
        ValueError
            If attribute not a valid option. Lists available ones
        """
        if attribute_name is None:
            _persist_to_file(
                data=self,
                expert_type="model_selector_result",
                partition_hash=self.partition_hash,
                path=path,
            )
        else:
            if attribute_name not in self._persist_attrs:
                raise ValueError(
                    f"Parameter attribute must be one of {self._persist_attrs}, "
                    f"but you provided {attribute_name}"
                )

            _persist_to_file(
                data=getattr(self, attribute_name),
                expert_type=attribute_name,
                partition_hash=self.partition_hash,
                path=path,
            )

    @property
    def df_plot(self):
        """Training data suitable for plotting.

        Utility, that prepares data from model selection to be used for further model performance analysis

        Returns
        -------
        pandas.DataFrame
            Data suitable for plotting
        """
        # ensures this will be called only once
        if self._df_plot is None:
            self._df_plot = (
                pd.merge(
                    self.y_train.rename("actuals"),
                    self.best_model_cv_data[["best_model", "split"]].rename(
                        {"best_model": f"cv_forecast({self.best_model_name})", "split": "cv_split"},
                        axis=1,
                    ),
                    left_index=True,
                    right_index=True,
                    how="outer",
                )
                # TODO add different error functions??
                .assign(error=lambda x: (x["actuals"] - x[f"cv_forecast({self.best_model_name})"]).abs())
                .assign(cv_split_str=lambda x: "cv_split=" + x["cv_split"].astype(str))
                .assign(
                    mae=lambda x: x["cv_split_str"].map(x.groupby(["cv_split_str"])["error"].mean().to_dict())
                )
                .assign(cv_split_str=lambda x: x["cv_split_str"] + ", mae=" + x["mae"].round(2).astype(str))
            )
        return self._df_plot

    @property
    def cv_splits_overlap(self):
        """Indicator for cv_splits overlap in training data

        Returns
        -------
        bool
            Whether cv_splits in training data contain overlap
        """
        return sum(self.df_plot.reset_index().groupby("index")["cv_split"].count() > 1) > 0

    def plot_result(self, plot_from=None, **plot_params):
        """Plot model performance from given `plot_from` timestamp

        Parameters
        ----------
        plot_from : str
            date from which to show actuals, cv_forecast and forecast,
            Default behavior does not filter dates

        plot_params : kwargs
            plotting parameters passed down to pandas.DataFrame.plot()
            dependent on your plotting backend
            e.g. figsize = (16,9), `title = 'Performance of Model'`

        Returns
        -------
        pandas.DataFrame.plot()
            plot depending on your plotting backend, by default plot from matplotlib
        """

        df = self.df_plot

        plot_from = plot_from or df.index.min()

        if self.cv_splits_overlap:
            # plot each split separately
            plts = []
            cv_fcst_col = f"cv_forecast({self.best_model_name})"
            for split in df["cv_split"].dropna().unique():
                plt = (
                    df[plot_from:]
                    .drop(
                        ["error", cv_fcst_col, "cv_split", "cv_split_str", "mae"],
                        axis=1,
                    )
                    .plot(**plot_params)
                )
                # get limits for shaded areas
                min_y, max_y = plt.get_ylim()

                plt_cv = df[plot_from:].loc[lambda x: x["cv_split"] == split, [cv_fcst_col, "cv_split_str"]]
                plt = plt_cv[[cv_fcst_col]].plot(
                    ax=plt,
                    title=f"{plt.get_title()} | ({plt_cv['cv_split_str'].unique()[0]})",
                )
                if not plt_cv.empty:
                    plt.fill_between(
                        x=plt_cv.index,
                        y1=max_y,
                        y2=min_y,
                        alpha=[0.2, 0.4][split % 2],
                        color=["gray"],
                        label=plt_cv["cv_split_str"].unique()[0],
                    )
                plt.legend(facecolor="white", framealpha=0.8, frameon=True)
                plts.append(plt)
            return plts
        else:
            plt = (
                df[plot_from:].drop(["error", "cv_split", "cv_split_str", "mae"], axis=1).plot(**plot_params)
            )
            # get limits for shaded areas
            min_y, max_y = plt.get_ylim()

            for split in df["cv_split"].dropna().unique():
                tmp_df = df[plot_from:].loc[lambda x: x["cv_split"] == split, ["cv_split_str"]]
                if not tmp_df.empty:
                    plt.fill_between(
                        x=tmp_df.index,
                        y1=max_y,
                        y2=min_y,
                        alpha=[0.2, 0.4][split % 2],
                        color=["gray"],
                        label=tmp_df["cv_split_str"].unique()[0],
                    )
            # include shaded area labels in the legend
            plt.legend(facecolor="white", framealpha=0.8, frameon=True)

            return plt

    def plot_error(self, **plot_params):
        """Plot model absolute error during model selection

        Parameters
        ----------
        plot_params : kwargs
            plotting parameters passed down to pandas.DataFrame.plot()
            dependent on your plotting backend
            e.g. figsize = (16,9), `title = 'Performance of Model'`

        Returns
        -------
        pandas.DataFrame.plot()
            plot depending on your plotting backend, by default plot from matplotlib
        """
        # TODO add different error functions??
        df = self.df_plot

        return df.dropna().groupby("cv_split_str")["error"].plot(legend=True, **plot_params)

    def __repr__(self):
        return (
            "ModelSelectorResult\n"
            f"-------------------\n"
            f"  best_model_name: {self.best_model_name}\n"
            f"  frequency: {self.frequency}\n"
            f"  horizon: {self.horizon}\n\n"
            f"  country_code_column: {self.country_code_column}\n\n"
            f"  partition: {dict(self.partition)}\n"
            f"  partition_hash: {self.partition_hash}\n\n"
            f"  df_plot: DataFrame {self.df_plot.shape} suited for plotting cv results with .plot()\n"
            f"  X_train: DataFrame {self.X_train.shape} with training feature values\n"
            f"  y_train: DataFrame {self.y_train.shape} with training target values\n"
            f"  cv_results: DataFrame {self.cv_results.shape} with gridsearch cv info\n"
            f"  best_model_cv_results: Series with gridsearch cv info\n"
            f"  cv_data: DataFrame {self.cv_data.shape} "
            f"with models predictions, split and true target values\n"
            f"  best_model_cv_data: DataFrame {self.best_model_cv_data.shape} "
            f"with model predictions, split and true target values\n\n"
            f"  model_reprs: Dict of model_hash and model_reprs\n"
            f"  best_model_hash: {self.best_model_hash}\n"
            f"  best_model: {self.best_model}\n"
            "-------------------\n"
        )
