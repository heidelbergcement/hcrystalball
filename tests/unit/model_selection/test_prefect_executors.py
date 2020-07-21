import pytest
from hcrystalball.model_selection import run_model_selection
from hcrystalball.model_selection import select_model_general
from hcrystalball.model_selection import ModelSelectorResult


@pytest.mark.parametrize(
    "train_data, grid_search, parallel_columns",
    [("two_regions", "", ["Region"])],
    indirect=["train_data", "grid_search"],
)
def test_prefect_executors(train_data, grid_search, parallel_columns):
    from prefect.engine.executors import DaskExecutor
    from prefect.engine.executors import LocalDaskExecutor
    from prefect.engine.executors import LocalExecutor
    from dask.distributed import Client

    client = Client()

    executors = {
        "dask_already_running": DaskExecutor(address=client.scheduler.address),
        "local": LocalExecutor(),
        "local_dask": LocalDaskExecutor(),
        # this spins up LocalDaskExecutor, but just to check the interface
        "dask_create_on_call": DaskExecutor(),
    }

    for executor_name, executor in executors.items():
        flow, state = run_model_selection(
            df=train_data,
            grid_search=grid_search,
            target_col_name="Quantity",
            frequency="D",
            partition_columns=["Product"],
            parallel_over_columns=parallel_columns,
            include_rules=None,
            exclude_rules=None,
            country_code_column="Holidays_code",
            output_path="",
            persist_cv_results=False,
            persist_cv_data=False,
            persist_model_reprs=False,
            persist_best_model=False,
            persist_partition=False,
            persist_model_selector_results=False,
            visualize_success=False,
            executor=executor,
        )
        assert state.is_successful()

        results = select_model_general(
            df=train_data,
            grid_search=grid_search,
            target_col_name="Quantity",
            frequency="D",
            partition_columns=["Product"],
            parallel_over_columns=parallel_columns,
            executor=executor,
            include_rules=None,
            exclude_rules=None,
            country_code_column="Holidays_code",
            output_path="",
            persist_cv_results=False,
            persist_cv_data=False,
            persist_model_reprs=False,
            persist_best_model=False,
            persist_partition=False,
            persist_model_selector_results=False,
        )

        assert len(results) == len(train_data[parallel_columns + ["Product"]].drop_duplicates())
        assert isinstance(results[0], ModelSelectorResult)

        if executor_name == "dask_already_running":
            client.shutdown()

    if client.status != "closed":
        client.shutdown()
