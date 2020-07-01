from hcrystalball.model_selection import select_model
import pytest


@pytest.mark.parametrize(
    "train_data, grid_search, parallel_over_dict",
    [("two_regions", "", {"Region": "region_0"}), ("two_regions", "", None)],
    indirect=["train_data", "grid_search"],
)
def test_select_model(train_data, grid_search, parallel_over_dict):

    _train_data = train_data

    if parallel_over_dict:
        col, value = list(parallel_over_dict.items())[0]
        _train_data = train_data[train_data[col] == value].drop(columns="Region")

    partition_columns = ["Region", "Product"]

    results = select_model(
        _train_data,
        target_col_name="Quantity",
        partition_columns=partition_columns,
        parallel_over_dict=parallel_over_dict,
        grid_search=grid_search,
        country_code_column="Holidays_code",
    )
    if parallel_over_dict:
        partitions = (
            train_data.loc[train_data[col] == value, partition_columns]
            .drop_duplicates()
            .to_dict(orient="records")
        )
    else:
        partitions = train_data[partition_columns].drop_duplicates().to_dict(orient="records")

    assert len(results) == len(partitions)

    for result in results:
        assert result.best_model_name == "good_dummy"
        assert result.partition in partitions
