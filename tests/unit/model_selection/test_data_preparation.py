import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from hcrystalball.model_selection import partition_data
from hcrystalball.model_selection import partition_data_by_values
from hcrystalball.model_selection import filter_data
from hcrystalball.model_selection import prepare_data_for_training


def test_partition_data(test_data_raw):

    n_region = 2
    n_plant = 3
    n_product = 4
    partition_by = ["Region", "Plant", "Product"]
    partitions = partition_data(test_data_raw, partition_by)

    assert isinstance(partitions, dict)

    assert "labels" in partitions
    assert isinstance(partitions["labels"], tuple)
    assert len(partitions["labels"]) == n_region * n_plant * n_product

    assert "data" in partitions
    assert isinstance(partitions["data"], tuple)
    assert len(partitions["data"]) == n_region * n_plant * n_product

    # Note that here it is assumed that the order of nesting is the order of the partition_by columns
    ind = 0
    for ir in range(n_region):
        region_value = "region_" + str(ir)
        for ip in range(n_plant):
            plant_value = "plant_" + str(ip)
            for ipr in range(n_product):
                product_value = "product_" + str(ipr)
                assert partitions["labels"][ind]["Region"] == region_value
                assert partitions["labels"][ind]["Plant"] == plant_value
                assert partitions["labels"][ind]["Product"] == product_value

                mask = (
                    (test_data_raw["Region"] == region_value)
                    & (test_data_raw["Plant"] == plant_value)
                    & (test_data_raw["Product"] == product_value)
                )

                df_tmp = test_data_raw.loc[mask, :]
                df_tmp.drop(["Region", "Plant", "Product"], axis=1, inplace=True)
                assert_frame_equal(df_tmp, partitions["data"][ind])

                ind += 1


def test_partition_data_by_values(test_data_raw):

    res = partition_data_by_values(
        test_data_raw,
        column="Plant",
        partition_values=["plant_0", "plant_23"],
        default_df=pd.DataFrame(
            {"Plant": ["dummy"], "Region": ["dummy"], "Product": ["dummy"], "Quantity": [0.0]}
        ),
    )
    assert res["labels"][0]["Plant"] == "plant_0"
    assert len(res["data"][0]) == 80
    assert res["labels"][1]["Plant"] == "plant_23"
    assert len(res["data"][1]) == 1


def test_filter_data_include(test_data_raw):

    rules = {"Plant": ["plant_0", "plant_1"], "Region": ["region_0"]}

    df = filter_data(test_data_raw, include_rules=rules)

    assert isinstance(df, pd.DataFrame)
    for key, value in rules.items():
        assert value == list(df[key].unique())

    with pytest.raises(TypeError):
        _ = filter_data(test_data_raw, include_rules=[])


def test_filter_data_exclude(test_data_raw):

    rules = {"Plant": ["plant_0", "plant_1"], "Region": ["region_0"]}

    df = filter_data(test_data_raw, exclude_rules=rules)

    assert isinstance(df, pd.DataFrame)
    for key, value in rules.items():
        filtered_values = list(df[key].unique())
        for ival in value:
            assert ival not in filtered_values

    with pytest.raises(TypeError):
        _ = filter_data(test_data_raw, exclude_rules=[])


def test_filter_data_include_and_exclude(test_data_raw):

    include_rules = {"Plant": ["plant_0"]}
    exclude_rules = {"Region": ["region_0"]}

    df = filter_data(test_data_raw, include_rules=include_rules, exclude_rules=exclude_rules)

    assert isinstance(df, pd.DataFrame)
    for key, value in exclude_rules.items():
        filtered_values = list(df[key].unique())
        for ival in value:
            assert ival not in filtered_values

    for key, value in include_rules.items():
        filtered_values = list(df[key].unique())
        for ival in value:
            assert ival in filtered_values


def test_filter_data_include_and_exclude_overlapping_conditions(test_data_raw):

    include_rules = {"Plant": ["plant_0", "plant_1"]}
    exclude_rules = {"Plant": ["plant_1"], "Region": ["region_0"]}

    with pytest.raises(ValueError):
        _ = filter_data(test_data_raw, include_rules=include_rules, exclude_rules=exclude_rules)


@pytest.mark.parametrize(
    "unprepared_data, prepared_data, target_col_name, partition_columns, expected_error",
    [
        ("", "", "delivery_quantity", ["cem_type"], None),
        ("", "without_logical_partition", "delivery_quantity", [], None),
    ],
    indirect=["unprepared_data", "prepared_data"],
)
def test_prepare_data_for_training(
    unprepared_data, prepared_data, target_col_name, partition_columns, expected_error
):

    if expected_error:
        with pytest.raises(expected_error):
            result = prepare_data_for_training(
                unprepared_data, frequency="D", partition_columns=partition_columns
            )
    else:
        result = prepare_data_for_training(
            unprepared_data, frequency="D", partition_columns=partition_columns
        )

        prepared_data = prepared_data.rename({"target": target_col_name}, axis=1)
        assert_frame_equal(prepared_data, result, check_like=True)
