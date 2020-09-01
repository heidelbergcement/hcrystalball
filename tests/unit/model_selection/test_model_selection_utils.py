from hcrystalball.model_selection.utils import persist_experts_in_physical_partition
from hcrystalball.model_selection.utils import _load_file
from hcrystalball.model_selection import select_model
import pickle
import json
import os
import pytest


@pytest.mark.parametrize("train_data, grid_search", [("", "")], indirect=["train_data", "grid_search"])
def test_persist_experts_in_physical_partition(train_data, grid_search, tmp_path):

    partition_columns = ["Product"]
    results = select_model(
        train_data,
        target_col_name="Quantity",
        partition_columns=partition_columns,
        grid_search=grid_search,
    )

    persist_experts_in_physical_partition(
        tmp_path,
        results,
        persist_cv_results=True,
        persist_cv_data=True,
        persist_model_reprs=True,
        persist_best_model=True,
        persist_partition=True,
        persist_model_selector_results=True,
    )
    files = os.listdir(tmp_path)

    assert len(files) == 18

    for result in results:
        with open(os.path.join(tmp_path, result.partition_hash + ".cv_results"), "rb") as file:
            cv_results = pickle.load(file)
            assert isinstance(cv_results, type(result.cv_results))
            assert all(cv_results.columns == result.cv_results.columns)

        with open(os.path.join(tmp_path, result.partition_hash + ".cv_data"), "rb") as file:
            cv_data = pickle.load(file)
            assert isinstance(cv_data, type(result.cv_data))
            assert all(cv_data.columns == result.cv_data.columns)

        with open(os.path.join(tmp_path, result.partition_hash + ".model_reprs")) as file:
            model_reprs = json.load(file)
            assert isinstance(model_reprs, type(result.model_reprs))
            assert model_reprs == result.model_reprs

        with open(os.path.join(tmp_path, result.partition_hash + ".best_model"), "rb") as file:
            model = pickle.load(file)
            assert isinstance(model, type(result.best_model))
            assert str(model.get_params()) == str(result.best_model.get_params())

        with open(os.path.join(tmp_path, result.partition_hash + ".partition")) as file:
            partition = json.load(file)
            assert partition == result.partition

        with open(
            os.path.join(tmp_path, result.partition_hash + ".model_selector_result"),
            "rb",
        ) as file:
            model_selector_result = pickle.load(file)
            assert isinstance(model_selector_result, type(result))
            assert str(model_selector_result.__dict__) == str(result.__dict__)


@pytest.mark.parametrize("train_data, grid_search", [("", "")], indirect=["train_data", "grid_search"])
def test_load_expert(train_data, grid_search, tmp_path):

    partition_columns = ["Product"]
    results = select_model(
        train_data,
        target_col_name="Quantity",
        partition_columns=partition_columns,
        grid_search=grid_search,
    )

    persist_experts_in_physical_partition(
        tmp_path,
        results,
        persist_cv_results=True,
        persist_cv_data=True,
        persist_model_reprs=True,
        persist_best_model=True,
        persist_partition=True,
        persist_model_selector_results=True,
    )
    files = os.listdir(tmp_path)

    assert len(files) == 18

    for result in results:
        cv_results = _load_file(partition_label=result.partition, path=tmp_path, expert_type="cv_results")
        assert isinstance(cv_results, type(result.cv_results))
        assert all(cv_results.columns == result.cv_results.columns)

        cv_data = _load_file(partition_label=result.partition, path=tmp_path, expert_type="cv_data")
        assert isinstance(cv_data, type(result.cv_data))
        assert all(cv_data.columns == result.cv_data.columns)

        model_reprs = _load_file(partition_label=result.partition, path=tmp_path, expert_type="model_reprs")
        assert isinstance(model_reprs, type(result.model_reprs))
        assert model_reprs == result.model_reprs

        pkl_model = _load_file(partition_label=result.partition, path=tmp_path, expert_type="best_model")
        assert isinstance(pkl_model, type(result.best_model))
        assert str(pkl_model.get_params()) == str(result.best_model.get_params())

        partition = _load_file(partition_label=result.partition, path=tmp_path, expert_type="partition")
        assert isinstance(partition, type(result.partition))
        assert partition == result.partition

        model_selector_result = _load_file(
            partition_label=result.partition,
            path=tmp_path,
            expert_type="model_selector_result",
        )
        assert isinstance(model_selector_result, type(result))
        assert str(model_selector_result.__dict__) == str(result.__dict__)

        # with partition_hash
        pkl_model = _load_file(
            partition_hash=result.partition_hash,
            path=tmp_path,
            expert_type="best_model",
        )
        assert isinstance(pkl_model, type(result.best_model))
        assert str(pkl_model.get_params()) == str(result.best_model.get_params())

        with pytest.raises(ValueError):
            _load_file(
                partition_label=result.partition,
                partition_hash=result.partition_hash,
                path=tmp_path,
                expert_type="best_model",
            )

        with pytest.raises(ValueError):
            _load_file(path=tmp_path, expert_type="best_model")
