from hcrystalball.model_selection.utils import persist_experts_in_physical_partition
from hcrystalball.model_selection.utils import _load_file
from hcrystalball.model_selection.utils import store_result_to_self
from hcrystalball.model_selection import select_model
import pickle
import json
import os
import pytest
import inspect


@pytest.mark.parametrize("train_data, grid_search", [("", "")], indirect=["train_data", "grid_search"])
def test_persist_experts_in_physical_partition(train_data, grid_search, tmp_path):

    partition_columns = ["Product"]
    results = select_model(
        train_data, target_col_name="Quantity", partition_columns=partition_columns, grid_search=grid_search
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

        with open(os.path.join(tmp_path, result.partition_hash + ".model_selector_result"), "rb") as file:
            model_selector_result = pickle.load(file)
            assert isinstance(model_selector_result, type(result))
            assert str(model_selector_result.__dict__) == str(result.__dict__)


@pytest.mark.parametrize("train_data, grid_search", [("", "")], indirect=["train_data", "grid_search"])
def test_load_expert(train_data, grid_search, tmp_path):

    partition_columns = ["Product"]
    results = select_model(
        train_data, target_col_name="Quantity", partition_columns=partition_columns, grid_search=grid_search
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
            partition_label=result.partition, path=tmp_path, expert_type="model_selector_result"
        )
        assert isinstance(model_selector_result, type(result))
        assert str(model_selector_result.__dict__) == str(result.__dict__)

        # with partition_hash
        pkl_model = _load_file(partition_hash=result.partition_hash, path=tmp_path, expert_type="best_model")
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


####################################################################################
######################## Test store_result_to_self #################################
####################################################################################
def my_sum(a, b, c=5):
    """Return sum of values"""
    return a + b + c


def double(number):
    """Return double"""
    return number * 2


def as_str(number):

    return str(number)


class MyObject:
    def __init__(self, a, b, c=None):

        self.a, self.b = a, b
        if c is not None:
            self.c = c
        self.my_sum = store_result_to_self(self, my_sum, "result", self_args=None, return_result=False)
        self.my_sum_a = store_result_to_self(
            self, my_sum, "result", self_args={"a": "a"}, return_result=False
        )
        self.my_sum_a_b = store_result_to_self(
            self, my_sum, "result", self_args={"a": "a", "b": "b"}, return_result=False
        )
        self.my_sum_a_b_c = store_result_to_self(
            self, my_sum, "result", self_args={"a": "a", "b": "b", "c": "c"}, return_result=False
        )
        # to ensure properties work as well
        self.double_my_result = store_result_to_self(
            self, double, "double_result", self_args={"number": "result"}, return_result=False
        )
        # one needs to pass _double_result in case of using properties
        self.double_my_result_str = store_result_to_self(
            self, as_str, None, self_args={"number": "_double_result"}, return_result=True
        )
        self._double_result = None

    @property
    def double_result(self):

        if self._double_result is None:
            raise ValueError("You need to run `double_my_result` first to obtain the `double_result`!")
        return self._double_result

    @double_result.setter
    def double_result(self, value):

        self._double_result = value


def test_store_result_to_self():

    a, b, c = 1, 2, 3
    func_default_c = 5
    result = 6

    obj = MyObject(a, b, c)

    # ensure docstrings and code of methods are the same as wrapped functions have
    assert obj.my_sum_a_b_c.__doc__ == my_sum.__doc__
    assert inspect.getsource(obj.my_sum_a_b_c) == inspect.getsource(my_sum)

    # result and double_result does not exist yet
    assert not hasattr(obj, "result")
    with pytest.raises(ValueError):
        print(obj.double_result)

    # ensure priorities of users parameters > object attributes values > wrapped function defaults
    # ensure when user supplies parameters, object attributes are updated

    # no self arguments used and expected function result stored to obj.result
    obj.my_sum(a, b, c)
    assert result == obj.result
    assert (a, b, c) == (obj.a, obj.b, obj.c)

    # b is not in self_args, thus results is bigger, but obj.b does not change
    obj.my_sum_a(b=b + 10, c=c)
    assert result + 10 == obj.result
    assert (a, b, c) == (obj.a, obj.b, obj.c)
    # a is in self_args, thus results is bigger and obj.a not changes to a + 10
    obj.my_sum_a(a=a + 10, b=b, c=c)
    assert result + 10 == obj.result
    assert (a + 10, b, c) == (obj.a, obj.b, obj.c)

    # a was changed in the past already plus we increase c, thus result + 10 + 100
    # obj.a == 10 and obj.c == c (as not in self_args)
    obj.my_sum_a_b(c=c + 100)
    assert result + 10 + 100 == obj.result
    assert (a + 10, b, c) == (obj.a, obj.b, obj.c)

    # used default function parameter c=5 when c not passed from user, nor from self_args
    obj_no_c = MyObject(a, b)
    obj_no_c.my_sum_a_b_c()
    assert a + b + func_default_c == obj_no_c.result
    assert (a, b) == (obj_no_c.a, obj_no_c.b)

    # ensure passing properties as self parameters works as well as returning value from method
    obj.double_my_result()
    assert obj.double_result == obj.result * 2
    assert obj.double_result == (result + 10 + 100) * 2
    assert obj.double_my_result_str() == str(obj.double_result)
