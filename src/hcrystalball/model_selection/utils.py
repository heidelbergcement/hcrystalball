import json
import pickle
from pathlib import Path
import functools
from collections import OrderedDict
from hcrystalball.utils import generate_partition_hash
import logging

logger = logging.getLogger(__name__)


def _load_file(expert_type="best_model", partition_label=None, partition_hash=None, path=""):
    """Returns unpickled file or json metadata from directory.

    Parameters
    ----------
    expert_type : string
        options for json format
        ['partition','model_reprs','logical_partition','physical_partition',
        'frequency','horizon','best_model_hash','best_model_name','best_model_repr']
        anything else is pickled (e.g. 'best_model', 'cv_data', 'cv_results')

    partition_label : dict
        Model's metadata used to generate hash for .pkl file.
        e.g. {'country':'Canada', 'city':'Vancouver', 'series':'Number of citizens'}

    partition_hash : string
        Partition's hash generated with `generate_partition_hash` function

    path : str
        Path to the directory, where files are to be stored,
        by default '' resulting in current working directory behaviour

    Returns
    -------
        unpickled expert
    """

    if partition_hash is None:
        if partition_label is not None:
            partition_hash = generate_partition_hash(partition_label)
        else:
            raise ValueError("Either one of `partition_label` or `partition_hash` must be set.")
    elif partition_label is not None:
        raise ValueError("Only one of `partition_label` or `partition_hash` must be set. You set both")

    file_path = Path(path, f"{partition_hash}.{expert_type}")

    if expert_type in [
        "partition",
        "model_reprs",
        "logical_partition",
        "physical_partition",
        "frequency",
        "horizon",
        "best_model_hash",
        "best_model_name",
        "best_model_repr",
    ]:
        with open(file_path) as json_file:
            expert = json.load(json_file)
            if expert_type == "partition":
                expert = OrderedDict(sorted(expert.items()))
    else:
        with open(file_path, "rb") as pickle_file:
            expert = pickle.load(pickle_file)

    return expert


load_best_model = functools.partial(_load_file, expert_type="best_model")
load_cv_data = functools.partial(_load_file, expert_type="cv_data")
load_cv_results = functools.partial(_load_file, expert_type="cv_results")
load_partition = functools.partial(_load_file, expert_type="partition")
load_model_reprs = functools.partial(_load_file, expert_type="model_reprs")

load_best_model.__doc__ = _load_file.__doc__
load_cv_data.__doc__ = _load_file.__doc__
load_cv_results.__doc__ = _load_file.__doc__
load_partition.__doc__ = _load_file.__doc__
load_model_reprs.__doc__ = _load_file.__doc__


def _persist_to_file(data, expert_type="best_model", partition_label=None, partition_hash=None, path=""):
    """Returns unpickled file or json metadata from directory.

    Parameters
    ----------
    expert_type : string
        options for json format
        ['partition','model_reprs','logical_partition','physical_partition',
        'frequency','horizon','best_model_hash','best_model_name','best_model_repr']
        anything else is pickled (e.g. 'best_model', 'cv_data', 'cv_results')

    partition_label : dict
        Model's metadata used to generate hash for .pkl file.
        e.g. {'country':'Canada', 'city':'Vancouver', 'series':'Number of citizens'}

    partition_hash : string
        Partition's hash generated with `generate_partition_hash` function

    path : str
        Path to the directory, where files are to be stored,
        by default '' resulting in current working directory behaviour
    """

    if partition_hash is None:
        if partition_label is not None:
            partition_hash = generate_partition_hash(partition_label)
        else:
            raise ValueError("Either one of `partition_label` or `partition_hash` must be set.")
    elif partition_label is not None:
        raise ValueError("Only one of `partition_label` or `partition_hash` must be set. You set both")

    file_path = Path(path, f"{partition_hash}.{expert_type}")

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    if expert_type in [
        "partition",
        "model_reprs",
        "logical_partition",
        "physical_partition",
        "frequency",
        "horizon",
        "best_model_hash",
        "best_model_name",
        "best_model_repr",
    ]:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
    else:
        with open(file_path, "wb") as pickle_file:
            pickle.dump(data, pickle_file)


def persist_experts_in_physical_partition(
    folder_path="results",
    results=None,
    persist_cv_results=False,
    persist_cv_data=False,
    persist_model_reprs=False,
    persist_best_model=False,
    persist_partition=False,
    persist_model_selector_results=True,
):
    """Store expert files for each partition.

    The file names follow {partition_hash}.{expert_type} e.g. 795dab1813f05b1abe9ae6ded93e1ec4.cv_data

    Parameters
    ----------
    results : list of ModelSelectorResults
        results of model selection for each partition

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

    Returns
    -------
    str
        Folder path where experts were stored
    """
    if results is not None:
        for result in results:
            if persist_best_model:
                result.persist(attribute_name="best_model", path=folder_path)
            if persist_partition:
                result.persist(attribute_name="partition", path=folder_path)
            if persist_model_selector_results:
                result.persist(path=folder_path)
            if persist_cv_results:
                result.persist(attribute_name="cv_results", path=folder_path)
            if persist_cv_data:
                result.persist(attribute_name="cv_data", path=folder_path)
            if persist_model_reprs:
                result.persist(attribute_name="model_reprs", path=folder_path)
    else:
        logging.info("You passed empty results. Nothing is being persisted.")

    return folder_path
