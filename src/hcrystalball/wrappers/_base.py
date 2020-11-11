import inspect
from types import FunctionType
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator


def get_clean_param_dict(signature):
    """Provide function signature withouth self, * and **.

    Create a dictionary of function parameters from a function
    signature object, omitting 'self' and */** parameters

    Parameters
    ----------
    signature: inspect.Signature
        Signature of function or method

    Returns
    -------
    dict
        Parameters and their defaults in form of {parameter_name: default_value}
    """
    return {
        p.name: p.default if p.default != inspect.Parameter.empty else None
        for p in signature.parameters.values()
        if p.name != "self" and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL
    }


def tsmodel_wrapper_constructor_factory(modeltype):
    """Bring original and modeltype class arguments under one constructor

    This parametrized decorator joins the signatures of the constructors
    of two classes into one constructor. It must only be used on the __init__ function
    of a class. It is intended to join the parameters of a specific model
    with the parameters of its wrapper in a way that is compliant with scikit-learn,
    enabling the use of get_params()/set_params() on the wrapper
    as if one was directly interacting with the model being wrapped.

    Parameters
    ----------
    modeltype : class
        Model class providing the constructor to be joined with the target
        constructor of the inner decorator

    Returns
    -------
    callable
        inner decorator applied to the target constructor
    """

    def tsmodel_wrapper_constructor(init_func):
        """Bring original and modeltype class arguments under one constructor

        This decorator picks up the constructor of the class supplied to
        the outer decorator and performs the join with the target constructor.
        building a new constructor from scratch using string compilation
        (https://docs.python.org/3/library/functions.html#compile)

        Parameters
        ----------
        init_func : callable
            The target constructor to be decorated

        Returns
        -------
        callable
            New constructor which accepts both the original arguments and agrumets from the 'modeltype' class
        """
        orig_signature = inspect.signature(init_func)
        orig_parameters = get_clean_param_dict(orig_signature)

        model_signature = inspect.signature(modeltype.__init__)
        model_parameters = get_clean_param_dict(model_signature)

        full_parameter_names = list(model_parameters) + list(orig_parameters)
        full_parameter_defaults = list(model_parameters.values()) + list(orig_parameters.values())
        assignments = "; ".join([f"self.{p}={p}" for p in full_parameter_names])

        constructor_code = compile(
            f'def __init__(self, {", ".join(full_parameter_names)}): ' f"{assignments}",
            "<string>",
            "exec",
        )
        modified_init_func = FunctionType(
            constructor_code.co_consts[0],
            globals(),
            "__init__",
            tuple(full_parameter_defaults),
        )
        return modified_init_func

    return tsmodel_wrapper_constructor


class TSModelWrapper(BaseEstimator, metaclass=ABCMeta):
    """Base class for all model wrappers in hcrystalball"""

    @abstractmethod
    def __init__(self):
        pass

    def _init_tsmodel(self, model_cls, **extra_args):
        """Initialiaze `model_cls`.

        The inner model with model default parameters plus parameters
        provided during initialization of the model wrapper"

        Parameters
        ----------
        model_cls : class
            Model class

        Returns
        -------
        Any
            instance of `model_cls`
        """
        model_signature = inspect.signature(model_cls.__init__)
        model_params = get_clean_param_dict(model_signature)
        params = {k: v for k, v in self.get_params().items() if k in model_params}
        return self._set_model_extra_params(model_cls(**{**params, **extra_args}))

    def _set_model_extra_params(self, model):
        return model

    @staticmethod
    def _transform_data_to_tsmodel_input_format(self, X, y=None):
        """Placeholder method for child classes

        Each class to develop model wrapper to transform X and y to
        model required format.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.

        y : array_like, (1d)
            Target vector.

        Returns
        -------
        X, y
            X - pandas.DataFrame with features
            y - array_like or None for target
        """
        return X, y

    def _clip_predictions(self, preds):
        """Clip provided predictions between `clip_predictions_lower` and `clip_predictions_upper`

        Parameters
        ----------
        preds : pandas.DataFrame
            Predictions

        Returns
        -------
        pandas.DataFrame
            Clipped predictions.
        """
        preds[self.name] = preds[self.name].clip(
            lower=self.clip_predictions_lower, upper=self.clip_predictions_upper
        )
        return preds


__all__ = ["TSModelWrapper"]
