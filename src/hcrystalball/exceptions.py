class PredictWithoutFitError(Exception):
    """Raise if wrapper's predict called without fit

    Parameters
    ----------
    msg
    model_name
    """

    def __init__(self, msg=None, model_name=None):
        if msg is None:
            if model_name:
                msg = "Trying to call the method of " + model_name + " without fitting the model first."
            else:
                msg = "Trying to call predict method without fitting the model first."

        super().__init__(msg)


class InsufficientDataLengthError(ValueError):
    """Error to be raised when the input data does not have sufficient rows / observations"""

    pass


class DuplicatedModelNameError(ValueError):
    """Error to be raised if a model name is duplicated (e.g. in an ensemble or CV?)"""

    pass
