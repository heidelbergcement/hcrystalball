from importlib import import_module
from ._sklearn import get_sklearn_wrapper as get_sklearn_wrapper

from sklearn import set_config

set_config(print_changed_only=False)

__all__ = ["get_sklearn_wrapper"]

# Following lines allow for lazy optinal imports with original ModuleNotFoundErrors
try:
    from ._prophet import ProphetWrapper

    __all__ += ["ProphetWrapper"]
except:

    class ProphetWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from . import _prophet


try:
    from ._sarimax import SarimaxWrapper

    __all__ += ["SarimaxWrapper"]
except:

    class SarimaxWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from . import _sarimax


try:
    from ._smoothing import ExponentialSmoothingWrapper, SimpleSmoothingWrapper, HoltSmoothingWrapper

    __all__ += ["ExponentialSmoothingWrapper", "SimpleSmoothingWrapper", "HoltSmoothingWrapper"]
except:

    class ExponentialSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from . import _smoothing

    class SimpleSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from . import _smoothing

    class HoltSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from . import _smoothing


try:
    from ._tbats import TBATSWrapper, BATSWrapper

    __all__ += ["TBATSWrapper", "BATSWrapper"]
except:

    class TBATSWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from . import _tbats

    class BATSWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from . import _tbats

