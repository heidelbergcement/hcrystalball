from ._sklearn import get_sklearn_wrapper as get_sklearn_wrapper
from sklearn import set_config

set_config(print_changed_only=False)

__all__ = ["get_sklearn_wrapper"]

# Following lines allow for lazy optinal imports with original ModuleNotFoundErrors
try:
    from ._prophet import ProphetWrapper

    __all__ += ["ProphetWrapper"]
except Exception:

    class ProphetWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._prophet import ProphetWrapper


try:
    from ._sarimax import SarimaxWrapper

    __all__ += ["SarimaxWrapper"]
except Exception:

    class SarimaxWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._sarimax import SarimaxWrapper


try:
    from ._smoothing import ExponentialSmoothingWrapper, SimpleSmoothingWrapper, HoltSmoothingWrapper

    __all__ += ["ExponentialSmoothingWrapper", "SimpleSmoothingWrapper", "HoltSmoothingWrapper"]
except Exception:

    class ExponentialSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._smoothing import ExponentialSmoothingWrapper

    class SimpleSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._smoothing import SimpleSmoothingWrapper

    class HoltSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._smoothing import HoltSmoothingWrapper


try:
    from ._tbats import TBATSWrapper, BATSWrapper

    __all__ += ["TBATSWrapper", "BATSWrapper"]
except Exception:

    class TBATSWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._tbats import TBATSWrapper

    class BATSWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._tbats import BATSWrapper
