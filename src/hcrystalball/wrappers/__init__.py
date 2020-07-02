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

    ProphetWrapper()


try:
    from ._sarimax import SarimaxWrapper

    __all__ += ["SarimaxWrapper"]
except Exception:

    class SarimaxWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._sarimax import SarimaxWrapper

    SarimaxWrapper()


try:
    from ._smoothing import ExponentialSmoothingWrapper

    __all__ += ["ExponentialSmoothingWrapper"]
except Exception:

    class ExponentialSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._smoothing import ExponentialSmoothingWrapper

    ExponentialSmoothingWrapper()

try:
    from ._smoothing import SimpleSmoothingWrapper

    __all__ += ["SimpleSmoothingWrapper"]
except Exception:

    class SimpleSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._smoothing import SimpleSmoothingWrapper

    SimpleSmoothingWrapper()

try:
    from ._smoothing import HoltSmoothingWrapper

    __all__ += ["HoltSmoothingWrapper"]
except Exception:

    class HoltSmoothingWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._smoothing import HoltSmoothingWrapper

    HoltSmoothingWrapper()


try:
    from ._tbats import TBATSWrapper

    __all__ += ["TBATSWrapper"]
except Exception:

    class TBATSWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._tbats import TBATSWrapper

    TBATSWrapper()

try:
    from ._tbats import BATSWrapper

    __all__ += ["BATSWrapper"]
except Exception:

    class BATSWrapper:
        """This is just helper class to inform user about missing dependencies at init time"""

        def __init__(self):
            # this always fails
            from ._tbats import BATSWrapper

    BATSWrapper()
