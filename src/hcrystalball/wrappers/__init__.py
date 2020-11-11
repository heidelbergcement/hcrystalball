from ._sklearn import get_sklearn_wrapper as get_sklearn_wrapper
from sklearn import set_config
from hcrystalball.utils import optional_import

set_config(print_changed_only=False)

__all__ = ["get_sklearn_wrapper"]

__all__.extend(optional_import("hcrystalball.wrappers._prophet", "ProphetWrapper", globals()))
__all__.extend(
    optional_import("hcrystalball.wrappers._statsmodels", "ExponentialSmoothingWrapper", globals())
)
__all__.extend(optional_import("hcrystalball.wrappers._statsmodels", "SimpleSmoothingWrapper", globals()))
__all__.extend(optional_import("hcrystalball.wrappers._statsmodels", "HoltSmoothingWrapper", globals()))
__all__.extend(optional_import("hcrystalball.wrappers._statsmodels", "ThetaWrapper", globals()))
__all__.extend(optional_import("hcrystalball.wrappers._sarimax", "SarimaxWrapper", globals()))
__all__.extend(optional_import("hcrystalball.wrappers._tbats", "TBATSWrapper", globals()))
__all__.extend(optional_import("hcrystalball.wrappers._tbats", "BATSWrapper", globals()))
