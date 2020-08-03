from pkg_resources import get_distribution
from pkg_resources import DistributionNotFound
import logging

try:
    # Change here if project is renamed and does not equal the package name
    __version__ = get_distribution(__name__).version
except Exception:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
