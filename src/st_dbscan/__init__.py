from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("st-dbscan")
except PackageNotFoundError:
    __version__ = "unknown"

from .st_dbscan import ST_DBSCAN

__all__ = ["ST_DBSCAN", "__version__"]
