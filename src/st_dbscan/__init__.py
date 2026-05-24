# -*- coding: utf-8 -*-
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = 'unknown'
finally:
    del PackageNotFoundError, version

from .st_dbscan import ST_DBSCAN  # noqa: E402
