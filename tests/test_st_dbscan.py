from importlib.metadata import version

import numpy as np

from st_dbscan import ST_DBSCAN, __version__


def test_package_version_is_consistent():
    assert __version__ == version("st-dbscan")
    assert not __version__.startswith("0.0")
    assert __version__ != "unknown"


def test_fit_clusters_spatially_and_temporally():
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.1],
            [2.0, 0.2, 0.2],
            [20.0, 10.0, 10.0],
        ]
    )

    result = ST_DBSCAN(eps1=0.5, eps2=5.0, min_samples=2).fit(data)

    np.testing.assert_array_equal(result.labels, [0, 0, 0, -1])


def test_fit_frame_split_clusters_without_losing_samples():
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.1],
            [2.0, 0.2, 0.2],
            [3.0, 0.3, 0.3],
            [10.0, 5.0, 5.0],
            [11.0, 5.1, 5.1],
        ]
    )

    result = ST_DBSCAN(
        eps1=0.5,
        eps2=2.0,
        min_samples=2,
    ).fit_frame_split(data, frame_size=4, frame_overlap=2)

    assert len(result.labels) == len(data)
