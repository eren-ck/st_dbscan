import numpy as np
import pytest

from st_dbscan import ST_DBSCAN


@pytest.mark.parametrize(
    'points',
    [
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0.5, 0, 0]],
        [[0, 0, 0], [0, 0.5, 0]],
    ],
    ids=['identical-points', 'identical-locations', 'identical-times'],
)
@pytest.mark.parametrize('metric', ['euclidean', 'cityblock'])
def test_sparse_fit_preserves_zero_distance_neighbors(points, metric):
    points = np.asarray(points, dtype=float)
    dense = ST_DBSCAN(eps1=1, eps2=1, min_samples=2, metric=metric).fit(points)
    # Trigger the sparse path without introducing additional neighbors.
    padding = np.column_stack([
        np.arange(1, 19999) * 10,
        np.arange(1, 19999) * 10,
        np.zeros(19998),
    ])
    sparse = ST_DBSCAN(eps1=1, eps2=1, min_samples=2, metric=metric).fit(
        np.vstack([points, padding])
    )

    np.testing.assert_array_equal(sparse.labels[:len(points)], dense.labels)
    assert np.all(sparse.labels[len(points):] == -1)


@pytest.mark.parametrize(
    'points',
    [
        [[0, 0, 0], [0.5, 0, 0], [10, 0, 0], [10.5, 0, 0]],
        [[0, 0, 0], [0.5, 0, 0], [0, 10, 0], [0.5, 10, 0]],
    ],
    ids=['temporally-distant', 'spatially-distant'],
)
def test_sparse_fit_keeps_clusters_separate(points):
    points = np.asarray(points, dtype=float)
    dense = ST_DBSCAN(eps1=1, eps2=1, min_samples=2).fit(points)
    padding = np.column_stack([
        np.arange(1, 19997) * 10 + 100,
        np.arange(1, 19997) * 10,
        np.zeros(19996),
    ])
    sparse = ST_DBSCAN(eps1=1, eps2=1, min_samples=2).fit(
        np.vstack([points, padding])
    )

    np.testing.assert_array_equal(sparse.labels[:len(points)], dense.labels)
    assert np.all(sparse.labels[len(points):] == -1)


def test_sparse_fit_includes_neighbors_on_both_thresholds():
    points = np.asarray([[0, 0, 0], [1, 1, 0]], dtype=float)
    padding = np.column_stack([
        np.arange(1, 19999) * 10,
        np.arange(1, 19999) * 10,
        np.zeros(19998),
    ])
    sparse = ST_DBSCAN(eps1=1, eps2=1, min_samples=2).fit(
        np.vstack([points, padding])
    )

    np.testing.assert_array_equal(sparse.labels[:len(points)], [0, 0])
    assert np.all(sparse.labels[len(points):] == -1)
