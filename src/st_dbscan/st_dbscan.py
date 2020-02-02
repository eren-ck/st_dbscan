# -*- coding: utf-8 -*-
"""
ST-DBSCAN - fast scalable implementation of ST DBSCAN
            scales also to memory by splitting into frames
            and merging the clusters together
"""

# Author: Eren Cakmak <eren.cakmak@uni-konstanz.de>
#
# License: MIT

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.utils import check_array


class ST_DBSCAN():
    """
    A class to perform the ST_DBSCAN clustering
    Parameters
    ----------
    eps1 : float, default=0.5
        The spatial density threshold (maximum spatial distance) between 
        two points to be considered related.
    eps2 : float, default=10
        The temporal threshold (maximum temporal distance) between two 
        points to be considered related.
    min_samples : int, default=5
        The number of samples required for a core point.
    metric : string default='euclidean'
        The used distance metric - more options are
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
    n_jobs : int or None, default=-1
        The number of processes to start -1 means use all processors 
    Attributes
    ----------
    labels : array, shape = [n_samples]
        Cluster labels for the data - noise is defined as -1
    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise".
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Birant, Derya, and Alp Kut. "ST-DBSCAN: An algorithm for clustering spatial–temporal data." Data & Knowledge Engineering 60.1 (2007): 208-221.
    
    Peca, I., Fuchs, G., Vrotsou, K., Andrienko, N. V., & Andrienko, G. L. (2012). Scalable Cluster Analysis of Spatial Events. In EuroVA@ EuroVis.
    """
    def __init__(self,
                 eps1=0.5,
                 eps2=10,
                 min_samples=5,
                 metric='euclidean',
                 n_jobs=-1):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        """
        Apply the ST DBSCAN algorithm 
        ----------
        X : 2D numpy array with
            The first element of the array should be the time 
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        n, m = X.shape

        # Compute sqaured form Euclidean Distance Matrix for 'time' attribute and the spatial attributes
        time_dist = squareform(pdist(X[:, 0].reshape(n, 1),
                                     metric=self.metric))
        euc_dist = squareform(pdist(X[:, 1:], metric=self.metric))

        # filter the euc_dist matrix using the time_dist
        dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)

        db = DBSCAN(eps=self.eps1,
                    min_samples=self.min_samples,
                    metric='precomputed')
        db.fit(dist)

        self.labels = db.labels_

        return self

    def fit_frame_split(self, X, frame_size, frame_overlap=None):
        """
        Apply the ST DBSCAN algorithm with splitting it into frames 
        Merging is still not optimal resulting in minor errors in 
        the overlapping area. In this case the input data has to be 
        sorted for by time. 
        ----------
        X : 2D numpy array with
            The first element of the array should be the time (sorted by time)
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        frame_size : float, default= None
            If not none the dataset is split into frames and merged aferwards
        frame_overlap : float, default=eps2
            If frame_size is set - there will be an overlap between the frames
            to merge the clusters afterwards 
        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        # default values for overlap
        if frame_overlap == None:
            frame_overlap = self.eps2

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        if not frame_size > 0.0 or not frame_overlap > 0.0 or frame_size < frame_overlap:
            raise ValueError(
                'frame_size, frame_overlap not correctly configured.')

        # unique time points
        time = np.unique(X[:, 0])

        labels = None
        right_overlap = 0
        max_label = 0

        for i in range(0, len(time), (frame_size - frame_overlap + 1)):
            for period in [time[i:i + frame_size]]:
                frame = X[np.isin(X[:, 0], period)]
                n, m = frame.shape

                # Compute sqaured form Euclidean Distance Matrix for 'time' attribute and the spatial attributes
                time_dist = squareform(
                    pdist(frame[:, 0].reshape(n, 1), metric=self.metric))
                euc_dist = squareform(pdist(frame[:, 1:], metric=self.metric))

                # filter the euc_dist matrix using the time_dist
                dist = np.where(time_dist <= self.eps2, euc_dist,
                                2 * self.eps1)

                db = DBSCAN(eps=self.eps1,
                            min_samples=self.min_samples,
                            metric='precomputed')
                db.fit(dist)

                # very simple merging - take just right clusters of the right frame
                # Change in future version to a better merging process
                if not type(labels) is np.ndarray:
                    labels = db.labels_
                else:
                    # delete the right overlap
                    labels = labels[0:len(labels) - right_overlap]
                    # change the labels of the new clustering and concat
                    labels = np.concatenate((labels, (db.labels_ + max_label)))

                right_overlap = len(X[np.isin(X[:, 0],
                                              period[-frame_overlap + 1:])])
                max_label = np.max(labels)

        self.labels = labels
        return self
