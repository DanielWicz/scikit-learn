# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
_kmeans_plusplus.pyx
--------------------

Cythonised version of the k-means++ seeding algorithm.

This file is a direct translation of the original Python/NumPy routine found in
``sklearn/cluster/_k_means.py``.  It keeps the public signature intact while
executing the tight loops at C-speed.

Only scikit-learn-approved dependencies are used (`numpy`, `scipy.sparse`,
internal helpers such as ``_euclidean_distances`` and ``stable_cumsum``).
"""

import numpy as np
cimport numpy as np
from cython cimport floating
import cython
from libc.math cimport log as clog
from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.utils.extmath import stable_cumsum
from scipy import sparse as sp


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple _kmeans_plusplus(
        object X,                                         # ndarray or CSR/CSC
        int n_clusters,
        np.ndarray[floating, ndim=1] x_squared_norms,
        np.ndarray[floating, ndim=1] sample_weight,
        object random_state,
        int n_local_trials = -1) except *:
    """
    Fast Arthur & Vassilvitskii k-means++ initialisation.

    Parameters
    ----------
    (Same as the pure-Python version.)

    Returns
    -------
    centers : ndarray (n_clusters, n_features)
    indices : ndarray (n_clusters,)
    """

    # ------------------------------------------------------------------
    # Shapes and dtypes
    cdef Py_ssize_t n_samples  = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef np.dtype dtype_X      = X.dtype               # works for sparse/dense

    # ------------------------------------------------------------------
    # Allocate outputs
    cdef np.ndarray[floating, ndim=2] centers = np.empty(
        (n_clusters, n_features), dtype=dtype_X)
    cdef np.ndarray[np.intp_t, ndim=1] indices = np.full(
        n_clusters, -1, dtype=np.intp)

    # ------------------------------------------------------------------
    # Lazy default for `n_local_trials`
    if n_local_trials < 0:
        n_local_trials = 2 + int(clog(n_clusters))

    # ------------------------------------------------------------------
    # 1. Pick first centre
    cdef int center_id = int(
        random_state.choice(n_samples,
                            p=sample_weight / sample_weight.sum())
    )
    if sp.issparse(X):
        centers[0] = X[[center_id]].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # ------------------------------------------------------------------
    # 2. Initial potentials
    cdef np.ndarray[floating, ndim=1] closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis],
        X,
        Y_norm_squared=x_squared_norms,
        squared=True,
    ).ravel()

    cdef floating current_pot = np.dot(closest_dist_sq, sample_weight)

    # ------------------------------------------------------------------
    # 3. Greedy addition of the remaining centres
    cdef int c, best_idx, best_candidate
    cdef np.ndarray[floating, ndim=2] distance_to_candidates
    cdef np.ndarray[floating, ndim=2] candidates_pot
    cdef np.ndarray[np.intp_t,   ndim=1] candidate_ids
    cdef np.ndarray[floating, ndim=1] rand_vals

    for c in range(1, n_clusters):
        # 3.a Draw candidate indices with probability ∝ distance²
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            stable_cumsum(sample_weight * closest_dist_sq),
            rand_vals
        )
        np.clip(candidate_ids, None, n_samples - 1, out=candidate_ids)

        # 3.b Distances from each candidate to all points
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids],
            X,
            Y_norm_squared=x_squared_norms,
            squared=True,
        )

        # 3.c Updated potentials for the candidates
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # 3.d Best candidate = argmin potential  (--- declarations were moved
        #                                        --- to the top of the loop)
        best_idx       = <int> np.argmin(candidates_pot)
        best_candidate = <int> candidate_ids[best_idx]
        current_pot    = <floating> candidates_pot[best_idx, 0]
        closest_dist_sq = distance_to_candidates[best_idx]

        # 3.e Commit best candidate
        if sp.issparse(X):
            centers[c] = X[[best_candidate]].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices

