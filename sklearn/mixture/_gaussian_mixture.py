"""Gaussian Mixture Model."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import linalg
import os
from functools import lru_cache, partial
from ..utils import check_array
import psutil
try:
    import jax.numpy as jnp
except:
    pass

from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class


@lru_cache(maxsize=1)
def _preferred_batch_cholesky_backend():
    """
    Decide the best batched-Cholesky backend and cache the result:

    * **jax**   – fastest when JAX is available (CPU or GPU).
    * **scipy** – SciPy ≥ 1.9 supports batched `linalg.cholesky`.
    * **numpy** – always available, reasonably fast.
    * **loop**  – last-resort pure-Python loop.
    """

    # ── ➋ SciPy ≥ 1.9 ────────────────────────────────────────────────────
    try:
        dummy = np.eye(2)[None, ...]
        linalg.cholesky(dummy, lower=True)
        return "scipy"
    except ValueError as exc:     # SciPy < 1.9 → “needs to be 2-D array”
        if "needs to be" not in str(exc):
            return "scipy"        # SciPy present but dummy not SPD
    except Exception:
        pass
        
        
    # ── ➌ NumPy ──────────────────────────────────────────────────────────
    try:
        dummy = np.eye(2)[None, ...]
        np.linalg.cholesky(dummy)
        return "numpy"
    except Exception:
        pass





    # ── ➊ JAX ─────────────────────────────────────────────────────────────
    try:
        import jax.numpy as jnp
        dummy = jnp.eye(2).reshape(1, 2, 2)
        _ = jnp.linalg.cholesky(dummy)          # triggers compilation once
        return "jax"
    except Exception:
        pass



    # ── ➍ Fallback loop ─────────────────────────────────────────────────
    return "loop"


@lru_cache(maxsize=1)
def _available_ram():
    """Return currently available RAM in MB (0 if ``psutil`` missing)."""
    return psutil.virtual_memory().total >> 20

@lru_cache(maxsize=1)
def _select_vectorised(n_samples: int, n_components: int, n_features: int,
                       dtype=np.float64, safety: float = 0.5) -> bool:
    """Decide whether the huge einsum tensor fits in RAM."""
    bytes_needed = (n_samples * n_components * n_features *
                    np.dtype(dtype).itemsize)
    return (bytes_needed/(1024**2)) < _available_ram() * safety

@lru_cache(maxsize=1)
def _fits_in_memory(n_bytes, safety=0.5):
    """
    Heuristic test whether allocating ``n_bytes`` is safe based on cached available RAM.

    Parameters
    ----------
    n_bytes : int
        Candidate allocation size in **bytes**.
    safety : float, default=0.5
        Fraction of the free memory to leave untouched (e.g. 0.5 = use up to 50% of free RAM).

    Returns
    -------
    bool
        True if allocation is safe within the cached available memory.
    """
    avail_mb = _available_ram()
    if not avail_mb:
        # Either psutil is missing or we couldn’t query; bail out conservatively.
        return False
    # Convert bytes to MiB and compare against the allowed fraction of free RAM
    return (n_bytes / (1024 ** 2)) < safety * avail_mb


@lru_cache(maxsize=128)
def _optimal_chunk_size(
    n_samples: int,
    n_features: int,
    dtype: np.dtype = np.float64,
    safety: float = 0.5,
) -> int:
    """
    Return the largest number of mixture *components* that can be processed in
    a single vectorised pass without exceeding the available RAM budget.

    The chosen ``chunk`` satisfies::

        n_samples · chunk · n_features · dtype.itemsize  <  safety · free-RAM

    The function never returns a value < 1.
    """
    bytes_per_elem = np.dtype(dtype).itemsize
    free_bytes = _available_ram() * 1024 ** 2 * safety
    if free_bytes <= 0:                # psutil unavailable or unknown
        return 1
    max_k = int(free_bytes // (n_samples * n_features * bytes_per_elem))
    return max(1, max_k)



def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), "weights")

    # check range
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f"
            % (np.min(weights), np.max(weights))
        )

    # check normalization
    atol = 1e-6 if weights.dtype == np.float32 else 1e-8
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0, atol=atol):
        raise ValueError(
            "The parameter 'weights' should be normalized, but got sum(weights) = %.5f"
            % np.sum(weights)
        )
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), "means")
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (
        np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)
    ):
        raise ValueError(
            "'%s precision' should be symmetric, positive-definite" % covariance_type
        )


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : str

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(
        precisions,
        dtype=[np.float64, np.float32],
        ensure_2d=False,
        allow_nd=covariance_type == "full",
    )

    precisions_shape = {
        "full": (n_components, n_features, n_features),
        "tied": (n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
    }
    _check_shape(
        precisions, precisions_shape[covariance_type], "%s precision" % covariance_type
    )

    _check_precisions = {
        "full": _check_precisions_full,
        "tied": _check_precision_matrix,
        "diag": _check_precision_positivity,
        "spherical": _check_precision_positivity,
    }
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)
def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """
    Estimate full covariance matrices with minimal memory copies,
    using batched np.matmul (multi-threaded BLAS).

    We precompute 3 broadcast views:
      X_b     = X[:, None, :]       # (n_samples, 1,  n_features)
      means_b = means[None, :, :]   # (1,       n_components, n_features)
      resp_b  = resp[:, :, None]    # (n_samples, n_components, 1)

    Then diff_b   = X_b - means_b  # view broadcast subtraction
        weighted = diff_b * resp_b  # elementwise weighting
    Finally cov = matmul(diff_b.T @ weighted)
    """
    n_components, n_features = means.shape
    n_samples = X.shape[0]
    dtype = X.dtype

    # broadcast views
    X_b     = X[:, None, :]       # (n, 1,  d)
    means_b = means[None, :, :]   # (1, k,  d)
    resp_b  = resp[:, :, None]    # (n, k,  1)

    # 1) tiny: pure Python loop
    LOOP_MAX_K = 4
    if n_components <= LOOP_MAX_K and not _select_vectorised(
        n_samples, n_components, n_features, dtype
    ):
        cov = np.empty((n_components, n_features, n_features), dtype=dtype)
        for k in range(n_components):
            diff_k   = X - means[k]            # (n, d) – one small broadcast
            weighted = diff_k * resp[:, k][:, None]
            c_k      = diff_k.T @ weighted     # BLAS-backed
            c_k.flat[:: n_features + 1] += reg_covar
            cov[k]   = c_k
        return cov

    # compute total bytes of the full diff array
    total_bytes = n_samples * n_components * n_features * dtype.itemsize

    # 2) medium: all-in-one batched matmul
    if _fits_in_memory(total_bytes):
        # one broadcast diff
        diff_b   = X_b - means_b               # view, no copy of shape (n, k, d)
        weighted = diff_b * resp_b            # (n, k, d)

        # matmul: (k, d, n) @ (k, n, d) → (k, d, d)
        # we only transpose the small precision dimension
        cov = np.matmul(
            diff_b.transpose(1, 2, 0),        # (k, d, n)
            weighted.transpose(1, 0, 2)       # (k, n, d)
        )
        cov /= nk[:, None, None]
        cov[:, np.arange(n_features), np.arange(n_features)] += reg_covar
        return cov

    # 3) large: chunk into fewest RAM-safe blocks
    chunk_size = _optimal_chunk_size(n_samples, n_features, dtype)
    cov_chunks = []
    for start in range(0, n_components, chunk_size):
        end     = min(start + chunk_size, n_components)
        # slice views – still no copies yet
        diff_chunk   = X_b - means_b[:, start:end, :]    # (n, k_c, d)
        resp_chunk   = resp_b[:, start:end, :]           # (n, k_c, 1)
        weighted_ch  = diff_chunk * resp_chunk           # (n, k_c, d)

        cov_ch = np.matmul(
            diff_chunk.transpose(1, 2, 0),              # (k_c, d, n)
            weighted_ch.transpose(1, 0, 2)               # (k_c, n, d)
        )
        cov_ch /= nk[start:end, None, None]
        cov_ch[:, np.arange(n_features), np.arange(n_features)] += reg_covar
        cov_chunks.append(cov_ch)

    return np.concatenate(cov_chunks, axis=0)


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    return avg_X2 - avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


###############################################################################
# Robust, version-agnostic batched precision-Cholesky
###############################################################################

def _compute_precision_cholesky(covariances, covariance_type, *, _tiny_kd=32):
    """
    Compute upper-triangular precision-Cholesky factors.

    A single backend choice (cached by
    ``_preferred_batch_cholesky_backend``) means the EM loop incurs **zero
    overhead** from backend detection after the first call.
    """
    dtype = covariances.dtype
    err_msg = (
        "Fitting failed: ill-defined empirical covariance. "
        "Try fewer components, larger reg_covar, or float64 inputs."
    )

    # ------------------------------------------------------------------ #
    # Fast paths for non-'full' cases (unchanged)                        #
    # ------------------------------------------------------------------ #
    if covariance_type != "full":
        if covariance_type == "tied":
            try:
                L = linalg.cholesky(covariances, lower=True)
            except linalg.LinAlgError:
                raise ValueError(err_msg)
            return linalg.solve_triangular(
                L, np.eye(L.shape[0], dtype=dtype), lower=True
            ).T
        else:  # diag / spherical
            if np.any(covariances <= 0.0):
                raise ValueError(err_msg)
            return 1.0 / np.sqrt(covariances)

    # ------------------------------------------------------------------ #
    # FULL covariance – backend-dependent path                           #
    # ------------------------------------------------------------------ #
    n_components, n_features, _ = covariances.shape

    # ❶ tiny problems – keep the simple Python loop
    if n_components * n_features <= _tiny_kd:
        chol = []
        for k in range(n_components):
            try:
                L = linalg.cholesky(covariances[k], lower=True)
            except linalg.LinAlgError:
                raise ValueError(err_msg)
            U = linalg.solve_triangular(
                L, np.eye(n_features, dtype=dtype), lower=True
            ).T
            chol.append(U)
        return np.stack(chol, axis=0)

    # ❷ large batch – use the cached fastest backend
    backend = _preferred_batch_cholesky_backend()

    # ── SciPy batched ──────────────────────────────────────────────────
    if backend == "scipy":
        L = linalg.cholesky(covariances, lower=True)
        U = np.linalg.inv(L).transpose(0, 2, 1)
        return U

    # ── JAX batched ────────────────────────────────────────────────────
    if backend == "jax":
        cov_jax = jnp.asarray(covariances)
        L = jnp.linalg.cholesky(cov_jax, lower=True)
        U = jnp.linalg.inv(L).transpose(0, 2, 1)
        return np.asarray(U, dtype=dtype)

    # ── NumPy batched ─────────────────────────────────────────────────
    if backend == "numpy":
        U_upper = np.linalg.cholesky(covariances)
        return np.linalg.inv(U_upper)

    # ❸ ultimate fallback – slow Python loop (rarely hit)
    chol = []
    for k in range(n_components):
        L = linalg.cholesky(covariances[k], lower=True)
        U = linalg.solve_triangular(
            L, np.eye(n_features, dtype=dtype), lower=True
        ).T
        chol.append(U)
    return np.stack(chol, axis=0)






def _flipudlr(array):
    """Reverse the rows and columns of an array."""
    return np.flipud(np.fliplr(array))


def _compute_precision_cholesky_from_precisions(precisions, covariance_type):
    r"""Compute the Cholesky decomposition of precisions using precisions themselves.

    As implemented in :func:`_compute_precision_cholesky`, the `precisions_cholesky_` is
    an upper-triangular matrix for each Gaussian component, which can be expressed as
    the $UU^T$ factorization of the precision matrix for each Gaussian component, where
    $U$ is an upper-triangular matrix.

    In order to use the Cholesky decomposition to get $UU^T$, the precision matrix
    $\Lambda$ needs to be permutated such that its rows and columns are reversed, which
    can be done by applying a similarity transformation with an exchange matrix $J$,
    where the 1 elements reside on the anti-diagonal and all other elements are 0. In
    particular, the Cholesky decomposition of the transformed precision matrix is
    $J\Lambda J=LL^T$, where $L$ is a lower-triangular matrix. Because $\Lambda=UU^T$
    and $J=J^{-1}=J^T$, the `precisions_cholesky_` for each Gaussian component can be
    expressed as $JLJ$.

    Refer to #26415 for details.

    Parameters
    ----------
    precisions : array-like
        The precision matrix of the current components.
        The shape depends on the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends on the covariance_type.
    """
    if covariance_type == "full":
        precisions_cholesky = np.array(
            [
                _flipudlr(linalg.cholesky(_flipudlr(precision), lower=True))
                for precision in precisions
            ]
        )
    elif covariance_type == "tied":
        precisions_cholesky = _flipudlr(
            linalg.cholesky(_flipudlr(precisions), lower=True)
        )
    else:
        precisions_cholesky = np.sqrt(precisions)
    return precisions_cholesky


###############################################################################
# Faster (and simpler) log-det computation
###############################################################################
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Return log-|precision_chol| for each component.

    Parameters
    ----------
    matrix_chol : ndarray
        Cholesky factors – see original doc-string.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int

    Returns
    -------
    log_det_precision_chol : ndarray or scalar
        * shape (n_components,) for 'full', 'diag', 'spherical'
        * scalar for 'tied'  (broadcasted by the caller)
    """
    if covariance_type == "full":
        # diagonal view – shape (k, d), no data copy
        diag = np.diagonal(matrix_chol, axis1=-2, axis2=-1)
        log_det_chol = np.log(diag).sum(axis=1)            # (k,)

    elif covariance_type == "tied":
        # scalar; the caller relies on numpy broadcasting
        log_det_chol = np.log(np.diagonal(matrix_chol)).sum()

    elif covariance_type == "diag":
        log_det_chol = np.log(matrix_chol).sum(axis=1)      # (k,)

    else:  # 'spherical'
        log_det_chol = n_features * np.log(matrix_chol)     # (k,)

    return log_det_chol


###############################################################################
# New fast implementation of the log-probability helper
# Place this right where the old `_estimate_log_gaussian_prob` used to be.
###############################################################################
from functools import partial

def _compute_quadratic_form_tied(X_prec, mu_prec):
    """Return ‖X_prec - mu_prec‖² for every (sample, component) pair.

    Parameters
    ----------
    X_prec : ndarray of shape (n_samples, n_features)
        Data already multiplied by the common precision-Cholesky.

    mu_prec : ndarray of shape (n_components, n_features)
        Means already multiplied by the common precision-Cholesky.

    Returns
    -------
    quad : ndarray of shape (n_samples, n_components)
    """
    # ||a - b||² = ||a||² - 2 a·bᵀ + ||b||²
    # Row norms of X_prec
    x2 = row_norms(X_prec, squared=True)[:, None]               # (n_samples, 1)
    # Column norms of mu_prec
    m2 = row_norms(mu_prec, squared=True)[None, :]              # (1, n_components)
    # Cross term (X_prec @ mu_prec.T) – only one BLAS call
    cross = X_prec @ mu_prec.T
    return x2 - 2.0 * cross + m2


def _estimate_log_gaussian_prob(
    X, means, precisions_chol, covariance_type, *, _loop_thr=4
):
    """
    Compute per-sample per-component log-probabilities for a Gaussian.

    The 'full' covariance path now uses batched np.matmul instead of einsum:
      - proj_X = batch_prec @ X.T  → transpose → (n_samples, n_components, n_features)
      - proj_mu = batch_prec @ means[..., None] → squeeze → (n_components, n_features)

    If the full tensor doesn’t fit in RAM, we split into the fewest
    RAM-safe component‐chunks and apply the same batched matmul logic.
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features
    )
    log2pi = n_features * np.log(2.0 * np.pi)

    if covariance_type == "full":
        # total bytes for full tensor of shape (n_samples, n_components, n_features)
        total_bytes = n_samples * n_components * n_features * X.dtype.itemsize

        # ❶ tiny: direct Python loop
        if n_components <= _loop_thr and not _fits_in_memory(total_bytes):
            log_quad = np.empty((n_samples, n_components), dtype=X.dtype)
            for k, (mu, prec) in enumerate(zip(means, precisions_chol)):
                y = X @ prec - mu @ prec
                log_quad[:, k] = np.square(y).sum(axis=1)
            return -0.5 * (log2pi + log_quad) + log_det

        # ❷ medium: all‐in‐one batched matmul if it fits
        if _fits_in_memory(total_bytes):
            # proj_X: (n_components, n_features, n_samples) → transpose to (n_samples, n_components, n_features)
            proj_X = np.matmul(
                precisions_chol, X.T
            ).transpose(2, 0, 1)
            # proj_mu: (n_components, n_features, 1) → squeeze to (n_components, n_features)
            proj_mu = np.matmul(
                precisions_chol, means[:, :, None]
            ).squeeze(-1)
            # compute quad
            diff = proj_X - proj_mu[None, :, :]
            log_quad = np.square(diff).sum(axis=2)
            return -0.5 * (log2pi + log_quad) + log_det

        # ❸ large: chunk into RAM-safe blocks
        chunk_size = _optimal_chunk_size(n_samples, n_features, X.dtype)
        log_quad = np.empty((n_samples, n_components), dtype=X.dtype)

        for start in range(0, n_components, chunk_size):
            end = min(start + chunk_size, n_components)
            prec_chunk = precisions_chol[start:end]     # (k_c, f, f)
            mu_chunk = means[start:end]                # (k_c, f)
            k_c = end - start

            # batched matmul for this slice
            proj_X_chunk = np.matmul(
                prec_chunk, X.T
            ).transpose(2, 0, 1)                       # (n, k_c, f)
            proj_mu_chunk = np.matmul(
                prec_chunk, mu_chunk[:, :, None]
            ).squeeze(-1)                               # (k_c, f)

            diff = proj_X_chunk - proj_mu_chunk[None, :, :]
            log_quad[:, start:end] = np.square(diff).sum(axis=2)

        return -0.5 * (log2pi + log_quad) + log_det

    # ------------------------------------------------------------------
    # TIED covariance – unchanged
    # ------------------------------------------------------------------
    if covariance_type == "tied":
        X_prec = X @ precisions_chol
        mu_prec = means @ precisions_chol
        quad = _compute_quadratic_form_tied(X_prec, mu_prec)
        return -0.5 * (log2pi + quad) + log_det

    # ------------------------------------------------------------------
    # DIAG covariance – unchanged
    # ------------------------------------------------------------------
    if covariance_type == "diag":
        precisions = precisions_chol ** 2
        quad = (
            (means**2 * precisions).sum(axis=1)[None, :]
            - 2 * X @ (means * precisions).T
            + (X**2) @ precisions.T
        )
        return -0.5 * (log2pi + quad) + log_det

    # ------------------------------------------------------------------
    # SPHERICAL covariance – unchanged
    # ------------------------------------------------------------------
    precisions = precisions_chol ** 2
    x2 = row_norms(X, squared=True)
    mu2 = row_norms(means, squared=True)
    quad = (
        mu2[None, :] * precisions
        - 2 * (X @ means.T) * precisions
        + x2[:, None] * precisions
    )
    return -0.5 * (log2pi + quad) + log_det



class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full': each component has its own general covariance matrix.
        - 'tied': all components share the same general covariance matrix.
        - 'diag': each component has its own diagonal covariance matrix.
        - 'spherical': each component has its own single variance.

        For an example of using `covariance_type`, refer to
        :ref:`sphx_glr_auto_examples_mixture_plot_gmm_selection.py`.

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'}, \
    default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        String must be one of:

        - 'kmeans' : responsibilities are initialized using kmeans.
        - 'k-means++' : use the k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.

        .. versionchanged:: v1.1
            `init_params` now accepts 'random_from_data' and 'k-means++' as
            initialization methods.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

        For an example of using covariances, refer to
        :ref:`sphx_glr_auto_examples_mixture_plot_gmm_covariances.py`.

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence of the best fit of EM was reached, False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    lower_bounds_ : array-like of shape (`n_iter_`,)
        The list of lower bound values on the log-likelihood from each
        iteration of the best fit of EM.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])

    For a comparison of Gaussian Mixture with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """

    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "covariance_type": [StrOptions({"full", "tied", "diag", "spherical"})],
        "weights_init": ["array-like", None],
        "means_init": ["array-like", None],
        "precisions_init": ["array-like", None],
    }

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _initialize_parameters(self, X, random_state):
        # If all the initial parameters are all provided, then there is no need to run
        # the initialization.
        compute_resp = (
            self.weights_init is None
            or self.means_init is None
            or self.precisions_init is None
        )
        if compute_resp:
            super()._initialize_parameters(X, random_state)
        else:
            self._initialize(X, None)

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        weights, means, covariances = None, None, None
        if resp is not None:
            weights, means, covariances = _estimate_gaussian_parameters(
                X, resp, self.reg_covar, self.covariance_type
            )
            if self.weights_init is None:
                weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        else:
            self.precisions_cholesky_ = _compute_precision_cholesky_from_precisions(
                self.precisions_init, self.covariance_type
            )

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # Attributes computation
        _, n_features = self.means_.shape

        dtype = self.precisions_cholesky_.dtype
        if self.covariance_type == "full":
            # (k, d, d)  @  (k, d, d)^T  →  (k, d, d)
            self.precisions_ = self.precisions_cholesky_ @ self.precisions_cholesky_.transpose(0, 2, 1)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2 if self.covariance_type in {"diag", "spherical"} else self.precisions_cholesky_ @ self.precisions_cholesky_.T


    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the BIC used.

        For an example of GMM selection using `bic` information criterion,
        refer to :ref:`sphx_glr_auto_examples_mixture_plot_gmm_selection.py`.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(
            X.shape[0]
        )

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the AIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
