# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# License: BSD 3 clause (C) 2014

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf

import warnings
from time import time
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import check_non_negative
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import copy
import math
from numba import jit

# # mypy error: Module 'sklearn.manifold' has no attribute '_utils'
from sklearn.manifold import _utils  # type: ignore

# # mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from sklearn.manifold import _barnes_hut_tsne  # type: ignore


MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    P = (conditional_P + conditional_P.T) / (2 * distances.shape[0])
    return P


def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return P


def _kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _kl_divergence_hyperbolic(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    P_global=None,
    lambda_val=8,
    skip_num_points=0,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """

    """
    import scipy.io as sio
    mat_contents = sio.loadmat('../tmp.mat')
    X_embedded=mat_contents['Y']
    X_embedded_log=mat_contents['Y_log']
    P=mat_contents['probMatX']
    P_global=mat_contents['probMatXGlobal']
    """
    X_embedded = params.reshape(n_samples, n_components)
    X_embedded_log = copy.deepcopy(X_embedded)
    X_embedded_log[:, 0] = np.log(X_embedded_log[:, 0] + 1e-6)

    # local geometry (Euclidean)
    euc_dist = _dist_Euclinean(X_embedded_log)
    numerator_Q = 1 / (1 + euc_dist**2)
    np.fill_diagonal(numerator_Q, 0)
    Q = np.maximum(numerator_Q / np.sum(numerator_Q), MACHINE_EPSILON)
    euc_der = _derivative_Euclidean(X_embedded_log)
    euc_der[:, :, 0] = euc_der[:, :, 0] / _repvec(X_embedded[:, 0], n_samples)
    pdiff_local = (P - Q) * numerator_Q
    np.fill_diagonal(pdiff_local, 0)

    # global geometry (hyperbolic)
    hyper_dist = _dist_Hyperbolic(X_embedded_log)
    numerator_Q_global = 1 + hyper_dist**2
    np.fill_diagonal(numerator_Q_global, 0)
    Q_global = np.maximum(
        numerator_Q_global / np.sum(numerator_Q_global), MACHINE_EPSILON
    )
    hyper_der = _derivative_Hyperbolic(X_embedded_log)
    hyper_der[:, :, 0] = hyper_der[:, :, 0] / _repvec(X_embedded[:, 0], n_samples)
    pdiff_global = -lambda_val * (P_global - Q_global) / numerator_Q_global
    np.fill_diagonal(pdiff_global, 0)

    # combined gradient
    grad_local = 4 * np.sum(
        (pdiff_local * euc_dist).reshape((*pdiff_local.shape, 1)) * euc_der, 1
    )
    grad_global = 4 * np.sum(
        (pdiff_global * hyper_dist).reshape((*pdiff_global.shape, 1)) * hyper_der, 1
    )
    grad = grad_local + grad_global

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = np.nansum(P * np.log(P / Q)) + lambda_val * np.nansum(
            P_global * np.log(P_global / Q_global)
        )
    else:
        kl_divergence = np.nan

    return kl_divergence, grad


# non python compile using numba
@jit(nopython=True, cache=True)
def _repvec(X, n):
    return np.repeat(X, n).reshape((n, n))


np.seterr(divide="ignore", invalid="ignore")


@jit(nopython=True, cache=True)
def _dist_Euclinean(
    X,
):
    n, p = X.shape
    M = np.zeros((n, n), dtype=np.float64)
    cos_angle = np.cos(_repvec(X[:, p - 1], n) - _repvec(X[:, p - 1], n).T)
    for count_angle in list(range(p - 2, 0, -1)):
        cos_angle = (
            _repvec(np.sin(X[:, count_angle]), n)
            * _repvec(np.sin(X[:, count_angle]), n).T
            * cos_angle
            + _repvec(np.cos(X[:, count_angle]), n)
            * _repvec(np.cos(X[:, count_angle]), n).T
        )
    M = np.real(
        np.sqrt(
            _repvec(X[:, 0] ** 2, n)
            + _repvec(X[:, 0] ** 2, n).T
            - 2 * (_repvec(X[:, 0], n) * _repvec(X[:, 0], n).T) * cos_angle
        )
    )
    np.fill_diagonal(M, 0)
    return M


@jit(nopython=True, cache=True)
def _derivative_Euclidean(
    X,
):
    n, p = X.shape
    Y = np.zeros((n, n, p), dtype=np.float64)
    cos_angle = np.cos(_repvec(X[:, p - 1], n) - _repvec(X[:, p - 1], n).T)
    for count_angle in list(range(p - 2, 0, -1)):
        cos_angle = (
            _repvec(np.sin(X[:, count_angle]), n)
            * _repvec(np.sin(X[:, count_angle]), n).T
            * cos_angle
            + _repvec(np.cos(X[:, count_angle]), n)
            * _repvec(np.cos(X[:, count_angle]), n).T
        )
    dist = np.sqrt(
        _repvec(X[:, 0] ** 2, n)
        + _repvec(X[:, 0] ** 2, n).T
        - 2 * (_repvec(X[:, 0], n) * _repvec(X[:, 0], n).T) * cos_angle
    )
    Y[:, :, 0] = 1 / dist * (_repvec(X[:, 0], n) - _repvec(X[:, 0], n).T * cos_angle)

    for count_angle in list(range(p - 1, 0, -1)):
        if count_angle == p - 1:
            Y[:, :, count_angle] = (
                1
                / dist
                * (_repvec(X[:, 0], n) * _repvec(X[:, 0], n).T)
                * np.sin(
                    _repvec(X[:, count_angle], n) - _repvec(X[:, count_angle], n).T
                )
            )
            cos_angle = np.cos(
                _repvec(X[:, count_angle], n) - _repvec(X[:, count_angle], n).T
            )
        else:
            Y[:, :, count_angle] = (
                -1
                / dist
                * (_repvec(X[:, 0], n) * _repvec(X[:, 0], n).T)
                * (
                    _repvec(np.cos(X[:, count_angle]), n)
                    * _repvec(np.sin(X[:, count_angle]), n).T
                    * cos_angle
                    - _repvec(np.sin(X[:, count_angle]), n)
                    * _repvec(np.cos(X[:, count_angle]), n).T
                )
            )
            cos_angle = (
                _repvec(np.sin(X[:, count_angle]), n)
                * _repvec(np.sin(X[:, count_angle]), n).T
                * cos_angle
                + _repvec(np.cos(X[:, count_angle]), n)
                * _repvec(np.cos(X[:, count_angle]), n).T
            )
        for ii in range(1, count_angle):
            Y[:, :, count_angle] = (
                _repvec(np.sin(X[:, ii]), n)
                * _repvec(np.sin(X[:, ii]), n).T
                * Y[:, :, count_angle]
            )
    Y = np.real(Y)
    for k in range(0, p):
        np.fill_diagonal(Y[:, :, k], 0)
    return Y


@jit(nopython=True, cache=True)
def _dist_Hyperbolic(
    X,
):
    n, p = X.shape
    cos_angle = np.cos(_repvec(X[:, p - 1], n) - _repvec(X[:, p - 1], n).T)
    for count_angle in list(range(p - 2, 0, -1)):
        cos_angle = (
            _repvec(np.sin(X[:, count_angle]), n)
            * _repvec(np.sin(X[:, count_angle]), n).T
            * cos_angle
            + _repvec(np.cos(X[:, count_angle]), n)
            * _repvec(np.cos(X[:, count_angle]), n).T
        )
    M = np.real(
        np.arccosh(
            _repvec(np.cosh(X[:, 0]), n) * _repvec(np.cosh(X[:, 0]), n).T
            - _repvec(np.sinh(X[:, 0]), n) * _repvec(np.sinh(X[:, 0]), n).T * cos_angle
        )
    )
    np.fill_diagonal(M, 0)
    return M


@jit(nopython=True, cache=True)
def _derivative_Hyperbolic(
    X,
):
    n, p = X.shape
    Y = np.zeros((n, n, p), dtype=np.float64)
    cos_angle = np.cos(_repvec(X[:, p - 1], n) - _repvec(X[:, p - 1], n).T)
    for count_angle in list(range(p - 2, 0, -1)):
        cos_angle = (
            _repvec(np.sin(X[:, count_angle]), n)
            * _repvec(np.sin(X[:, count_angle]), n).T
            * cos_angle
            + _repvec(np.cos(X[:, count_angle]), n)
            * _repvec(np.cos(X[:, count_angle]), n).T
        )

    cosh_angle = (
        _repvec(np.cosh(X[:, 0]), n) * _repvec(np.cosh(X[:, 0]), n).T
        - _repvec(np.sinh(X[:, 0]), n) * _repvec(np.sinh(X[:, 0]), n).T * cos_angle
    )
    Y[:, :, 0] = (
        1
        / np.sqrt(cosh_angle**2 - 1)
        * (
            _repvec(np.sinh(X[:, 0]), n) * _repvec(np.cosh(X[:, 0]), n).T
            - _repvec(np.cosh(X[:, 0]), n) * _repvec(np.sinh(X[:, 0]), n).T * cos_angle
        )
    )

    for count_angle in list(range(p - 1, 0, -1)):
        if count_angle == p - 1:
            Y[:, :, count_angle] = (
                1
                / np.sqrt(cosh_angle**2 - 1)
                * (_repvec(np.sinh(X[:, 0]), n) * _repvec(np.sinh(X[:, 0]), n).T)
                * np.sin(
                    _repvec(X[:, count_angle], n) - _repvec(X[:, count_angle], n).T
                )
            )
            cos_angle = np.cos(
                _repvec(X[:, count_angle], n) - _repvec(X[:, count_angle], n).T
            )
        else:
            Y[:, :, count_angle] = (
                -1
                / np.sqrt(cosh_angle**2 - 1)
                * (_repvec(np.sinh(X[:, 0]), n) * _repvec(np.sinh(X[:, 0]), n).T)
                * (
                    _repvec(np.cos(X[:, count_angle]), n)
                    * _repvec(np.sin(X[:, count_angle]), n).T
                    * cos_angle
                    - _repvec(np.sin(X[:, count_angle]), n)
                    * _repvec(np.cos(X[:, count_angle]), n).T
                )
            )
            cos_angle = (
                _repvec(np.sin(X[:, count_angle]), n)
                * _repvec(np.sin(X[:, count_angle]), n).T
                * cos_angle
                + _repvec(np.cos(X[:, count_angle]), n)
                * _repvec(np.cos(X[:, count_angle]), n).T
            )
        for ii in range(1, count_angle):
            Y[:, :, count_angle] = (
                _repvec(np.sin(X[:, ii]), n) * _repvec(np.sin(X[:, ii]), n).T
            ) * Y[:, :, count_angle]
    Y = np.real(Y)
    for k in range(0, p):
        np.fill_diagonal(Y[:, :, k], 0)
    return Y


# @jit(nopython=True, cache=True)
def _rescale_angle(
    X,
):
    s, d = X.shape
    Y = X.copy()
    for k in range(1, d):
        Y[:, k] = _rescale_period(Y[:, k], 0, 2 * np.pi)

    change_sign = np.zeros(s)
    for k in range(1, d - 1):
        Y[:, k] = _rescale_period(Y[:, k] + change_sign * np.pi, 0, 2 * np.pi)
        change_sign = Y[:, k] > np.pi
        Y[change_sign, k] = 2 * np.pi - Y[change_sign, k]

    Y[:, d - 1] = _rescale_period(Y[:, d - 1] + change_sign * np.pi, 0, 2 * np.pi)
    return Y


@jit(nopython=True, cache=True)
def _rescale_period(X, l, r):
    # Rescale the input value into a given range [l,r]
    delta = r - l
    Y = X.copy()
    for k in range(0, len(Y)):
        if Y[k] < l:
            Y[k] = Y[k] + delta
            while Y[k] < l:
                Y[k] = Y[k] + delta
        elif Y[k] > r:
            Y[k] = Y[k] - delta
            while Y[k] > r:
                Y[k] = Y[k] - delta
    return Y


def _kl_divergence_bh(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    skip_num_points=0,
    verbose=False,
    compute_error=True,
    num_threads=1,
):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2).

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : sparse matrix of shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized. Matrix should be of CSR format.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float, default=0.5
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : int, default=False
        Verbosity level.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    num_threads : int, default=1
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        dof=degrees_of_freedom,
        compute_error=compute_error,
        num_threads=num_threads,
    )
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad


def _gradient_descent(
    objective,
    p0,
    it,
    n_iter,
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.2,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    args=None,
    kwargs=None,
):
    """Batch gradient descent with momentum and individual gains.

    Parameters
    ----------
    objective : callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.

    p0 : array-like of shape (n_params,)
        Initial parameter vector.

    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).

    n_iter : int
        Maximum number of gradient descent iterations.

    n_iter_check : int, default=1
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization.

    momentum : float within (0.0, 1.0), default=0.8
        The momentum generates a weight for previous gradients that decays
        exponentially.

    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.

    min_gain : float, default=0.01
        Minimum individual gain for each parameter.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be aborted.

    verbose : int, default=0
        Verbosity level.

    args : sequence, default=None
        Arguments to pass to objective function.

    kwargs : dict, default=None
        Keyword arguments to pass to objective function.

    Returns
    -------
    p : ndarray of shape (n_params,)
        Optimum parameters.

    error : float
        Optimum.

    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)
        # print(grad.reshape((1174,3)))

        grad = grad.copy().ravel()
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.1
        gains[dec] *= 0.9
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        # print(grad_norm)
        # print(grad.reshape((1174,3)))
        # print(p.reshape((1174,3)))

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print(
                    "[h-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[h-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[h-SNE] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm)
                    )
                break

    return p, error, i


def trustworthiness(X, X_embedded, *, n_neighbors=5, metric="euclidean"):
    r"""Expresses to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : ndarray of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

        .. versionadded:: 0.20

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.
    """
    n_samples = X.shape[0]
    if n_neighbors >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )
    dist_X = pairwise_distances(X, metric=metric)
    if metric == "precomputed":
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(X_embedded)
        .kneighbors(return_distance=False)
    )

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    )
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t


# @jit(nopython=True, cache=True)
def _sampling_native(n_samples, n_components, init_R, min_ratio):
    # Use Native model of hyperbolic geometry
    cart_pos = np.zeros((n_samples, n_components), dtype=np.float64)
    for k in range(0, n_samples):
        cart_pos[k, :] = np.random.randn(1, n_components)
        cart_pos[k, :] = cart_pos[k, :] / np.linalg.norm(cart_pos[k, :])

    Y = _cart2polar(cart_pos)
    k = 0
    while k < n_samples:
        r = min_ratio + np.random.rand(1) * (1 - min_ratio)
        y = np.random.rand(1) * np.sinh(init_R) ** (n_components - 1)
        if np.sinh(init_R * r) ** (n_components - 1) >= y:
            Y[k, 0] = init_R * r
            k = k + 1
    return Y


@jit(nopython=True, cache=True)
def _cart2polar(X):
    r, c = X.shape
    Y = np.ones((r, c), dtype=np.float64)
    Y[:, 0] = np.sqrt(np.sum(X**2, axis=1))
    temp = Y[:, 0]
    if c > 1:
        c = c - 1
        for k in range(0, c - 1):
            theta = np.arccos(X[:, c - k] / temp)
            Y[:, k + 1] = theta
            temp = temp * np.sin(theta)

        for k in range(0, r):
            if X[k, 1] / temp[k] > 0:
                Y[k, c] = np.arccos(X[k, 0] / temp[k])
            else:
                Y[k, c] = 2 * np.pi - np.arccos(X[k, 0] / temp[k])

    return Y


class HSNE(BaseEstimator):
    """hyperbolic T-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

    Read more in the :ref:`User Guide <t_sne>`.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less that the number
        of samples.

    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float or 'auto', default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
        Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
        etc.) use a definition of learning_rate that is 4 times smaller than
        ours. So our learning_rate=200 corresponds to learning_rate=800 in
        those other implementations. The 'auto' option sets the learning_rate
        to `max(N / early_exaggeration / 4, 50)` where N is the sample size,
        following [4] and [5]. This will become default in 1.2.

    n_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 1.1

    init : {'random', 'pca'} or ndarray of shape (n_samples, n_components), \
            default='random'
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization. `init='pca'`
        will become default in 1.2.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term:`Glossary <random_state>`.

    method : str, default='barnes_hut'
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.22

    square_distances : True, default='deprecated'
        This parameter has no effect since distance values are always squared
        since 1.1.

        .. deprecated:: 1.1
             `square_distances` has no effect from 1.1 and will be removed in
             1.3.

    Attributes
    ----------
    embedding_ : array-like of shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    MDS : Manifold learning using multidimensional scaling.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/

    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

    [4] Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J.,
        & Snyder-Cappione, J. E. (2019). Automated optimized parameters for
        T-distributed stochastic neighbor embedding improve visualization
        and analysis of large datasets. Nature Communications, 10(1), 1-12.

    [5] Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
        transcriptomics. Nature Communications, 10(1), 1-14.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2, learning_rate='auto',
    ...                   init='random', perplexity=3).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    """

    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 10

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=4.0,
        learning_rate=0.7,
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="native",
        verbose=10,
        random_state=None,
        method="exact",
        angle=0.5,
        n_jobs=None,
        square_distances="deprecated",
        init_R=1,
        upper_raius=False,
        momentum=0.2,
        lambda_val=8,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.square_distances = square_distances
        self.init_R = init_R
        self.upper_raius = upper_raius
        self.momentum = momentum
        self.lambda_val = lambda_val

    def _check_params_vs_input(self, X):
        if self.perplexity >= X.shape[0]:
            raise ValueError("perplexity must be less than n_samples")

    def _fit(self, X, skip_num_points=0):
        """Private function to fit the model using X as training data."""

        if isinstance(self.init, str) and self.init == "warn":
            # See issue #18018
            warnings.warn(
                "The default initialization in TSNE will change "
                "from 'random' to 'pca' in 1.2.",
                FutureWarning,
            )
            self._init = "random"
        else:
            self._init = self.init
        if self.learning_rate == "warn":
            # See issue #18018
            warnings.warn(
                "The default learning rate in TSNE will change "
                "from 200.0 to 'auto' in 1.2.",
                FutureWarning,
            )
            self._learning_rate = 200.0
        else:
            self._learning_rate = self.learning_rate

        if isinstance(self._init, str) and self._init == "pca" and issparse(X):
            raise TypeError(
                "PCA initialization is currently not supported "
                "with the sparse input matrix. Use "
                'init="random" instead.'
            )
        if self.method not in ["barnes_hut", "exact"]:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.square_distances != "deprecated":
            warnings.warn(
                "The parameter `square_distances` has not effect and will be "
                "removed in version 1.3.",
                FutureWarning,
            )
        if self._learning_rate == "auto":
            # See issue #18018
            self._learning_rate = X.shape[0] / self.early_exaggeration / 4
            self._learning_rate = np.maximum(self._learning_rate, 50)
        else:
            if not (self._learning_rate > 0):
                raise ValueError("'learning_rate' must be a positive number or 'auto'.")
        if self.method == "barnes_hut":
            X = self._validate_data(
                X,
                accept_sparse=["csr"],
                ensure_min_samples=2,
                dtype=[np.float32, np.float64],
            )
        else:
            X = self._validate_data(
                X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
            )
        if self.metric == "precomputed":
            if isinstance(self._init, str) and self._init == "pca":
                raise ValueError(
                    'The parameter init="pca" cannot be used with metric="precomputed".'
                )
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(
                X,
                "TSNE.fit(). With metric='precomputed', X "
                "should contain positive distances.",
            )

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    "or provide the dense distance matrix."
                )

        if self.method == "barnes_hut" and self.n_components > 3:
            raise ValueError(
                "'n_components' should be inferior to 4 for the "
                "barnes_hut algorithm as it relies on "
                "quad-tree or oct-tree."
            )
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError(
                "early_exaggeration must be at least 1, but is {}".format(
                    self.early_exaggeration
                )
            )

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    # Euclidean is squared here, rather than using **= 2,
                    # because euclidean_distances already calculates
                    # squared distances, and returns np.sqrt(dist) for
                    # squared=False.
                    # Also, Euclidean is slower for n_jobs>1, so don't set here
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    metric_params_ = self.metric_params or {}
                    distances = pairwise_distances(
                        X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_
                    )

            if np.any(distances < 0):
                raise ValueError(
                    "All distances should be positive, the metric given is not correct"
                )

            if self.metric != "euclidean":
                distances **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(
                P <= 1
            ), "All probabilities should be less or then equal to one"

            # diagonal axis set to zero
            np.fill_diagonal(P, 0)

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors...".format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(
                algorithm="auto",
                n_jobs=self.n_jobs,
                n_neighbors=n_neighbors,
                metric=self.metric,
                metric_params=self.metric_params,
            )
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Indexed {} samples in {:.3f}s...".format(
                        n_samples, duration
                    )
                )

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode="distance")
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Computed neighbors for {} samples in {:.3f}s...".format(
                        n_samples, duration
                    )
                )

            # Free the memory used by the ball_tree
            del knn

            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)

        if isinstance(self._init, np.ndarray):
            X_embedded = self._init
        elif self._init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            # TODO: Update in 1.2
            # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
            # the default value for random initialization. See issue #18018.
            warnings.warn(
                "The PCA initialization in TSNE will change to "
                "have the standard deviation of PC1 equal to 1e-4 "
                "in 1.2. This will ensure better convergence.",
                FutureWarning,
            )
            # X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
        elif self._init == "random":
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.standard_normal(
                size=(n_samples, self.n_components)
            ).astype(np.float32)
        elif self._init == "native":
            X_embedded = _sampling_native(
                n_samples=n_samples,
                n_components=self.n_components,
                init_R=self.init_R,
                min_ratio=0,
            )
            X_embedded[:, 0] = np.exp(X_embedded[:, 0])

        else:
            raise ValueError("'init' must be 'pca', 'random', or a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._hsne(
            P,
            X,
            self.lambda_val,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            neighbors=neighbors_nn,
            skip_num_points=skip_num_points,
        )

    def _hsne(
        self,
        P,
        X,
        lambda_val,
        degrees_of_freedom,
        n_samples,
        X_embedded,
        neighbors=None,
        skip_num_points=0,
    ):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self._learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": self.momentum,
        }
        if self.method == "barnes_hut":
            obj_func = _kl_divergence_bh
            opt_args["kwargs"]["angle"] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args["kwargs"]["verbose"] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args["kwargs"]["num_threads"] = _openmp_effective_n_threads()
        elif self.method == "exact":
            obj_func = _kl_divergence_hyperbolic
            X_sum = (X**2).sum(axis=1)
            P_global = (
                1
                + _repvec(X_sum, X_sum.shape[0])
                + _repvec(X_sum, X_sum.shape[0]).T
                - 2 * np.dot(X, X.T)
            )
            P_global = np.maximum(P_global / np.sum(P_global), MACHINE_EPSILON)
            opt_args["kwargs"]["P_global"] = P_global
            opt_args["kwargs"]["lambda_val"] = lambda_val

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
        if self.verbose:
            print(
                "[t-SNE] KL divergence after %d iterations with early exaggeration: %f"
                % (it + 1, kl_divergence)
            )

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args["n_iter"] = self.n_iter
            opt_args["it"] = it + 1
            opt_args["momentum"] = 0.8
            opt_args["n_iter_without_progress"] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print(
                "[h-SNE] KL divergence after %d iterations: %f"
                % (it + 1, kl_divergence)
            )

        X_embedded = params.reshape(n_samples, self.n_components)
        if self.method == "exact":
            X_embedded[:, 0] = np.log(X_embedded[:, 0] + 1e-6)

        self.kl_divergence_ = kl_divergence

        return X_embedded

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self._check_params_vs_input(X)
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit_transform(X)
        return self

    def _more_tags(self):
        return {"pairwise": self.metric == "precomputed"}
