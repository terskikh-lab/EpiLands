"""
Hyperbolic Multi-dimensional Scaling (MDS).
"""
# Kenta Ninomiya@ Sanford Burnham Prebys Medical Discovery Institute

import numpy as np
from joblib import Parallel, effective_n_jobs
from numba import jit

import warnings

from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.fixes import delayed

from scipy.spatial.distance import squareform
from scipy.optimize import fminbound

from hsnepy import _sampling_native
from hsnepy import _dist_Hyperbolic
from hsnepy import _derivative_Hyperbolic


def _hgd(
    dissimilarities,
    metric=True,
    r_max=3,
    min_ratio=0,
    n_components=2,
    init=None,
    max_iter=300,
    verbose=0,
    eps=1e-5,
    random_state=None,
    update_radius=True,
):
    """Computes multidimensional scaling using SMACOF algorithm.

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    n_iter : int
        The number of iterations corresponding to the best stress.
    """
    linea_search_eps = 1e-3
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)
    n_samples = dissimilarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * dissimilarities).ravel()
    nonzero_sim_flat = sim_flat != 0
    sim_flat_w = sim_flat[nonzero_sim_flat]
    if init is None:
        # Randomly initialize
        X = _sampling_native(
            n_samples=n_samples,
            n_components=n_components,
            init_R=r_max,
            min_ratio=min_ratio,
        )
    else:
        """
        Check this section
        """
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )
        X = init

    reset_CG = True
    old_stress = np.inf
    old_norm_grad = 0
    step_len = np.inf
    ir = IsotonicRegression()
    all_stress = list()
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = _dist_Hyperbolic(X)
        if metric:
            disparities_flat = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[nonzero_sim_flat]
            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)

        stress, grad = _stress_criterion(
            X, nonzero_sim_flat, disparities_flat, True, n_components, update_radius
        )

        # normalize the gradient
        norm_grad = np.linalg.norm(grad.ravel())

        # early stopping
        if verbose >= 2:
            print("it: %d, stress %s" % (it, stress))

        if stress < eps:
            # if verbose >= 2:
            if True:
                print("stats:mdscale:TerminatedCriterion")
            break
        elif norm_grad < eps * stress:
            # if verbose >= 2:
            if True:
                print("stats:mdscale:TerminatedRelativeNormOfGradient")
            break
        elif (old_stress - stress) < eps * stress:
            # if verbose >= 2:
            if True:
                print("stats:mdscale:TerminatedRelativeChangeInCriterion")
            break
        elif step_len < eps * np.linalg.norm(X.ravel()):
            # if verbose >= 2:
            if True:
                print("stats:mdscale:TerminatedNormOfChangeInConfiguration")
            break

        # Polak-Riviere conjgate gradient serching
        if reset_CG:
            p = -grad
            reset_CG = False
        else:
            # beta = max(((grad(:)-oldGrad(:))'*grad(:)) / oldNormGrad^2, 0);
            beta = (
                (grad.ravel() - old_grad.ravel()) * (grad.ravel()) / old_norm_grad**2
            ).max()
            p = -grad + beta * p

        # initialize the old parameters
        old_stress = stress
        old_grad = grad
        old_norm_grad = norm_grad

        # line search
        max_step_len = 2
        # find the upper bound on step length gives a stress increase
        while True:
            stress = _line_search_stress(
                max_step_len,
                _stress_criterion,
                X,
                p,
                nonzero_sim_flat,
                disparities_flat,
                getGrad=False,
            )
            if stress >= old_stress or np.isnan(stress):
                break
            else:
                max_step_len = max_step_len * 2

        # find the local minimum with scipy optimize fminbound
        alpha, stress, err, output = fminbound(
            func=_line_search_stress,
            x1=0,
            x2=max_step_len,
            xtol=linea_search_eps,
            full_output=True,
            args=(_stress_criterion, X, p, nonzero_sim_flat, disparities_flat),
        )
        if err == 1:
            # [addition] warnings
            # configure the way to warn users
            print("stats:mdscale:LineSrchIterLimit")
            # warning(message('stats:mdscale:LineSrchIterLimit'));

        elif stress > old_stress:
            # in case fminbound found a local minimum that is higher than the
            # previous stress, because the stress initially decreases to the true
            # minimum, then increases and has a local min beyond that.  Have no
            # truck with that.
            while True:
                alpha = alpha / 2
                if alpha <= 1e-12:
                    print("Nosolution")
                    # error(message('stats:mdscale:NoSolution'));
                stress = _line_search_stress(
                    alpha,
                    _stress_criterion,
                    X,
                    p,
                    nonzero_sim_flat,
                    disparities_flat,
                    getGrad=False,
                )
                if stress <= old_stress:
                    break
            resetCG = True
        # Take the downhill step.
        X_step = alpha * p
        step_len = alpha * np.linalg.norm(p.ravel())
        temp = X[:, 0]
        # [addition] update reference
        X = X + X_step
        X[X[:, 0] > r_max, 0] = r_max
        X[X[:, 0] <= 0, 0] = temp[X[:, 0] <= 0]
        all_stress.append(stress)
        # print('X_step: ',X_step[:5])
        # Tighten up the line search tolerance, but not beyond the requested
        # tolerance.
        linea_search_eps = max(linea_search_eps / 2, eps)

    # if verb > 1:
    # [addition] progress printing
    # fprintf('%s\n',getString(message('stats:mdscale:IterationsStress',iter,sprintf('%g',stress))));
    print(it)
    return X, stress, it + 1


# @jit(nopython=True)
def _stress_criterion(
    X,
    nonzero_indices,
    disparities,
    getGrad=False,
    n_components=None,
    update_radius=None,
):
    """Compute the stress criterion.
    distances: array of shape (n_samples * (n_samples - 1) / 2,)
        The flattened upper triangular part of the distance matrix.
    disparities: array of shape (n_samples * (n_samples - 1) / 2,)
        The flattened upper triangular part of the disparity matrix.
    """

    # Compute stress
    distances = _dist_Hyperbolic(X.astype(np.float64)).ravel()[nonzero_indices]
    num_diff = distances - disparities
    stress_num = (num_diff**2).sum()
    stress_den = (distances**2).sum()
    stress = np.sqrt(stress_num / stress_den)

    if not getGrad:
        return stress

    grad = np.zeros(X.shape)
    if stress_num > 0:
        if (distances > 0).all():
            dS = squareform(num_diff / stress_num - distances / stress_den)
            dX = _derivative_Hyperbolic(X)
            for i in range(0, n_components):
                grad[:, i] = (dS * dX[:, :, i]).sum(axis=1) * stress / 2
            if not update_radius:
                grad[:, 0] = 0
        else:
            # configure the way to warn users
            print("debug")
            # error(message('stats:mdscale:ColocatedPoints'));

    return stress, grad


def _line_search_stress(
    x_variable,
    fun,
    X,
    p,
    nonzero_indices,
    disparities,
    getGrad=False,
    n_components=None,
    update_radius=None,
):
    return fun(
        X + x_variable * p,
        nonzero_indices,
        disparities,
        getGrad,
        n_components,
        update_radius,
    )


def hyperbolicgd(
    dissimilarities,
    *,
    metric=False,
    r_max=3,
    min_ratio=0,
    n_components=2,
    init=None,
    n_init=8,
    n_jobs=None,
    max_iter=300,
    verbose=0,
    eps=1e-5,
    random_state=None,
    return_n_iter=False,
):
    """Compute multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, default=8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.

    Notes
    -----
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)

    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)

    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)

    if hasattr(init, "__array__"):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                "Explicit initial positions passed: "
                "performing only one init of the MDS instead of %d" % n_init
            )
            n_init = 1

    best_pos, best_stress = None, None

    if effective_n_jobs(n_jobs) == 1:
        for it in range(n_init):
            pos, stress, n_iter_ = _hgd(
                dissimilarities,
                metric=metric,
                r_max=r_max,
                min_ratio=min_ratio,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=random_state,
            )
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(_hgd)(
                dissimilarities,
                metric=metric,
                n_components=n_components,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                eps=eps,
                random_state=seed,
            )
            for seed in seeds
        )
        positions, stress, n_iters = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter
    else:
        return best_pos, best_stress


class HMDS(BaseEstimator):
    """Multidimensional scaling with hyperbolic geometry.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.

    metric : bool, default=True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:

        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.

        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.

    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).

    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Symmetric matrix that:

        - either uses a custom dissimilarity matrix by setting `dissimilarity`
          to 'precomputed';
        - or constructs a dissimilarity matrix from data using
          Euclidean distances.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The number of iterations corresponding to the best stress.

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    TSNE : T-distributed Stochastic Neighbor Embedding.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    References
    ----------
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)

    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)

    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import MDS
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = MDS(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """

    def __init__(
        self,
        n_components=5,
        *,
        metric=False,
        r_max=3,
        min_ratio=0,
        n_init=4,
        max_iter=1000,
        verbose=0,
        eps=1e-5,
        n_jobs=None,
        random_state=None,
        dissimilarity="precomputed",
    ):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.r_max = r_max
        self.min_ratio = min_ratio
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _more_tags(self):
        return {"pairwise": self.dissimilarity == "precomputed"}

    def fit(self, X, y=None, init=None):
        """
        Compute the position of the points in the embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples,), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from `X`, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        """
        X = self._validate_data(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn(
                "The MDS API has changed. ``fit`` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "``dissimilarity='precomputed'``."
            )

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError(
                "Proximity must be 'precomputed' or 'euclidean'. Got %s instead"
                % str(self.dissimilarity)
            )

        self.embedding_, self.stress_, self.n_iter_ = hyperbolicgd(
            self.dissimilarity_matrix_,
            metric=self.metric,
            r_max=self.r_max,
            min_ratio=self.min_ratio,
            n_components=self.n_components,
            init=init,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            verbose=self.verbose,
            eps=self.eps,
            random_state=self.random_state,
            return_n_iter=True,
        )

        return self.embedding_
