"""Utility operations built on scikit-learn"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition.nmf import _initialize_nmf
from scipy.optimize import fmin_powell
from multiprocessing import Pool
from tslearn.metrics import dtw
import numpy as np


class NMFWithDTW(BaseEstimator, TransformerMixin):
    """Nonnegative Matrix Factorization using Dynamic Time Warping (DTW) as loss function

    NMF is typically defined as using beta loss (e.g., Frobenius) to measure the difference
    between an original signal and the reconstructed one. This new implementation uses
    DTW to measure the distance between two patterns because it can account for non-linear
    distortions between the signals (e.g., caused by shifts in the peak locations of XRD
    due to thermal expansion).

    The DTW optimizer, by itself, does not take any match with regard to the
    non-distorted signal into account. As we want components that are meaningful,
    we need them to reconstruct at least some patterns without distortion.
    We achieve this by adding the minimum :math:`L_2` error between
    any signal and the reconstructed signal, which -- in effect -- penalizes
    reconstructions that do not match *any* patterns well.

    DTW is not differentiable, so we are going to use derivative-free optimizers.

    Our objective function is::
    
        DTW(X, WH)
            + frobenius * min_i(||X_i - W_iH_i||_Fro)
            + alpha * l1_ratio * ||vec(W)||_1
            + alpha * l1_ratio * ||vec(H)||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
            + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

    Parameters
    ----------
        n_components : int
            Number of components
        init : None | 'random' | 'nndsvd' |  'nndsvda' | 'nndsvdar' | 'custom'
            Method used to initialize the procedure.
            Default: None.
            Valid options:
            - None: 'nndsvd' if n_components <= min(n_samples, n_features),
                otherwise random.
            - 'random': non-negative random matrices, scaled with:
                sqrt(X.mean() / n_components)
            - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)
            - 'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)
            - 'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)
            - 'custom': use custom matrices W and H
            .. versionadded:: 0.19
        absolute_error_weight: float, default: 0
            Weight for the error between original and reconstructed patterns
        tol : float, default: 1e-4
            Tolerance of the stopping condition.
        max_iter : integer, default: 200
            Maximum number of iterations before timing out.
        random_state : int, RandomState instance or None, optional, default: None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        alpha : double, default: 0.
            Constant that multiplies the regularization terms. Set it to zero to
            have no regularization.
        l1_ratio : double, default: 0.
            The regularization mixing parameter, with 0 <= l1_ratio <= 1.
            For l1_ratio = 0 the penalty is an elementwise L2 penalty
            (aka Frobenius Norm).
            For l1_ratio = 1 it is an elementwise L1 penalty.
            For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
            .. versionadded:: 0.17
               Regularization parameter *l1_ratio* used in the Coordinate Descent
               solver.
        verbose : bool, default=False
            Whether to be verbose
        optimizer_args : dict, default=None
            Options for the :meth:`fmin_powell` optimizer
        n_jobs: int, default=1
            How many processors to use for the DTW calculation
    """

    def __init__(self, n_components=2, absolute_error_weight=0,
                 init=None, tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 optimizer_args=None, n_jobs=1):
        self.n_components = n_components
        self.absolute_error_weight = absolute_error_weight
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.optimizer_args = optimizer_args or {}
        self.n_jobs = n_jobs

        # Placeholder for result variables
        self.components_ = self.reconstruction_error_ = None
        self.error_history_ = None

    def fit(self, X, y=None, W=None, H=None):
        """Determine the factorization

        Args:
            X (ndarray): Array to be factorized
            y: Not used
            W (ndarray): Initial guess for the weights for each row
            H (ndarray): Initial guess for the components
        Returns:
            self
        """

        self.fit_transform(X, y, W, H)
        return self

    def fit_transform(self, X, y=None, W=None, H=None):
        """Determine and apply the factorization

        Args:
            X (ndarray): Array to be factorized
            y: Not used
            W (ndarray): Initial guess for the weights for each row
            H (ndarray): Initial guess for the components
        Returns:
            (ndarray) The weights for each row
        """

        # Make the initial guess
        if self.init == 'custom':
            assert W is not None, "W must be supplied if `init` is custom"
            assert H is not None, "H must be supplied if `init` is custom"
        elif W is None or H is None:
            W, H = self.make_initial_guesses(X)
        W_shape = np.shape(W)
        W_size = np.size(W)
        H_shape = np.shape(H)

        # Make a process pool, if needed
        pool = None
        if self.n_jobs != 1:
            pool = Pool(self.n_jobs)

        # Make the loss function
        def loss_function(params):
            """Compute the loss function for the pattern match

            Args:
                params (ndarray): Logarithm of desired parameters
            """

            # Compute the exponential of the input parameters
            params = np.exp(params)

            # Split the parameter matrix back into separate W and H matrices
            W = np.reshape(params[:W_size], W_shape)
            H = np.reshape(params[W_size:], H_shape)

            # Compute the reconstructed signal
            recon_X = np.matmul(W, H)

            # Compute the DTW similarity between each reconstructed pattern
            dtw_score = 0
            if pool is None:
                for x, recon_x in zip(X, recon_X):
                    dtw_score += dtw(x, recon_x)
            else:
                dtw_scores = pool.starmap(dtw, zip(X, recon_X))
                dtw_score += sum(dtw_scores)

            # Get the best match overall of the patterns
            if self.absolute_error_weight > 0:
                match_score = np.abs(recon_X - X).sum()
                dtw_score += self.absolute_error_weight * match_score

            # Add in the other losses
            if self.alpha > 0:
                reg_losses = 0
                if self.l1_ratio > 0:
                    reg_losses += self.l1_ratio * np.abs(W).sum()
                    reg_losses += self.l1_ratio * np.abs(H).sum()
                if self.l1_ratio < 1:
                    reg_losses += 0.5 * (1 - self.l1_ratio) * np.power(W, 2).sum()
                    reg_losses += 0.5 * (1 - self.l1_ratio) * np.power(H, 2).sum()
                dtw_score += self.alpha * reg_losses
            return dtw_score

        # Run the optimizer
        x0 = np.concatenate([W.flatten(), H.flatten()])
        x0 = np.log(np.clip(x0, a_min=1e-6, a_max=None))  # Log to force positive
        result = fmin_powell(loss_function, x0,
                             full_output=True, **self.optimizer_args)
        x_best = np.exp(result[0])
        self.reconstruction_error_ = result[1]

        # Close the process pool
        if pool is not None:
            pool.close()

        # Store the components and optimization results
        self.components_ = np.reshape(x_best[W_size:], H_shape)

        # Return the components
        return np.reshape(x_best[:W_size], W_shape)

    def make_initial_guesses(self, X):
        return _initialize_nmf(X, self.n_components, self.init,
                               random_state=self.random_state)

    def transform(self, X, y):
        raise NotImplementedError('Not sure if I need this. -wardlt')


