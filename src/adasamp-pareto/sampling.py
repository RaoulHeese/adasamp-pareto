# -*- coding: utf-8 -*-
"""Adaptive sampling: Adaptive optimization algorithm for black-box 
multi-objective optimization problems with binary constraints on the 
foundation of Bayes optimization.

Algorithm from the paper "Adaptive Sampling of Pareto Frontiers with Binary 
Constraints Using Regression and Classification" authored by Raoul Heese and 
Michael Bortz, Fraunhofer Center for Machine Learning, Fraunhofer Institute 
for Industrial Mathematics ITWM (2020). Preprint available on arXiv:
https://arxiv.org/abs/2008.12005

Author: Raoul Heese 

Created on Thu Aug 21 12:00:00 2020
"""


__version__ = "1.1"


from abc import ABC, abstractmethod
from itertools import product, combinations
from scipy import optimize
from scipy.special import erf
import numpy as np
import datetime
import time
try: # Ensure minimal package dependency: install pathos to enable parallel computing.
    import dill
    dill._dill._reverse_typemap['ClassType'] = type
    from pathos.multiprocessing import ProcessingPool
    __POOL_AVAILABLE__ = True
except ModuleNotFoundError:
    __POOL_AVAILABLE__ = False
__LOGGING_ENABLED__ = True # Set to False to disable logging and use print instead.
if __LOGGING_ENABLED__:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
def log_wrapper(verbose, level, msg):
    """Wrapper to savely log messages.
     
    Use the ``logging`` module if enabled and use ``print`` otherwise.
    
    Parameters
    ----------
    
    verbose : bool
        Set to True to log messages.
        
    level : int
        Logging level for the ``logging`` module.
    
    msg : str
        Actual message to log.
    """

    if verbose:
        try:
            if __LOGGING_ENABLED__:      
                logging.log(int(level), str(msg))
            else:
                print(str(msg))
        except Exception as e:
            print(e)


class RegressionModel(ABC):
    """Regression model wrapper.
    
    Estimator for the adaptive sampling regression problem of predicting 
    ``Y`` (goals) from ``X`` (features)."""
    
    def __init__(self):
        pass
        
    @abstractmethod
    def fit(self, X, Y):
        """Train estimator.
        
        Is called at least once before any prediction.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_samples, X_dim)
            Array of features (estimator inputs).
            
        Y : ndarray of shape (n_samples, Y_dim)
            Array of regression values (estimator targets).
        """
        
        raise NotImplementedError            
        
    @abstractmethod
    def predict(self, X, return_std):
        """Estimator prediction.
         
        It is assumed that the predictive distribution is a Gaussian specified 
        by a mean and a standard deviation.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_query_points, X_dim)
            Query points where the estimator is evaluated.
        
        return_std : bool
            If set to True, return both the means and the standard deviations.
            If set to False, only return the means.
            
        Returns
        -------
        
        Y_mu : ndarray of shape (n_query_points, Y_dim)
            Mean of the predictive distribution at the query points.
            
        Y_sigma : ndarray of shape (n_query_points, Y_dim)
           Standard deviation of the predictive distribution at the query 
           points. Covariances are not returned.
        """
        
        raise NotImplementedError  
        
        
class ClassificationModel(ABC):
    """Classification model wrapper.
    
    Estimator for the adaptive sampling classification problem of predicting 
    ``f`` (binary feasibilities) from ``X`` (features)."""
    
    def __init__(self):
        pass
        
    @abstractmethod
    def fit(self, X, f):
        """Train estimator.
        
        Is called at least once before any prediction. The class labels can be 
        either True or False and are automatically converted to integers 
        before the training.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_samples, X_dim)
            Array of features (estimator inputs).
            
        f : ndarray of shape (n_samples,)
            Array of binary classification labels (estimator targets).
        """
        
        raise NotImplementedError            
        
    @abstractmethod
    def predict(self, X):
        """Classifier prediction.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_query_points, X_dim)
            Query points where the estimator is evaluated.
            
        Returns
        -------
        
        f : ndarray of shape (n_query_points,)
            Predicted labels at the query points.
        """
        
        raise NotImplementedError   

    @abstractmethod
    def predict_true_proba(self, X):
        """Classifier probability prediction.
        
        Returns the predicted probability of a True label at the respective 
        query points.
        
        Parameters
        ----------
        
        X : ndarray of shape (n_query_points, X_dim)
            Query points where the estimator is evaluated.
            
        Returns
        -------
        
        p : ndarray of shape (n_query_points,)
            Predicted probability for a True label at the query points. It is 
            assumed that ``0 <= p <= 1``.
        """

        raise NotImplementedError   


class AdaptiveSampler():
    """Adaptive sampler.
    
    Parameters
    ----------
    
    simulation_func : callable
        Function calculating the goals and feasibilities for given features.
        Must be of the form ``simulation_func(X, **kwargs)`` and returns a 
        tuple ``(Y, f)``, where ``X`` is an ndarray of shape (n_samples, 
        X_dim) and ``kwargs`` is a dict of any additional fixed parameters 
        needed to completely specify the function. The returned value ``Y`` is 
        an ndarray of shape (n_samples, Y_dim) representing the resulting goal 
        functions and ``f`` is an ndarray of shape (n_samples,) representing 
        the resulting binary feasibilities. The ``kwargs`` parameter is 
        provided when starting the adaptive sampling run via ``sample``. 
    
    X_limits : list of float tuples (pairs)
        Feature space limits given by a list of pairs of lower and upper 
        bounds: ``[ (x1min, x1max), (x2min, x2max), ... ]``. This list also 
        specifies the dimensionality ``X_dim = len(X_limits)`` of the feature 
        space.
    
    Y_ref : list of float
        Goal space reference point of the form ``[ y1max, y2max, ... ]``. All 
        resulting goal function values must be dominated by the reference 
        point or undesired behaviour might occur. This list also specifies 
        the dimensionality ``Y_dim = len(Y_ref)`` of the goal space.
        
    iterations : int
        Number of adaptive sampling iterations.
        
    Y_model : RegressionModel
        Estimator object for the internal regression problem of predicting 
        ``Y`` (goals) from ``X`` (features). See RegressionModel for details.
        
    f_model : ClassificationModel
        Estimator object for the internal classification problem of predicting 
        ``f`` (feasibilities) from ``X`` (features). See ClassificationModel 
        for details.
        
    initial_samples : int, optional (default: 0)
        Number of initial samples to calculate before starting the adaptive 
        sampling loop.
        
    virtual_iterations : int, optional (default: 1)
        Number of virtual adaptive sampling iterations. Specifies the number 
        of suggested samples per adaptive sampling iteration. Must be at least 
        1.
    
    initial_sampling_func : str or callable, optional (default: "random")
        Function suggesting the initial sampling points. Can either be a 
        string or a callable. The string can either be 'random' for uniformly 
        distributed random samples or 'factorial' for a (full or reduced) 
        factorial design of experiments. The callable must be of the form 
        ``initial_sampling_func(initial_samples, X_initial_sample_limits, seed)``,
        where ``initial_samples`` (int) represents the number of initial 
        samples, `X_initial_sample_limits`` (list of tuples) the respective 
        feature space limits and ``seed`` (int) a given random seed.
        
    utility_parameter_options : dict, optional (default: {})
        Set parameters specifying the utility function. If not set, default 
        values are used. The following parameters are available (=default 
        values):
            entropy_weight=1: entropic weight
            optimization_weight=1: optimality weight
            repulsion_weight=1: repulsion weight
            repulsion_gamma=1: repulsion coefficient
            repulsion_distance_func="default": distance function (either 
                "default" or a callable of the form
                ``repulsion_distance_func(x, y)`` returning the scalar 
                distance of two points ``x`` and ``y``.)
            evi_gamma = 1: Pareto volume parameter
            sector_cutoff = 1: Pareto volume cutoff
        
    decision_parameter_options : dict, optional (default: {})
        Set decision specifying the utility function. If not set, default 
        values are used. The following parameters are available (=default 
        values):
            popsize=15: differential evolution setting
            maxiter=1000: differential evolution setting
            tol=.01: differential evolution setting
            atol=.05: differential evolution setting
            polish=True: differential evolution setting
            polish_extratol=.1: differential evolution polishing setting
            polish_maxfun=100: differential evolution polishing setting
            de_workers=-1: number of workers (-1: use all available)
            polish_workers=-1: number of workers (-1: use all available)
           
    X_initial_sample_limits : list of tuples or None, optional (default: None)
        Feature space limits for the initial sampling given by a list of 
        pairs of lower and upper bounds in analogy to ``X_limits``. If set to 
        None, ``X_limits`` is used instead.
        
    callback_func : callable or None, optional (default: None)
        Function which is called after every adaptive sampling iteration and 
        after the inital sampling. Must be of the form 
        ``callback_func(sampler, X, Y, f, iteration)``, where ``sampler`` is 
        the AdaptiveSampler object (self), ``X`` is an ndarray of shape 
        (n_samples, X_dim), ``Y`` is an ndarray of shape (n_samples, Y_dim) 
        and ``f`` is an ndarray of shape (n_samples,) representing all samples 
        until the current iteration given by ``iteration`` (int or None for 
        the inital sampling call). The return value is stored in the ``info`` 
        property. The callback function ignored if set to None.
        
    stopping_condition_func : callable or None, optional (default: None)
        Function evaluating a specified stopping criterion. Must be of the form
        ``stopping_condition_func(X, Y, f)``, where `X`` is an ndarray of 
        shape (n_samples, X_dim), ``Y`` is an ndarray of shape (n_samples, 
        Y_dim) and ``f`` is an ndarray of shape (n_samples,) representing all 
        samples. The function is called at the end of every adaptive sampling 
        iteration. Its return value is used to determine whether the adaptive 
        sampling loop is stopped prematurely before the number of iterations 
        is reached: a True return value leads to a stop. The stopping 
        condition function ignored if set to None.
        
    seed : int or None, optional (default: None)
        Random seed used for all non-deterministic parts of the algorithm. If 
        set to None, an unspecified (pseudo-random) seed is used.
        
    verbose : bool, optional (default: False)
        Set to True to print status messages. Use the ``logging`` module if 
        enabled.
        
    save_memory_flag : bool, optional (default: False)
        Set to True to activate the memory saving mode, which switches to a 
        memory efficient Pareto volume calculation at the cost of a possibly 
        longer runtime.
        
    Properties
    ----------
    
    info : dict
        Current sampling information (statistics etc.) in form of a dictionary.
        
    opt_func : None or callable
        Current optimization function of the form ``opt_func(X, workers)``, 
        where ``X`` is an ndarray of shape (1, X_dim) corresponding to a 
        single sampling point. The property defaults to None if the 
        optimization function has not yet been specified.
    """
    
    def __init__(self, simulation_func, X_limits, Y_ref, iterations, Y_model, f_model, initial_samples=0, virtual_iterations=1, initial_sampling_func="random", utility_parameter_options=dict(), decision_parameter_options=dict(), X_initial_sample_limits=None, callback_func=None, stopping_condition_func=None, seed=None, verbose=False, save_memory_flag=False):
        self._dtype_X = np.float64
        self._dtype_Y = np.float64
        self._dtype_f = np.int64
        self._f_values_dict = {False: int(False), True: int(True)}
        self._simulation_func = simulation_func
        self._X_limits = np.asarray(X_limits, dtype=self._dtype_X).tolist()
        self._Y_ref = np.asarray(Y_ref, dtype=self._dtype_Y).tolist()
        self._iterations = int(iterations)
        self._Y_model = Y_model
        self._f_model = f_model
        self._initial_samples = int(initial_samples)
        self._virtual_iterations = int(virtual_iterations)
        self._initial_sampling_func = initial_sampling_func        
        self._utility_parameter_options = dict(utility_parameter_options)
        self._decision_parameter_options = dict(decision_parameter_options)
        self._X_initial_sample_limits = np.asarray(X_initial_sample_limits, dtype=np.float64).tolist() if X_initial_sample_limits is not None else self._X_limits
        self._callback_func = callback_func
        self._stopping_condition_func = stopping_condition_func
        self._seed = seed
        self._verbose = bool(verbose)
        self._save_memory_flag = bool(save_memory_flag)
        self._default_utility_parameters = dict(entropy_weight=1, optimization_weight=1, repulsion_weight=1, repulsion_gamma=1, repulsion_distance_func="default", evi_gamma=1, sector_cutoff=1)
        self._default_decision_parameters = dict(popsize=15, maxiter=1000, tol=.01, atol=.05, polish=True, polish_extratol=.1, polish_maxfun=100, de_workers=-1, polish_workers=-1)
        self._init_properties()
        self._verify_self()

    @property
    def info(self):
        """Current sampling information."""    
        
        return self._info
    
    @property
    def opt_func(self):
        """Current optimization function."""    
        
        return self._opt_func
    
    def _init_properties(self):
        """Initialize sampler properties (called in ``__init__``)."""
        
        self._info = dict()
        self._opt_func = None
    
    
    def _verify_self(self):
        """Verify certain sampler attributes (called in ``__init__``)."""
        
        if not callable(self._simulation_func):
            raise ValueError("Verification error: simulation_func is not a callable.")
        if np.array(self._X_limits).size != len(self._X_limits)*2:
            raise ValueError("Verification error: invalid X_limits shape.")
        if not isinstance(self._Y_model, RegressionModel):
            raise ValueError("Verification error: Y_model is not a RegressionModel.")
        if not isinstance(self._f_model, ClassificationModel):
            raise ValueError("Verification error: f_model is not a ClassificationModel.")
        if self._virtual_iterations < 1:
            raise ValueError("Verification error: virtual_iterations must be 1 or more.")
        if type(self._initial_sampling_func) is not str and not callable(self._initial_sampling_func):
            raise ValueError("Verification error: initial_sampling_func of invalid type.")
        if np.array(self._X_initial_sample_limits).size != len(self._X_initial_sample_limits)*2 or len(self._X_initial_sample_limits) != len(self._X_limits):
            raise ValueError("Verification error: invalid X_initial_sample_limits shape.")
        if self._callback_func is not None and not callable(self._callback_func):
            raise ValueError("Verification error: callback_func of invalid type.")
        if self._stopping_condition_func is not None and not callable(self._stopping_condition_func):
            raise ValueError("Verification error: stopping_condition_func of invalid type.")
        if self._seed is not None and type(self._seed) is not int:
            raise ValueError("Verification error: seed of invalid type.")
        if type(self._seed) is int and not (self._seed >= 0 and self._seed < np.iinfo(np.int32).max):
            raise ValueError("Verification error: seed not in the valid range [0, {}).".format(np.iinfo(np.int32).max))

    def _convert_X(self, X):
        """Convert feature data into a standard format."""
                
        return np.asarray(X, dtype=self._dtype_X).reshape(-1, self._X_dim)
    
    def _convert_Y(self, Y):
        """Convert goal data into a standard format."""
        
        return np.asarray(Y, dtype=self._dtype_Y).reshape(-1, self._Y_dim)

    def _convert_f(self, f):
        """Convert feasibility data into a standard format."""
        
        return np.asarray(np.vectorize(self._f_values_dict.get)(f.astype(bool)), dtype=self._dtype_f).ravel()
    
    def initial_sampling_random_uniform(self, initial_samples, X_initial_sample_limits, seed):   
        """Create a random initial design of experiments within the given 
        limits of the feature space."""
        
        rng = np.random.RandomState(seed)
        X_init = np.concatenate([rng.uniform(limits[0], limits[1], initial_samples).reshape(initial_samples,1) for limits in X_initial_sample_limits], axis=1)
        return X_init.reshape(-1,self._X_dim)

    def initial_sampling_factorial(self, initial_samples, X_initial_sample_limits, seed):       
        """Create an initial factorial design of experiments. If no full 
        factorial design is possible, a random subsampling is used."""
        
        X_init = []
        suggestions_per_dimension = int((np.ceil(initial_samples**(1/self._X_dim))))
        for x in product(*[np.linspace(limits[0], limits[1], suggestions_per_dimension) for limits in X_initial_sample_limits]):
            X_init.append(x)
        X_init = np.asarray(X_init).reshape(-1,self._X_dim)    
        if X_init.shape[0] > initial_samples:
            rng = np.random.RandomState(seed)
            rng.shuffle(X_init)
            X_init = X_init[:initial_samples,:]
        return X_init
    
    def _is_pareto_efficient(self, costs, exclude_duplicates=True):
        """Determine Pareto-efficient indices of a cost vector. Note: Assume 
        maximization. Exclude duplicates by default."""

        if exclude_duplicates:
            comparison_func = lambda c1,c2: c1>c2
        else:
            comparison_func = lambda c1,c2: c1>=c2
        is_efficient = np.arange(costs.shape[0])
        next_point_index = 0
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(comparison_func(costs,costs[next_point_index]), axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        return is_efficient
    
    def _create_pareto_grid_cached(self, Y_grid_lines, Y_grid_scale, Y_pareto, Y_ref, Y_dim):
        """Create a Pareto grid. Note: Grid functions with best speed but also 
        large memory requirements (grid is fully stored in memory)."""

        # Build grid and grid mask
        grid = np.array(list(product(*Y_grid_lines)), dtype=np.float64) # contains actual grid points
        grid_mask = np.full(grid.shape, False, dtype=np.bool) # grid point domination: true, when dominated
        grid_mask[np.logical_or.reduce([np.all(grid<Y_pareto[idx,:],axis=1) for idx in range(Y_pareto.shape[0])])] = True
        Y_grid = grid.reshape(*[len(Y_grid_lines[d]) for d in range(Y_dim)],Y_dim)
        Y_grid_mask = grid_mask.reshape(*[len(Y_grid_lines[d]) for d in range(Y_dim)],Y_dim)
        Y_grid_size = np.prod([len(line) for line in Y_grid_lines])
    
        # Build functions    
        grid_lens_iterator_list = [range(d-1) for d in Y_grid.shape[:-1]]
        Y_grid_idx_iter_func = lambda: product(*grid_lens_iterator_list)
        Y_grid_map_func = lambda idx: Y_grid[idx]
        Y_grid_dom_func = lambda idx_nodim: np.all(Y_grid_mask[idx_nodim])
        return Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size
        
    def _create_pareto_grid_runtime(self, Y_grid_lines, Y_grid_scale, Y_pareto, Y_ref, Y_dim):
        """Create a Pareto grid. Note: Grid functions with very small memory 
        requirements (store almost nothing in memory). Is also a bit slower."""

        # Build functions
        line_lens_iterator_list = [range(len(line)-1) for line in Y_grid_lines]
        Y_grid_idx_iter_func = lambda: product(*line_lens_iterator_list)
        Y_grid_map_func = lambda idx: Y_grid_lines[idx[-1]][idx[:-1][idx[-1]]]
        Y_grid_dom_func = lambda idx_nodim: np.logical_or.reduce([np.all(np.array([Y_grid_map_func(list(idx_nodim)+[d]) for d in range(Y_dim)]).reshape(1,-1)<Y_pareto[idx,:],axis=1) for idx in range(Y_pareto.shape[0])])
        Y_grid_size = np.prod([len(line) for line in Y_grid_lines])
        return Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size
    
    def _pareto_grid_map_idx_nodim_to_start_idx(self, idx_nodim, d):
        """Helper functions to map indices for a Pareto grid. Map to the start 
        index."""
        
        return tuple(idx for idx in idx_nodim) + (d,)
    
    def _pareto_grid_map_idx_nodim_to_stop_idx(self, Y_dim, idx_nodim, d):
        """Helper functions to map indices for a Pareto grid. Map to the stop 
        index."""
               
        return tuple(np.array(idx_nodim) + np.array([1 if k==d else 0 for k in range(Y_dim)])) + (d,)
    
    def _create_pareto_grid(self, creator_func, Y_pareto, Y_ref, Y_dim, cut_ref_violation, scale=True):
        """Create a non-uniform grid for Pareto volume calculations (Pareto 
        grid). Note: Assume maximization. The resulting grid is asymmetric 
        when ``np.any(Y_ref == Y_pareto)`` due to the exclusion of zero area 
        grid sectors, otherwise it's symmetric."""
            
        # Remarks:
        #
        # Arguments:
        # creator_func: (Y_grid_lines, Y_grid_scale, Y_pareto, Y_ref, Y_dim) -> return values
        # Y_pareto, Y_ref, Y_dim: properties of the grid
        # cut_ref_violation: flag, defines whether points lower than Y_ref are cut off or an exception is raised
        #
        # Return values:
        # Y_grid_idx_iter_func: () -> iterator. Iterate over all possible idx_nodim = [line1 index, ... linen index]
        # Y_grid_map_func: (idx) -> float. Extract grid point from index: idx -> Y_grid[idx], where idx = [line1 index, ... linen index, Y index]
        # Y_grid_dom_func: (idx) -> float. Check if grid point is dominated (=True): idx_nodim -> np.all(Y_grid_mask[idx_nodim]), where idx_nodim = [line1 index, ... linen index]
        # Y_grid_scale: np.array of floats defining the grid axis scales
        # Y_grid_size: size of grid (i.e., number of elements in grid)"""

        # Prepare
        Y_pareto = np.asarray(Y_pareto)
        Y_ref = np.asarray(Y_ref)
        if Y_dim != Y_ref.size or (Y_dim != Y_pareto.shape[1] and Y_pareto.size > 0):
            raise Exception("Invalid dimensions: Y_dim = {}, Y_ref.shape = {}, Y_pareto.shape = {}!".format(Y_dim, Y_ref.shape, Y_pareto.shape))
        if np.any(Y_ref>Y_pareto):
            if cut_ref_violation:
                for p in range(Y_pareto.shape[0]):
                    Y_pareto[p,:][Y_ref>Y_pareto[p,:]] = Y_ref[Y_ref>Y_pareto[p,:]] # cut all violation points to the correct Y_ref limits
            else:
                raise Exception("Invalid reference point: {} > {}, but Y_ref <= Y_pareto required!".format(Y_ref, Y_pareto[np.where(np.any(Y_ref>Y_pareto,axis=1))]))
           
        # Non-vanishing pareto set
        if Y_pareto.size > 0:
            # Rescaling
            Y_max = np.max(Y_pareto,axis=0)
            if scale:
                Y_grid_scale = Y_max-Y_ref
                Y_grid_scale[Y_grid_scale == 0] = 1 # Fall back to unit scaling where we hit the reference point (this should be a very rare case)
                Y_pareto = Y_pareto / Y_grid_scale
                Y_max = Y_max / Y_grid_scale
                Y_ref = Y_ref / Y_grid_scale
            else:
                Y_grid_scale = np.ones(Y_dim) # use unit scaling
    
            # Create point cloud
            Y_set = np.concatenate((Y_pareto,Y_ref.reshape(1,-1),Y_max.reshape(1,-1)))
         
        # Vanishing pareto set (is not used since Y_mu, Y_sigma cannot be retrieved without data)
        else:
            # Rescaling: use unit scaling
            Y_grid_scale = np.ones(Y_dim)
            
            # Create point cloud
            Y_set = np.copy(Y_ref.reshape(1,-1))
            
        # Create sorted edge points
        Y_dset = []
        for d in range(Y_dim):
            Y_dset.append(Y_set[Y_set[:,d].argsort()][:,d])
        
        # Create lines
        Y_grid_lines = [[] for _ in range(Y_dim)]
        for idx in range(Y_set.shape[0]):
            for d in range(Y_dim):
                if len(Y_grid_lines[d]) == 0 or Y_grid_lines[d][-1] < Y_dset[d][idx]:
                    Y_grid_lines[d].append(Y_dset[d][idx])
        for d in range(Y_dim):
            Y_grid_lines[d].append(np.inf)   
        if np.any([len(line) < 2 for line in Y_grid_lines]):
            raise Exception("Invalid Y_grid_lines with shape = {}!".format([len(line) for line in Y_grid_lines]))        
        
        # Compile and return functions
        if creator_func is not None:
            return creator_func(Y_grid_lines, Y_grid_scale, Y_pareto, Y_ref, Y_dim)
        else:
            return Y_grid_lines, Y_grid_scale, Y_pareto, Y_ref, Y_dim
       
    def _expected_non_dominated_volume_improvement(self, Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size, Y_mu, Y_sigma, allow_outliers=True, workers=1):
        """Calculate the expected improvement of the non-dominated Pareto 
        volume."""
        
        # Define helper functions
        theta0 = .5
        def gaussian_integral_linear(mu, sigma, a, b, ref):
            result = np.zeros(mu.shape)
            mu0 = mu[sigma==0]
            mu1 = mu[sigma!=0]
            sigma1 = sigma[sigma!=0]
            if np.isposinf(b):
                result[sigma==0] = (mu0-ref) * np.heaviside(mu0-a, theta0)
                result[sigma!=0] = (ref-mu1)/2 * ( erf((a-mu1)/(np.sqrt(2)*sigma1))-1 ) + sigma/np.sqrt(2*np.pi)*( np.exp(-(a-mu1)**2/(2*sigma1**2)) )
            else:
                result[sigma==0] = (mu0-ref) * np.heaviside(b-mu0, theta0) * np.heaviside(mu0-a, theta0)
                result[sigma!=0] = (ref-mu1)/2 * ( erf((a-mu1)/(np.sqrt(2)*sigma1))-erf((b-mu1)/(np.sqrt(2)*sigma1)) ) + sigma1/np.sqrt(2*np.pi)*( np.exp(-(a-mu1)**2/(2*sigma1**2)) - np.exp(-(b-mu1)**2/(2*sigma1**2)) ) # TODO: invalid value encountered in multiply
            return result
        def gaussian_integral_const(mu, sigma, a, b):
            result = np.zeros(mu.shape)
            mu0 = mu[sigma==0]
            mu1 = mu[sigma!=0]
            sigma1 = sigma[sigma!=0]        
            result[sigma==0] = np.heaviside(b-mu0, theta0) * np.heaviside(mu0-a, theta0)
            result[sigma!=0] = 1/2 * ( erf((b-mu1)/(np.sqrt(2)*sigma1))-erf((a-mu1)/(np.sqrt(2)*sigma1)) ) # TODO: invalid value encountered in true_divide
            return result
        
        # Setup
        if self._Y_dim != Y_mu.shape[1] or Y_grid_scale.size != Y_mu.shape[1] or Y_mu.shape[1] != Y_sigma.shape[1]:
            raise Exception("Invalid dimensions: Y_dim = {}, Y_grid_scale.size = {}, Y_mu.shape = {}, Y_sigma.shape = {}!".format(self._Y_dim, Y_grid_scale.size, Y_mu.shape, Y_sigma.shape))
        num_points = Y_mu.shape[0]
        Y_mu = Y_mu / Y_grid_scale
        Y_sigma = Y_sigma / Y_grid_scale
        sector_hit_sigma = self._utility_parameters['sector_cutoff']
        
        # Loop function
        def sector_vol(sector_idx_array_nodim):       
            # Volume definition for this sector
            sector_vol = np.zeros(num_points,dtype=np.float64)
            if Y_grid_dom_func(sector_idx_array_nodim):
                return sector_vol # sector is dominated
    
            # Calculate sectors
          
            # 0) Prepare sector
            a_sector = []
            b_sector = []
            for d in range(self._Y_dim):
                idx_array_start = self._pareto_grid_map_idx_nodim_to_start_idx(sector_idx_array_nodim, d)
                idx_array_stop = self._pareto_grid_map_idx_nodim_to_stop_idx(self._Y_dim, sector_idx_array_nodim, d)
                a_sector.append(Y_grid_map_func(idx_array_start))
                b_sector.append(Y_grid_map_func(idx_array_stop))  
            
            # 1) New method: check sector contribution and skip if neglectable
            if sector_hit_sigma > 0:
                sector_distances = np.zeros((num_points,self._Y_dim),dtype=np.float32) # distance in each dimension
                for d in range(self._Y_dim):
                    outlier_idx = np.logical_or(a_sector[d] > Y_mu[:,d],Y_mu[:,d] > b_sector[d]) # lies not within a and b limits
                    border_distances = np.concatenate((np.abs(a_sector[d] - Y_mu[:,d]).reshape(-1,1), np.abs(b_sector[d] - Y_mu[:,d]).reshape(-1,1)),axis=1)
                    border_distance = np.min(border_distances,axis=1)
                    sector_distances[outlier_idx,d] = border_distance[outlier_idx]
                #sector_hit_idx = np.all(sector_hit_sigma*Y_sigma > sector_distances, axis=1) # alternative box method
                Y_sigma[Y_sigma==0] = np.inf # handle division by zero; TODO: verify this approach
                sector_hit_idx = np.linalg.norm(sector_distances/Y_sigma,axis=1,ord=self._Y_dim) < sector_hit_sigma # ellipsoid method
                if not np.any(sector_hit_idx):
                    return sector_vol # skip sector if no Y_mu+-Y_sigma hits the sector (Note: this introduces an uncertainty)
            
            # 2) Active sector calculation
            active_sector_vol = np.ones(num_points,dtype=np.float64)
            for d in range(self._Y_dim):
                pv = gaussian_integral_linear(Y_mu[:,d], Y_sigma[:,d], a_sector[d], b_sector[d], a_sector[d])
                active_sector_vol *= pv
            sector_vol += active_sector_vol
            
            # 3) Subsectors calculation
            for sub_sector_idx_array_nodim in Y_grid_idx_iter_func():  
                if Y_grid_dom_func(sub_sector_idx_array_nodim) or np.any(np.array(sub_sector_idx_array_nodim)>np.array(sector_idx_array_nodim)) or np.all(np.array(sub_sector_idx_array_nodim)==np.array(sector_idx_array_nodim)):          
                    continue # subsector is either dominated or not a subsector in the first place
                active_sector_vol = np.ones(num_points,dtype=np.float64)
                for d in range(self._Y_dim):
                    idx_array_start = self._pareto_grid_map_idx_nodim_to_start_idx(sub_sector_idx_array_nodim, d)
                    a = Y_grid_map_func(idx_array_start)
                    if sub_sector_idx_array_nodim[d] == sector_idx_array_nodim[d]:
                        pv = gaussian_integral_linear(Y_mu[:,d], Y_sigma[:,d], a_sector[d], b_sector[d], a)
                    else: 
                        idx_array_stop = self._pareto_grid_map_idx_nodim_to_stop_idx(self._Y_dim, sub_sector_idx_array_nodim, d)
                        b = Y_grid_map_func(idx_array_stop)                    
                        pv = (b - a) * gaussian_integral_const(Y_mu[:,d], Y_sigma[:,d], a_sector[d], b_sector[d])
                    active_sector_vol *= pv
                sector_vol += active_sector_vol   
                
            # Return gained sector volume
            return sector_vol

        # Check for outliers (i.e. predicted expecation value worse than Y_ref) if requested
        if not allow_outliers:
            Y_ref = self._Y_ref / Y_grid_scale
            if np.any(Y_mu<Y_ref):
                raise Exception("Invalid prediction (Y_mu < Y_ref): {} < {} (with rescaling {})!".format(Y_mu * Y_grid_scale, Y_ref * Y_grid_scale, Y_grid_scale))
    
        # Calculate volume by traversing each sector of the grid using the loop function
        # Use either sequential or parallel computing
        if workers == 1 or not __POOL_AVAILABLE__:
            sector_vol_list = np.zeros((Y_grid_size,num_points), dtype=np.float64)
            for grid_idx, sector_idx_array_nodim in enumerate(Y_grid_idx_iter_func()):
                sector_vol_list[grid_idx,:] = sector_vol(sector_idx_array_nodim)
        else:
            pool = ProcessingPool(None if workers == -1 else workers)
            sector_vol_list = pool.map(sector_vol, Y_grid_idx_iter_func())
        non_dominated_volume = np.sum(sector_vol_list, axis=0, dtype=np.float64)
            
        return non_dominated_volume
    
    def _total_dominated_volume(self, Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size, workers=1):
        """Calculate the total Pareto-dominated volume. Note: Return total 
        unscaled dominated volume (i.e., with removed scaling)."""
  
        # Loop function
        def sector_dom_vol(idx_array_nodim):
            if Y_grid_dom_func(idx_array_nodim): # sector only contributes if it is dominated
                sector_dominated_volume = 1
                for d in range(self._Y_dim):
                    idx_array_start = self._pareto_grid_map_idx_nodim_to_start_idx(idx_array_nodim, d)
                    idx_array_stop = self._pareto_grid_map_idx_nodim_to_stop_idx(self._Y_dim, idx_array_nodim, d)
                    a = Y_grid_map_func(idx_array_start)
                    b = Y_grid_map_func(idx_array_stop)
                    sector_dominated_volume *= (b-a)
                return sector_dominated_volume
            return 0
        
        # Calculate total dominanted volume by traversing each sector of the grid using the loop function
        # Use either sequential or parallel computing
        if workers == 1 or not __POOL_AVAILABLE__:
            sector_dom_vol_list = np.zeros(Y_grid_size, dtype=np.float64)
            for grid_idx, idx_array_nodim in enumerate(Y_grid_idx_iter_func()):
                sector_dom_vol_list[grid_idx] = sector_dom_vol(idx_array_nodim)
        else:
            pool = ProcessingPool(None if workers == -1 else workers)
            sector_dom_vol_list = pool.map(sector_dom_vol, Y_grid_idx_iter_func())
        total_dominated_volume = np.sum(sector_dom_vol_list, axis=0, dtype=np.float64)
            
        # Rescale and return result
        return total_dominated_volume * np.prod(Y_grid_scale)
    
    def _pareto_not_dominated_probability(self, Y_pareto, Y_mu, Y_sigma, workers=1):
        """Calculate probability of being not Pareto-dominated. Note: Assume 
        maximization. Assume a ``Y_estimator`` with Gaussian probability 
        distribution."""
        
        # Define helper functions
        theta0 = .5
        def p_nd(mu, sigma, y):
            result = np.zeros(mu.shape)
            mu0 = mu[sigma==0]
            mu1 = mu[sigma!=0]
            sigma1 = sigma[sigma!=0]        
            result[sigma==0] = np.heaviside(y-mu0,theta0)
            result[sigma!=0] = (1+erf((mu1-y)/(np.sqrt(2)*sigma1)))/2
            return result
        def p_d(mu, sigma, y):
            result = np.zeros(mu.shape)
            mu0 = mu[sigma==0]
            mu1 = mu[sigma!=0]
            sigma1 = sigma[sigma!=0]        
            result[sigma==0] = np.heaviside(mu0-y,theta0)
            result[sigma!=0] = (1-erf((mu1-y)/(np.sqrt(2)*sigma1)))/2
            return result
        
        # Setup
        num_points = Y_mu.shape[0]    
        num_pareto = Y_pareto.shape[0]
        if workers != 1 and __POOL_AVAILABLE__:
            pool = ProcessingPool(None if workers == -1 else workers)
            
        # Loop function core
        def probability_sum(index_list, probability_d, probability_nd):
            return np.prod([probability_d[:,-i-1] if i < 0 else probability_nd[:,i-1] for i in index_list],axis=0)
        
        # Non-vanishing pareto set
        if num_pareto > 0:
            probability_list = np.zeros((num_pareto,num_points), dtype=np.float64)
            index_lists = [x for x in combinations(np.concatenate((np.linspace(1,self._Y_dim,self._Y_dim,dtype=np.int32),np.linspace(-1,-self._Y_dim,self._Y_dim,dtype=np.int32))), r=self._Y_dim) if np.any(np.array(x)>0) and np.unique(np.abs(x)).size == len(x)]
            for y_idx in range(num_pareto):
                probability_nd = np.zeros((num_points, self._Y_dim))
                probability_d = np.zeros((num_points, self._Y_dim))
                for i in range(self._Y_dim):
                    probability_nd[:,i] = p_nd(Y_mu[:,i], Y_sigma[:,i], Y_pareto[y_idx,i])
                    probability_d[:,i] = p_d(Y_mu[:,i], Y_sigma[:,i], Y_pareto[y_idx,i])
                if workers == 1 or not __POOL_AVAILABLE__:
                    p_sum_list = np.zeros((len(index_lists),num_points), dtype=np.float64)
                    for idx, index_list in enumerate(index_lists):
                        p_sum_list[idx,:] = probability_sum(index_list, probability_d, probability_nd)
                else:
                    loop_fun = lambda index_list: probability_sum(index_list, probability_d, probability_nd)
                    p_sum_list = pool.map(loop_fun, index_lists)
                probability_list[y_idx,:] = np.sum(p_sum_list, axis=0, dtype=np.float64)
            probability = np.prod(probability_list, axis=0, dtype=np.float64)
                    
        # Vanishing pareto set (is not used since Y_mu, Y_sigma cannot be retrieved without data)
        else:
            probability = np.ones(num_points)
        
        return probability
        
    def _probability_feasible(self, X):
        """Prediction of the feasibility probability using the previously 
        trained classifier."""
        
        return self._f_model.predict_true_proba(X)
      
    def _optutility_func(self, X, X_r_explored_scaled, Y_pareto, Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size, repulsion_transformation_func, repulsion_distance_func, workers=1): 
        """Base function for the utility calculation. Note: Assume 
        maximization. Assume a ``Y_estimator`` with Gaussian probability 
        distribution. Assume a single sample ``X``."""
        
        # options
        tol = 1e-5
        allow_outliers = True
        catch_nan_input = True
        
        # setup
        X = np.asarray(X).reshape(1,self._X_dim)
        if catch_nan_input and np.any(np.isnan(X)):
            return np.array(0).ravel()
        num_points = X.shape[0]
        s = self._utility_parameters['entropy_weight']
        o = self._utility_parameters['optimization_weight']
        r = self._utility_parameters['repulsion_weight']
        evi_gamma = self._utility_parameters['evi_gamma']
        
        # clipping wrapper function
        def clip(value, tol, name):
            if np.any(value < 0-tol) or np.any(value > 1+tol):
                np.set_printoptions(suppress = True, precision = 10)
                raise Exception("incorrect value for '{}': {} +- {}.".format(name, value, tol))
            return np.clip(value, 0, 1)
        
        # calculate
        if self._Y_model_is_ready:
            try:
                Y_mu, Y_sigma = self._Y_model.predict(X, return_std=True)
                Y_mu = Y_mu.reshape(1,self._Y_dim)
                Y_sigma = Y_sigma.reshape(1,self._Y_dim)
            except Exception as e:
                raise Exception("Estimator cannot predict mu, sigma: '{}'.".format(e)) 
            if o != 0:
                expected_vol = self._expected_non_dominated_volume_improvement(Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size, Y_mu, Y_sigma, allow_outliers=allow_outliers, workers=workers) # [0,inf]
            else:
                expected_vol = np.zeros(num_points) # values not needed
            if s != 0 or r != 0:
                p_not_dominated = self._pareto_not_dominated_probability(Y_pareto, Y_mu, Y_sigma, workers=workers) # [0,1]
            else:
                p_not_dominated = np.zeros(num_points) # values not needed
        else:
            expected_vol = np.zeros(num_points) # no improvement without prediction
            p_not_dominated = np.ones(num_points) # nothing is dominated
        if evi_gamma is not None:
            expected_vol_gain = 1 - np.exp(-evi_gamma * np.clip(expected_vol,0,np.inf)) # [0,1]
        else:
            expected_vol_gain = np.clip(expected_vol,0,np.inf) # DEBUG [0,inf]
        try:
            p_feasible = self._probability_feasible(X) # [0,1]
            entropy = -np.sum([p * np.log(p) if (p > 0) else 0.0 for p in p_feasible]) / np.log(2) # [0,1]
        except:
            p_feasible = 1 # assume total feasibility if no prediction is available
            entropy = .5
        if r != 0:
            if X_r_explored_scaled.size > 0:
                X_r_scaled = repulsion_transformation_func(X)
                min_distance = np.array([np.min(repulsion_distance_func(X_r_scaled[idx,:],X_r_explored_scaled)) for idx in range(num_points)]) # [0,1]
                repulsion = min_distance  #[0,1]
            else:
                repulsion = 1 # no distance available
        else:
            repulsion = 1 # values not needed
        S = entropy * p_not_dominated # [0,1]
        O = expected_vol_gain * p_feasible # [0,1]
        R = repulsion * p_not_dominated # [0,1]
        
        # finalize results
        S = clip(S, tol, "S")
        O = clip(O, tol, "O")
        R = clip(R, tol, "R")
        utility = (s * S + o * O + r * R) / (s+o+r)
        return np.array(utility).ravel() # [0,1]
    
    def _opt_func_provider(self, X, Y, f, X_virtual, Y_virtual, f_virtual):
        """Provide a utility function for the adaptive sampling decision."""
        
        # Options
        cut_ref_violation = True # set to False for debugging
        grid_scaling = True
        
        # Prepare
        data_min = np.array([limits[0] for limits in self._X_limits])
        data_max = np.array([limits[1] for limits in self._X_limits])
        feature_range = [0, 1]
        X_r_scale = (feature_range[1] - feature_range[0]) / (data_max - data_min)
        X_r_shift = feature_range[0] - data_min * X_r_scale
        def repulsion_transformation_func(X):
            return X * X_r_scale + X_r_shift        
        if X_virtual.size > 0:
            X_explored = np.concatenate((X, X_virtual))
        else:
            X_explored = X
        if X_explored.size > 0:
           X_r_explored_scaled = repulsion_transformation_func(X_explored)
        else:
           X_r_explored_scaled = np.array([], dtype=self._dtype_X).reshape(0,self._X_dim)
        if Y_virtual.size > 0:
            Y_total = np.concatenate((Y, Y_virtual))
        else:
            Y_total = Y
        if f_virtual.size > 0:
            f_total = np.concatenate((f, f_virtual))
        else:
            f_total = f
        Y_feasible = Y_total[f_total!=self._f_values_dict[False]]
        if Y_feasible.size > 0:
            Y_pareto = Y_feasible[self._is_pareto_efficient(Y_feasible),:]
            Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size = self._create_pareto_grid(self._grid_creator_func, Y_pareto, self._Y_ref, self._Y_dim, cut_ref_violation=cut_ref_violation, scale=grid_scaling)
        else:
            Y_pareto = np.array([], dtype=self._dtype_Y).reshape(0,self._Y_dim)
            Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size = None, None, None, None, None # not used
        repulsion_gamma = self._utility_parameters['repulsion_gamma']
        if callable(self._utility_parameters['repulsion_distance_func']):
            repulsion_distance_func = self._utility_parameters['repulsion_distance_func']
        elif self._utility_parameters['repulsion_distance_func'] == "default":
            min_limits_point_scaled = repulsion_transformation_func(np.min(self._X_limits,axis=1).reshape(1,self._X_dim))
            max_limits_point_scaled = repulsion_transformation_func(np.max(self._X_limits,axis=1).reshape(1,self._X_dim))
            maximum_distance = np.linalg.norm(min_limits_point_scaled-max_limits_point_scaled)**2
            def default_repulsion_distance_func(x, y):
                # note: distance funct for step (2), where: (1) perform scaling of x and y with X_r_scaler, (2) mapping: 1 - e(-gamma||x-y||^2) with scaled x and scaled y    
                return (1 - np.exp(-repulsion_gamma*np.linalg.norm(x-y,axis=-1)**2)) / (1 - np.exp(-repulsion_gamma*maximum_distance)) #[0,1]
            repulsion_distance_func = default_repulsion_distance_func
        else:
            raise ValueError("repulsion distance function unspecified: '{}'".format(self._utility_parameters['repulsion_distance_func']))
           
        # Finish
        opt_func = lambda X, workers: -self._optutility_func(X, X_r_explored_scaled, Y_pareto, Y_grid_idx_iter_func, Y_grid_map_func, Y_grid_dom_func, Y_grid_scale, Y_grid_size, repulsion_transformation_func, repulsion_distance_func, workers) # mind the negative sign for minimization
        return opt_func
    
    def _sampling_decision(self, opt_func):
        """Minimize ``opt_func`` to determine a single adaptive sampling 
        suggestion."""
        
        # TODO: provide alternative optimization strategies
        
        bounds = [(l, u) for (l, u) in zip([limits[0] for limits in self._X_limits], [limits[1] for limits in self._X_limits])]
        seed = self._rng.randint(0, np.iinfo(np.int32).max)
        x_precision = 12 # try to omit rounding errors by fixing the precision of the result (e.g., bounds can be violated due to rounding errors)
        popsize = self._decision_parameters['popsize']
        maxiter = self._decision_parameters['maxiter']
        tol = self._decision_parameters['tol']
        atol = self._decision_parameters['atol']
        polish = self._decision_parameters['polish']
        polish_extratol = self._decision_parameters['polish_extratol']
        polish_maxfun = self._decision_parameters['polish_maxfun']
        workers = self._decision_parameters['de_workers']
        polish_workers = self._decision_parameters['polish_workers']
        if workers == 1 or not __POOL_AVAILABLE__:
            worker_runner = 1
        else:
            worker_runner = ProcessingPool(None if workers == -1 else workers).map
        updating = 'deferred' # makes multi-worker result compatible with single worker result (originally 'immediate' for workers=1)
        args = (1,) # use 1 worker to evaluate opt_func
        np.random.seed(seed)
        opt_result = optimize.differential_evolution(func=opt_func, bounds=bounds, args=args, popsize=popsize, maxiter=maxiter, tol=tol, atol=atol, polish=False, seed=seed, updating=updating, workers=worker_runner)
        if polish:
            polish_method = 'L-BFGS-B'
            polish_options = dict(ftol = polish_extratol * tol, maxfun = polish_maxfun) # default for direct polish from optimize.differential_evolution: ftol = 2.220446049250313e-09, maxfun = 15000
            polished_args = (polish_workers,) # use 1 or more workers to evaluate opt_func
            np.random.seed(seed)
            polished_opt_result = optimize.minimize(fun=opt_func, x0=np.copy(opt_result.x), args=polished_args, method=polish_method, bounds=bounds, options=polish_options)
            opt_result.nfev += polished_opt_result.nfev
            if polished_opt_result.fun < opt_result.fun:
                opt_result.fun = polished_opt_result.fun
                opt_result.x = polished_opt_result.x
                opt_result.jac = polished_opt_result.jac
        else:
            polished_opt_result = None # for consistency in exception only
        x = np.round(np.array(opt_result.x).reshape(1,-1), x_precision)
        bounds = np.array(bounds)
        if not np.any(np.isnan(x)) and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]): # valid if x is not nan and lies within bounds
            return x
        else:
            raise Exception("Sampling decision failed: x = {}, bounds = {}, opt_result = {}, polished_opt_result = {}!".format(x, bounds, opt_result, polished_opt_result))
            
    def _evaluate_simulation(self, X):
        """Evaluate the goal function and the binary feasibility of one or 
        more features ``X`` by executing ``_simulation_func``. Also measure 
        the required calculation time."""
        
        t = time.time()
        Y, f = self._simulation_func(X, **self._kwargs)
        t = time.time() - t
        Y = np.array(Y, dtype=np.float64)
        f = np.array(f, dtype=np.int64)
        Y[np.logical_or(f==self._f_values_dict[False], f==np.nan)] = np.nan
        return self._convert_Y(Y), self._convert_f(f), t
    
    def _update_estimator(self, X, Y, f):
        """Update the internal estimators for the goal function (regressor) 
        and the binary feasibility (classifier).""" 
        
        if self._update_Y_estimator_flag:
            try:
                self._Y_model.fit(X[f!=self._f_values_dict[False]], Y[f!=self._f_values_dict[False]])
                self._Y_model_is_ready = True
            except:
                self._Y_model_is_ready = False
        try:
            self._f_model.fit(X, f)
            self._f_model_is_ready = True
        except:
            self._f_model_is_ready = False
    
    def _estimate_simulation(self, X):
        """Predict the goal function and the binary feasibility of one or more 
        features ``X`` based on the previously trained estimators."""
        
        num_points = X.shape[0]
        try:
            Y = self._Y_model.predict(X, return_std=False).reshape(num_points,-1)
        except:
            Y = np.full((num_points, self._Y_dim), np.nan)
        try:
            f = self._f_model.predict(X).reshape(num_points)
        except:
            f = np.full((num_points), np.nan)
        return self._convert_Y(Y), self._convert_f(f)
    
    def _evaluate_initial_sampling(self, seed):
        """Perform the intital sampling at the start of the adaptive sampling 
        run by executing ``_initial_sampling_func``.""" 
        
        if callable(self._initial_sampling_func):
            X_init = self._initial_sampling_func(self._initial_samples, self._X_initial_sample_limits, seed)
        elif self._initial_sampling_func == "random":
            X_init = self.initial_sampling_random_uniform(self._initial_samples, self._X_initial_sample_limits, seed)
        elif self._initial_sampling_func == "factorial":
            X_init = self.initial_sampling_factorial(self._initial_samples, self._X_initial_sample_limits, seed)       
        else:
            raise ValueError("initial sampling function unspecified: '{}'".format(self._initial_sampling_func))
        return self._convert_X(X_init)
    
    def _evaluate_callback(self, X, Y, f, iteration):
        """Evalutate the callback function by executing ``_callback_func``. 
        The function is evaluated in each iteration step of the adaptive 
        sampling run. Also measure the runtime."""
        
        t = time.time()
        if self._callback_func is not None:
            self.info['callback_results'].append(self._callback_func(self, X, Y, f, iteration))
        t = time.time() - t
        return t        
    
    def _evaluate_stopping_condition(self, X, Y, f):
        """Evalutate the stopping criterion by executing 
        ``_stopping_condition_func``. A positive return value prematurely 
        stops the main adaptive sampling loop. Also measure the runtime."""
        
        t = time.time()
        if self._stopping_condition_func is not None:
            stop_flag = self._stopping_condition_func(X, Y, f)
        else:
            stop_flag = False
        t = time.time() - t
        return stop_flag, t    

    def _initialize_sampling(self, **kwargs):
        """Prepare an adaptive sampling run."""
        
        self._info['start_timestamp'] = datetime.datetime.now().timestamp()
        self._kwargs = kwargs
        self._X_dim = len(self._X_limits)
        self._Y_dim = len(self._Y_ref)
        self._rng = np.random.RandomState(self._seed)
        self._info = dict(simulation_time = 0, callback_time = 0, condition_time = 0, stop_flag = False, evaluation_batches = [], evaluation_batches_time = [], callback_results = [], initial_samples = self._initial_samples, seed = self._seed)
        self._opt_func = None
        self._update_Y_estimator_flag = True
        self._Y_model_is_ready = False
        self._f_model_is_ready = False
        if self._save_memory_flag:
            self._grid_creator_func = self._create_pareto_grid_runtime
        else:
            self._grid_creator_func = self._create_pareto_grid_cached
        self._utility_parameters = self._default_utility_parameters.copy()
        self._utility_parameters.update(**self._utility_parameter_options)
        self._decision_parameters = self._default_decision_parameters.copy()
        self._decision_parameters.update(**self._decision_parameter_options)
        
    def _initial_sampling(self):
        """Perform the initial sampling for an adaptive sampling run."""
        
        log_wrapper(self._verbose, 20, "[initial sampling start]")
        t = time.time()
        seed = self._rng.randint(0, np.iinfo(np.int32).max)
        X = self._evaluate_initial_sampling(seed)
        t = time.time() - t
        self._info['evaluation_batches'].append(X.shape[0])
        self._info['evaluation_batches_time'].append(t)
        Y, f, t_sim = self._evaluate_simulation(X)
        self._info['simulation_time'] += t_sim
        t_callback = self._evaluate_callback(X, Y, f, None)
        self._info['callback_time'] += t_callback
        log_wrapper(self._verbose, 20, "[initial sampling end] new points = {:d}".format(f.size))
        return X, Y, f
        
    def _adaptive_sampling_loop(self, X, Y, f):
        """Run the main adaptive sampling loop, which consists of an outer 
        loop (iterate over all sampling iterations) and an inner loop (iterate 
        over all virtual sampling iterations). In each iteration, the callback 
        function is executed. The outer loop is stopped prematurely as soon as 
        the stopping criterion is positive.
        """
        
        iteration = None
        for iteration in range(self._iterations):
            t = time.time()
            log_wrapper(self._verbose, 20, "[iteration {:d} start] f distribution = {}, total points = {:d}".format(iteration, {f_: f[f==f_].size for f_ in np.unique(f)}, f.size))
            X_virtual = np.array([], dtype=self._dtype_X).reshape(0,self._X_dim)
            Y_virtual = np.array([], dtype=self._dtype_Y).reshape(0,self._Y_dim)
            f_virtual = np.array([], dtype=self._dtype_f).reshape(0)
            self._update_estimator(X, Y, f)
            for virtual_iteration in range(self._virtual_iterations):
                self._opt_func = self._opt_func_provider(X, Y, f, X_virtual, Y_virtual, f_virtual)
                X_suggestion = self._convert_X(self._sampling_decision(self._opt_func))
                X_virtual = np.concatenate((X_virtual, X_suggestion))
                if virtual_iteration < self._virtual_iterations - 1:
                    Y_estimate, f_estimate = self._estimate_simulation(X_suggestion)
                    Y_virtual = np.concatenate((Y_virtual, Y_estimate))
                    f_virtual = np.concatenate((f_virtual, f_estimate))         
            t = time.time() - t
            self._info['evaluation_batches'].append(X_virtual.shape[0])
            self._info['evaluation_batches_time'].append(t)
            Y_sample, f_sample, t_sim = self._evaluate_simulation(X_virtual)
            self._info['simulation_time'] += t_sim
            self._update_Y_estimator_flag = np.any(f_sample)
            X = np.concatenate((X, X_virtual))
            Y = np.concatenate((Y, Y_sample))
            f = np.concatenate((f, f_sample))
            t_callback = self._evaluate_callback(X, Y, f, iteration)
            self._info['callback_time'] += t_callback
            stop_flag, t_stop = self._evaluate_stopping_condition(X, Y, f)
            self._info['condition_time'] += t_stop
            log_wrapper(self._verbose, 20, "[iteration {:d} end] new points = {:d}".format(iteration, f_sample.size))
            if stop_flag:
                self._info['stop_flag'] = True
                break
        self._info['iteration'] = iteration
        return X, Y, f
            
    def _finalize_sampling(self):
       """Finish an adaptive sampling run."""
       
       self._info['end_timestamp'] = datetime.datetime.now().timestamp()
       
    def sample(self, **kwargs):
        """Start sampling.
         
        Perform an adaptive sampling with this sampler and return the sampled 
        results.
        
        Parameters
        ----------
        
        kwargs : dict, optional
            Any additional fixed parameters needed to completely specify 
            ``simulation_func``.
            
        Returns
        -------
        
        X : ndarray of shape (n_samples, X_dim)
            Resulting array of sampled features.
            
        Y : ndarray of shape (n_samples, Y_dim)
            Resulting array of corresponding goals from the simulation.
            
        f : ndarray of shape (n_samples,)
            Resulting array of corresponding binary feasibilities from the 
            simulation.    
        """
        
        self._initialize_sampling(**kwargs)
        X, Y, f = self._initial_sampling()
        X, Y, f = self._adaptive_sampling_loop(X, Y, f)
        self._finalize_sampling()
        return X, Y, f
