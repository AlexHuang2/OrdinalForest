import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from math import floor
from itertools import permutations
from random import sample
import heapq
import threading

from sklearn.ensemble._forest import ForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed
from warnings import warn

from scipy.sparse import issparse

from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble._forest import _partition_estimators

# You might need to implement or import these functions if they're not available
from sklearn.ensemble._forest import (
    _accumulate_prediction,
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
    MAX_INT,
)

class OrdinalForestClassifier(ForestClassifier):
    
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        B_sets=None,
        n_perm=500,
            
    ):
        super().__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        
        # Ordinal- specific variables
        # By default, use 10 times the number of finalized tree constituents to select best partition off of
        self.B_sets = self.n_estimators*10 if B_sets is None or B_sets<self.n_estimators else B_sets
        self.z_mappings = None# map ordinal classes to interval midpoints
        self.z_intervals = None
        self.best_z_mapping = None
        self.best_z_interval = None
        self.ordinal_fitted = None
        self.scoring_function = None
        self.oob_score = 'all-threshold'
        # A hyperparameter for the selection of the next candidate score set
        self.n_perm = n_perm
        
    def _ordinal_z(self,z,z_interval=None):
        z_interval = self.best_z_interval if z_interval is None else z_interval
        return (z.reshape(-1, 1) >= z_interval[:-1]).sum(axis=1) - 1
    def _generate_rankings(self,B_sets,levels,N_perm):
        levels = list(set(levels))
        j = len(levels)
        if B_sets>factorial(j):
            rankings = list(permutations(levels))
            j_rankings = len(rankings)
            rankings = sample(rankings,j_rankings)
            j_copies = floor(B_sets/factorial(j))-1
            rankings *= j_copies + 1
            rankings += sample(rankings,B_sets%j_rankings)
        else:
            rankings = []
            rankings.append(tuple(sample(levels,j)))
            for k in range(1,B_sets):
                candidate_rankings = sample(list(permutations(levels)),N_perm)
                candidate_permutation = sorted(candidate_rankings, key=lambda X: sum((x0-x)**2 for x0,x in zip(rankings[-1],X)))[-1]
                rankings.append(candidate_permutation)
        return rankings
    # Can we optimize this by sampling differently?
    def _generate_z_intervals(self,rankings):
        j = len(rankings[0])
        B_sets = len(rankings)
        rankings = np.array([np.array(ranking) for ranking in rankings]) - 1
        endpoints = np.hstack([np.zeros((B_sets,1)),np.ones((B_sets,1)),np.random.uniform(0,1,size=(B_sets,j-1))])
        interval_lengths = np.diff(np.sort(endpoints),axis=1)
        intervals = np.zeros((B_sets,j+1))
        intervals[:,1:] = np.cumsum((-np.sort(-interval_lengths))[np.arange(B_sets)[:,np.newaxis],rankings],axis=1)
        return intervals
    @staticmethod
    def _get_oob_predictions(tree, X):
        """Use the Regressor version instead.
    
        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
        y_pred = tree.predict(X, check_input=False)
        if y_pred.ndim == 1:
            # single output regression
            y_pred = y_pred[:, np.newaxis, np.newaxis]
        else:
            # multioutput regression
            y_pred = y_pred[:, np.newaxis, :]
        return y_pred


    def predict_proba(self, X):
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        # from random import sample
        # display(X)
        # display(set(sample(list(self.estimators_),1)[0].predict(X)))
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction
                   )(lambda X, *args, **kwargs: self.n_classes_ordinal_encode_[self._ordinal_z(e.predict(X))], X, all_proba, lock)
            for e in self.estimators_
        )

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba
        
    def fit(self, X, y, sample_weight=None):
        # Some of these can be moved under "if not self.ordinal_fitted:" but keep here just to be safe for now
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
            
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            force_all_finite=False,
        )
        
        
# =============================================================================
#         # _compute_missing_values_in_feature_mask checks if X has missing values and
#         # will raise an error if the underlying tree base estimator can't handle missing
#         # values. Only the criterion is required to determine if the tree supports
#         # missing values.
#         estimator = type(self.base_estimator)(criterion=self.criterion)
#         missing_values_in_feature_mask = (
#             estimator._compute_missing_values_in_feature_mask(
#                 X, estimator_name=self.__class__.__name__
#             )
#         )
# =============================================================================
    
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
    
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()
    
        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )
        
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
    
        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )
    
        self._n_samples, self.n_outputs_ = y.shape
    
        y, expanded_class_weight = self._validate_y_class_weight(y)
    
        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
    
        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight
    
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None
    
        self._n_samples_bootstrap = n_samples_bootstrap
    
        self._validate_estimator()
    
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")
    
        random_state = check_random_state(self.random_state)
        
        if not self.ordinal_fitted:
            
            assert hasattr(self, "classes_") and self.n_outputs_ == 1, "y should be univariate."
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            
            assert np.isin(y,np.arange(self.n_classes_)).all(), "Make sure y is univariate and ordered 0 to n-1, for the n classes in the sample." 
            rankings = self._generate_rankings(self.B_sets,self.classes_,self.n_perm)
            intervals = self._generate_z_intervals(rankings)
            self.z_intervals = norm.ppf(intervals)
            self.z_mappings = norm.ppf((intervals[:,:-1]+intervals[:,1:])/2)
            
            self.n_classes_ordinal_encode_ = np.eye(self.n_classes_)
            trees = [
                    self._make_estimator(append=False, random_state=random_state)
                    for i in range(self.B_sets)
                ]
            trees = Parallel(
                        n_jobs=self.n_jobs,
                        verbose=self.verbose,
                        prefer="threads",
                    )(
                        delayed(_parallel_build_trees)(
                            t,
                            self.bootstrap,
                            X,
                            z_mapping[y.squeeze().astype(int)],
                            sample_weight,
                            i,
                            len(trees),
                            verbose=self.verbose,
                            class_weight=self.class_weight,
                            n_samples_bootstrap=n_samples_bootstrap# ,
                            # missing_values_in_feature_mask=missing_values_in_feature_mask,
                        )
                        for i, (t, z_mapping) in enumerate(zip(trees, self.z_mappings))
                    )
            self.B_estimators_ = trees
            trees_pq = []
            min_top_score = None
            for estimator,z_interval,interval in zip(self.B_estimators_,self.z_intervals,intervals):
                unsampled_indices = _generate_unsampled_indices(
                    estimator.random_state,
                    self._n_samples,
                    n_samples_bootstrap,
                )
                z_pred = self._get_oob_predictions(estimator, X[unsampled_indices, :])
                # make sure this is correct
                if self.oob_score=='all-threshold':
                    if not callable(getattr(self, 'all_threshold_loss', None)):
                        def _all_threshold_loss(y,z_pred,z_interval=None):
                            assert not (self.best_z_interval is None and z_interval is None), "Please either specify interval or fit to obtain best interval first"
                            z_interval = self.best_z_interval if z_interval is None else z_interval
                            return np.maximum(0,
                                              1+np.multiply(np.where(y[unsampled_indices, np.newaxis] < self.classes_, -1, 1),
                                                            z_interval[:-1].reshape(1, -1) - z_pred.reshape(-1, 1))
                                             ).sum(axis=1)
                        self.all_threshold_loss = _all_threshold_loss
                    score = self.all_threshold_loss(y.squeeze(),z_pred.squeeze(),z_interval).mean()
                else:
                    if self.oob_score is None:
                        self.oob_score = accuracy_score
                    y_pred = self._ordinal_z(z_pred,z_interval)
                    score = self.oob_score(y, y_pred)
                if len(trees_pq) < self.n_estimators:
                    heapq.heappush(trees_pq, (score,interval))
                elif score > min_top_score:
                    heapq.heapreplace(trees_pq, (score,interval))
                min_top_score = trees_pq[0][0]
            best_interval = np.stack([interval for score,interval in trees_pq]).mean(axis=0)
            self.best_z_interval = norm.ppf(best_interval)
            self.best_z_mapping = norm.ppf((best_interval[:-1] + best_interval[1:]) / 2)
            self.ordinal_fitted = True
            self.fit(X,y.ravel(),sample_weight)
        else:
            assert not self.warm_start, "warm_start not implemented yet for Ordinal Forest."
            if not self.warm_start or not hasattr(self, "estimators_"):
                # Free allocated memory, if any
                self.estimators_ = []
    
            n_more_estimators = self.n_estimators - len(self.estimators_)
    
            if n_more_estimators < 0:
                raise ValueError(
                    "n_estimators=%d must be larger or equal to "
                    "len(estimators_)=%d when warm_start==True"
                    % (self.n_estimators, len(self.estimators_))
                )
    
            elif n_more_estimators == 0:
                warn(
                    "Warm-start fitting without increasing n_estimators does not "
                    "fit new trees."
                )
            else:
                if self.warm_start and len(self.estimators_) > 0:
                    # We draw from the random state to get the random state we
                    # would have got if we hadn't used a warm_start.
                    random_state.randint(MAX_INT, size=len(self.estimators_))
    
                trees = [
                    self._make_estimator(append=False, random_state=random_state)
                    for i in range(n_more_estimators)
                ]
            trees = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    prefer="threads",
                )(
                    delayed(_parallel_build_trees)(
                        t,
                        self.bootstrap,
                        X,
                        self.best_z_mapping[y.squeeze().astype(int)],
                        sample_weight,
                        i,
                        len(trees),
                        verbose=self.verbose,
                        class_weight=self.class_weight,
                        n_samples_bootstrap=n_samples_bootstrap#,
                        # missing_values_in_feature_mask=missing_values_in_feature_mask,
                    )
                    for i, t in enumerate(trees)
                )
                
            # Collect newly grown trees
            self.estimators_.extend(trees)
    
            if self.oob_score and (
                n_more_estimators > 0 or not hasattr(self, "oob_score_")
            ):
                y_type = type_of_target(y)
                if y_type == "unknown" or (
                    self._estimator_type == "classifier"
                    and y_type == "multiclass-multioutput"
                ):
                    # FIXME: we could consider to support multiclass-multioutput if
                    # we introduce or reuse a constructor parameter (e.g.
                    # oob_score) allowing our user to pass a callable defining the
                    # scoring strategy on OOB sample.
                    raise ValueError(
                        "The type of target cannot be used to compute OOB "
                        f"estimates. Got {y_type} while only the following are "
                        "supported: continuous, continuous-multioutput, binary, "
                        "multiclass, multilabel-indicator."
                    )
    
                if callable(self.oob_score):
                    self._set_oob_score_and_attributes(
                        X, y, scoring_function=self.oob_score
                    )
                else:
                    self._set_oob_score_and_attributes(X, y)
    
            # Decapsulate classes_ attributes
            if hasattr(self, "classes_") and self.n_outputs_ == 1:
                self.n_classes_ = self.n_classes_[0]
                self.classes_ = self.classes_[0]
                
            return self