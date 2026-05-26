from abc import ABCMeta
from functools import partial
import threading
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import (
    BaseForest,
    _accumulate_prediction,
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
)
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.docstrings import append_cumulative_hazard_example, append_survival_function_example
from sksurv.metrics import concordance_index_censored
from sksurv.util import check_array_survival

from sksurv_importance._criterion import get_unique_times
from sksurv_importance.tree import _array_to_step_function
from sksurv_importance.tree import ExtraSurvivalTree, SurvivalTree
from sksurv_importance._tree import DTYPE

__all__ = ["RandomSurvivalForest", "ExtraSurvivalTrees"]


class _BaseSurvivalForest(BaseForest, metaclass=ABCMeta):
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        spline_importance=None,
    ):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=None,
            max_samples=max_samples,
        )
        self.spline_importance = spline_importance

    def fit(self, X, y, sample_weight=None):
        self._validate_params()

        X = validate_data(self, X, dtype=DTYPE, accept_sparse="csc", ensure_min_samples=2, ensure_all_finite=False)
        event, time = check_array_survival(X, y)

        estimator = type(self.estimator)()
        missing_values_in_feature_mask = estimator._compute_missing_values_in_feature_mask(
            X, estimator_name=self.__class__.__name__
        )

        self._n_samples, self.n_features_in_ = X.shape
        time = time.astype(np.float64)
        self.unique_times_, self.is_event_time_ = get_unique_times(time, event)
        self.n_outputs_ = self.unique_times_.shape[0]

        y_numeric = np.empty((X.shape[0], 2), dtype=np.float64)
        y_numeric[:, 0] = time
        y_numeric[:, 1] = event.astype(np.float64)

        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(n_samples=X.shape[0], max_samples=self.max_samples)
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators > 0:
            trees = [self._make_estimator(append=False, random_state=random_state) for _ in range(n_more_estimators)]

            for tree in trees:
                tree.spline_importance = self.spline_importance

            y_tree = (y_numeric, self.unique_times_, self.is_event_time_)

            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y_tree,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                )
                for i, t in enumerate(trees)
            )

            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score_and_attributes(X, (event, time))

        return self

    def _set_oob_score_and_attributes(self, X, y):
        n_samples = X.shape[0]
        event, time = y
        predictions = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)

        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.max_samples)

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap)
            p_estimator = estimator.predict(X[unsampled_indices, :], check_input=False)
            predictions[unsampled_indices] += p_estimator
            n_predictions[unsampled_indices] += 1

        if (n_predictions == 0).any():
            warnings.warn("Some inputs do not have OOB scores.", stacklevel=3)
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions
        self.oob_score_ = concordance_index_censored(event, time, predictions)[0]

    def _predict(self, predict_fn, X):
        check_is_fitted(self, "estimators_")
        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        if predict_fn == "predict":
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)

        def _get_fn(est, name):
            fn = getattr(est, name)
            if name in ("predict_cumulative_hazard_function", "predict_survival_function"):
                fn = partial(fn, return_array=True)
            return fn

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(_get_fn(e, predict_fn), X, [y_hat], lock) for e in self.estimators_
        )

        y_hat /= len(self.estimators_)
        return y_hat

    def predict(self, X):
        return self._predict("predict", X)

    def predict_cumulative_hazard_function(self, X, return_array=False):
        arr = self._predict("predict_cumulative_hazard_function", X)
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)

    def predict_survival_function(self, X, return_array=False):
        arr = self._predict("predict_survival_function", X)
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)


class RandomSurvivalForest(SurvivalAnalysisMixin, _BaseSurvivalForest):
    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        low_memory=False,
        spline_importance=None,
    ):
        super().__init__(
            estimator=SurvivalTree(),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
                "low_memory",
                "spline_importance",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            spline_importance=spline_importance,
        )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.low_memory = low_memory
        self.spline_importance = spline_importance
