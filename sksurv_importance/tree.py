# Full tree.py with spline_importance logic (abbreviated for message size)
from math import ceil
from numbers import Integral, Real
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data
from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival
from sksurv_importance._criterion import WeightLogrankCriterion, LogrankCriterion, get_unique_times
from sksurv_importance._splitter import BestSplitter, BestSparseSplitter, RandomSplitter, RandomSparseSplitter
from sksurv_importance._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder, Tree
from sksurv_importance import _tree


class SurvivalTree(BaseEstimator, SurvivalAnalysisMixin):
    def __init__(self, *, splitter="best", max_depth=None, min_samples_split=6, min_samples_leaf=3,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                 low_memory=False, spline_importance=None):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.low_memory = low_memory
        self.spline_importance = spline_importance
        self.importance_matrix = None

    def _fit(self, X, y, sample_weight=None, check_input=True, missing_values_in_feature_mask=None):
        random_state = check_random_state(self.random_state)

        if check_input:
            X = validate_data(self, X, dtype=_tree.DTYPE, ensure_min_samples=2, accept_sparse="csc", ensure_all_finite=False)
            event, time = check_array_survival(X, y)
            time = time.astype(np.float64)

            if self.spline_importance is not None:
                self.unique_times_, self.is_event_time_ = get_unique_times(time, event)

                if hasattr(self, 'feature_names_in_'):
                    feature_names = self.feature_names_in_
                else:
                    feature_names = [f"f{i}" for i in range(X.shape[1])]

                valid_features = []
                feature_indices = []
                for idx, name in enumerate(feature_names):
                    if name in self.spline_importance:
                        valid_features.append(name)
                        feature_indices.append(idx)

                if not valid_features:
                    raise ValueError("No matching features in spline_importance")

                n_times = len(self.unique_times_)
                importance_matrix = np.zeros((len(valid_features), n_times), dtype=np.float64)

                for f_idx, feat_name in enumerate(valid_features):
                    spline = self.spline_importance[feat_name]
                    importance_matrix[f_idx, :] = spline(self.unique_times_)

                self.importance_matrix = importance_matrix
                self.feature_indices_for_importance = np.array(feature_indices)
            else:
                self.unique_times_, self.is_event_time_ = get_unique_times(time, event)
                self.importance_matrix = None

            self.n_outputs_ = self.unique_times_.shape[0]

            y_numeric = np.empty((X.shape[0], 2), dtype=np.float64)
            y_numeric[:, 0] = time
            y_numeric[:, 1] = event.astype(np.float64)
        else:
            y_numeric, self.unique_times_, self.is_event_time_ = y

        n_samples = X.shape[0]

        if self.importance_matrix is not None:
            criterion = WeightLogrankCriterion(self.n_outputs_, n_samples, self.unique_times_, self.is_event_time_)
            criterion.importance_matrix = self.importance_matrix
        else:
            criterion = LogrankCriterion(self.n_outputs_, n_samples, self.unique_times_, self.is_event_time_)

        # Rest of tree building logic (splitter, builder) - same as before
        # ... (copy from your original tree.py)

        return self

# ExtraSurvivalTree and other methods remain the same
