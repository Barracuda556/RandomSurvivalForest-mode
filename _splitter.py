# _splitter.py
import numpy as np
from scipy.sparse import issparse
import math

from sklearn.tree._criterion import Criterion
from ._partitioner import (
    FEATURE_THRESHOLD, DensePartitioner, SparsePartitioner,
    shift_missing_values_to_left_if_required
)
from ._utils import rand_int, rand_uniform, RandomState, RAND_R_MAX

INFINITY = np.inf


class SplitRecord:
    """Record of a split for a node."""
    
    def __init__(self, start_pos=0):
        self.impurity_left = INFINITY
        self.impurity_right = INFINITY
        self.pos = start_pos
        self.feature = 0
        self.threshold = 0.0
        self.improvement = -INFINITY
        self.missing_go_to_left = False
        self.n_missing = 0
    
    def __repr__(self):
        return (f"SplitRecord(feature={self.feature}, threshold={self.threshold:.4f}, "
                f"pos={self.pos}, improvement={self.improvement:.4f})")
    
    def copy_from(self, other):
        """Copy data from another SplitRecord."""
        self.impurity_left = other.impurity_left
        self.impurity_right = other.impurity_right
        self.pos = other.pos
        self.feature = other.feature
        self.threshold = other.threshold
        self.improvement = other.improvement
        self.missing_go_to_left = other.missing_go_to_left
        self.n_missing = other.n_missing


class ParentInfo:
    """Information about parent node."""
    
    def __init__(self):
        self.n_constant_features = 0
        self.impurity = INFINITY
        self.lower_bound = -INFINITY
        self.upper_bound = INFINITY


class Splitter:
    """Abstract splitter class.
    
    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __init__(
        self,
        criterion,
        max_features,
        min_samples_leaf,
        min_weight_leaf,
        random_state,
        monotonic_cst,
    ):
        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None
        
        # Initialize random state
        if isinstance(random_state, np.random.RandomState):
            seed = random_state.randint(0, RAND_R_MAX)
        elif hasattr(random_state, 'randint'):
            # Наш RandomState или совместимый объект
            seed = random_state.randint(0, RAND_R_MAX)
        else:
            # Целое число или None
            seed = random_state if random_state is not None else np.random.randint(0, RAND_R_MAX)
        
        # Используем массив numpy для изменения состояния
        self.rand_r_state = np.array([seed], dtype=np.uint32)
        
        # Buffers
        self.samples = None
        self.features = None
        self.feature_values = None
        self.constant_features = None
        self.y = None
        self.sample_weight = None
        
        # Node state
        self.start = 0
        self.end = 0
        self.weighted_n_samples = 0.0
        self.partitioner = None

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __reduce__(self):
        return (type(self), (self.criterion,
                             self.max_features,
                             self.min_samples_leaf,
                             self.min_weight_leaf,
                             self.random_state,
                             self.monotonic_cst), self.__getstate__())

    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        """Initialize the splitter.
        
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        n_samples = X.shape[0]
        
        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.empty(n_samples, dtype=np.intp)
        
        j = 0
        weighted_n_samples = 0.0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                self.samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y
        self.sample_weight = sample_weight
        
        if missing_values_in_feature_mask is not None:
            self.criterion.init_sum_missing()
        
        return 0

    def node_reset(self, start, end, weighted_n_node_samples):
        """Reset splitter on node samples[start:end].
        
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    def node_split(self, parent_record, split):
        """Find the best split on node samples[start:end].
        
        This is a placeholder method.
        """
        raise NotImplementedError("Subclasses must implement node_split")

    def node_value(self, dest):
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    def clip_node_value(self, dest, lower_bound, upper_bound):
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""
        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    def node_impurity(self):
        """Return the impurity of the current node."""
        return self.criterion.node_impurity()


def node_split_best(splitter, partitioner, criterion, split, parent_record, importance_matrix=None):
    """Find the best split on node samples[start:end]"""
    monotonic_cst = splitter.monotonic_cst
    with_monotonic_cst = splitter.with_monotonic_cst
    
    # Find the best split
    start = splitter.start
    end = splitter.end
    n_missing = 0
    has_missing = False
    n_searches = 0
    n_left, n_right = 0, 0
    missing_go_to_left = False

    samples = splitter.samples
    features = splitter.features
    constant_features = splitter.constant_features
    n_features = splitter.n_features

    feature_values = splitter.feature_values
    max_features = splitter.max_features
    min_samples_leaf = splitter.min_samples_leaf
    min_weight_leaf = splitter.min_weight_leaf
    
    # Получаем указатель на состояние через массив numpy
    random_state_ptr = splitter.rand_r_state

    best_split = SplitRecord(end)
    current_split = SplitRecord()
    current_proxy_improvement = -INFINITY
    best_proxy_improvement = -INFINITY

    impurity = parent_record.impurity
    lower_bound = parent_record.lower_bound
    upper_bound = parent_record.upper_bound

    f_i = n_features
    f_j = 0
    p = 0
    p_prev = 0  # Инициализация, но значение будет обновлено в цикле

    n_visited_features = 0
    # Number of features discovered to be constant during the split search
    n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    n_drawn_constants = 0
    n_known_constants = parent_record.n_constant_features
    n_total_constants = n_known_constants

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm
    while (f_i > n_total_constants and  # Stop early if remaining features are constant
           (n_visited_features < max_features or
            # At least one drawn features must be non constant
            n_visited_features <= n_found_constants + n_drawn_constants)):

        n_visited_features += 1

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state_ptr)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[
        current_split.feature = features[f_j]

        if importance_matrix is not None:
            importance_array = importance_matrix[features[f_j],:]
        else: importance_array = None
        
        partitioner.sort_samples_and_feature_values(current_split.feature)
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (end_non_missing == start or  # All values for this feature are missing
            feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD):
            # We consider this feature constant
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]
            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)  # initialize even when n_missing == 0

        # Evaluate all splits
        # If there are missing values, then we search twice for the most optimal split.
        # The first search will have all the missing values going to the right node.
        # The second search will have all the missing values going to the left node.
        # If there are no missing values, then we search only once for the most
        # optimal split.
        n_searches = 2 if has_missing else 1

        for i in range(n_searches):
            missing_go_to_left = i == 1
            criterion.missing_go_to_left = missing_go_to_left
            criterion.reset()

            p = start

            while p < end_non_missing:
                # Используем next_p, который возвращает значения
                result = partitioner.next_p(p_prev, p)
                if result is None:
                    break
                p_prev, p = result

                if p >= end_non_missing:
                    continue

                if missing_go_to_left:
                    n_left = p - start + n_missing
                    n_right = end_non_missing - p
                else:
                    n_left = p - start
                    n_right = end_non_missing - p + n_missing

                # Reject if min_samples_leaf is not guaranteed
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                current_split.pos = p
                criterion.update(current_split.pos)

                # Reject if monotonicity constraints are not satisfied
                if (with_monotonic_cst and
                    monotonic_cst[current_split.feature] != 0 and
                    not criterion.check_monotonicity(
                        monotonic_cst[current_split.feature],
                        lower_bound,
                        upper_bound,
                    )):
                    continue

                # Reject if min_weight_leaf is not satisfied
                if ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    continue

                if importance_array is not None and len(importance_array) != criterion.n_unique_times:
                    raise ValueError(f"importance_array length mismatch: got {len(importance_array)}, "
                                    f"expected {criterion.n_unique_times}")

                if hasattr(criterion, 'importance_matrix') and criterion.importance_matrix is not None:
                    current_proxy_improvement = criterion.proxy_impurity_improvement(importance_array)
                else:
                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                #if hasattr(criterion, 'proxy_impurity_improvement'):
                #    if importance_array is not None:
                #        current_proxy_improvement = criterion.proxy_impurity_improvement(importance_array)
                #    else:
                #        current_proxy_improvement = criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement
                    # sum of halves is used to avoid infinite value
                    current_split.threshold = (
                        feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                    )

                    if (current_split.threshold == feature_values[p] or
                        current_split.threshold == INFINITY or
                        current_split.threshold == -INFINITY):
                        current_split.threshold = feature_values[p_prev]

                    current_split.n_missing = n_missing

                    # if there are no missing values in the training data, during
                    # test time, we send missing values to the branch that contains
                    # the most samples during training time.
                    if n_missing == 0:
                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    best_split.copy_from(current_split)

        # Evaluate when there are missing values and all missing values goes
        # to the right node and non-missing values goes to the left node.
        if has_missing:
            n_left, n_right = end - start - n_missing, n_missing
            p = end - n_missing
            missing_go_to_left = False

            if not (n_left < min_samples_leaf or n_right < min_samples_leaf):
                criterion.missing_go_to_left = missing_go_to_left
                criterion.update(p)

                if not ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):

                    if importance_array is not None and len(importance_array) != criterion.n_unique_times:
                        raise ValueError(f"importance_array length mismatch: got {len(importance_array)}, "
                                        f"expected {criterion.n_unique_times}")

                    if hasattr(criterion, 'importance_matrix') and criterion.importance_matrix is not None:
                        # Получаем importance_array для текущего признака
                        if hasattr(splitter, 'importance_matrix') and splitter.importance_matrix is not None:
                            importance_array = splitter.importance_matrix[current_split.feature, :]
                        else:
                            importance_array = np.ones(criterion.n_unique_times)
                        current_proxy_improvement = criterion.proxy_impurity_improvement(importance_array)
                    else:
                        current_proxy_improvement = criterion.proxy_impurity_improvement()
                    
                    #if hasattr(criterion, 'proxy_impurity_improvement'):
                    #    if importance_array is not None:
                    #        current_proxy_improvement = criterion.proxy_impurity_improvement(importance_array)
                    #    else:
                    #        current_proxy_improvement = criterion.proxy_impurity_improvement()
                    #current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.threshold = INFINITY
                        current_split.missing_go_to_left = missing_go_to_left
                        current_split.n_missing = n_missing
                        current_split.pos = p
                        best_split.copy_from(current_split)

    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            best_split.pos,
            best_split.threshold,
            best_split.feature,
            best_split.n_missing
        )
        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        best_split.impurity_left, best_split.impurity_right = criterion.children_impurity()
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right,
            importance_array
        )

        shift_missing_values_to_left_if_required(best_split, samples, end)

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    features[:n_known_constants] = constant_features[:n_known_constants]

    # Copy newly found constant features
    constant_features[n_known_constants:n_known_constants + n_found_constants] = \
        features[n_known_constants:n_known_constants + n_found_constants]

    # Return values
    parent_record.n_constant_features = n_total_constants
    split.copy_from(best_split)
    return 0


def node_split_random(splitter, partitioner, criterion, split, parent_record, importance_matrix=None):
    """Find the best random split on node samples[start:end]"""
    monotonic_cst = splitter.monotonic_cst
    with_monotonic_cst = splitter.with_monotonic_cst

    # Draw random splits and pick the best
    start = splitter.start
    end = splitter.end
    n_missing = 0
    has_missing = False
    n_left, n_right = 0, 0
    missing_go_to_left = False

    samples = splitter.samples
    features = splitter.features
    constant_features = splitter.constant_features
    n_features = splitter.n_features

    max_features = splitter.max_features
    min_samples_leaf = splitter.min_samples_leaf
    min_weight_leaf = splitter.min_weight_leaf
    
    # Получаем указатель на состояние через массив numpy
    random_state_ptr = splitter.rand_r_state

    best_split = SplitRecord(end)
    current_split = SplitRecord()
    current_proxy_improvement = -INFINITY
    best_proxy_improvement = -INFINITY

    impurity = parent_record.impurity
    lower_bound = parent_record.lower_bound
    upper_bound = parent_record.upper_bound

    f_i = n_features
    f_j = 0
    # Number of features discovered to be constant during the split search
    n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    n_drawn_constants = 0
    n_known_constants = parent_record.n_constant_features
    n_total_constants = n_known_constants
    n_visited_features = 0
    min_feature_value = 0.0
    max_feature_value = 0.0

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm
    while (f_i > n_total_constants and  # Stop early if remaining features are constant
           (n_visited_features < max_features or
            # At least one drawn features must be non constant
            n_visited_features <= n_found_constants + n_drawn_constants)):
        n_visited_features += 1

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state_ptr)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[

        current_split.feature = features[f_j]
        
        if importance_matrix is not None:
            importance_array = importance_matrix[features[f_j],:]
        else: importance_array = None

        # Find min, max as we will randomly select a threshold between them
        min_feature_value, max_feature_value = partitioner.find_min_max(current_split.feature)
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (end_non_missing == start or  # All values for this feature are missing
            max_feature_value <= min_feature_value + FEATURE_THRESHOLD):
            # We consider this feature constant
            features[f_j], features[n_total_constants] = features[n_total_constants], current_split.feature
            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)

        # Draw a random threshold
        current_split.threshold = rand_uniform(
            min_feature_value,
            max_feature_value,
            random_state_ptr,
        )

        if has_missing:
            # If there are missing values, then we randomly make all missing
            # values go to the right or left.
            missing_go_to_left = bool(rand_int(0, 2, random_state_ptr))
        else:
            missing_go_to_left = False
        criterion.missing_go_to_left = missing_go_to_left

        if current_split.threshold == max_feature_value:
            current_split.threshold = min_feature_value

        # Partition
        current_split.pos = partitioner.partition_samples(current_split.threshold)

        if missing_go_to_left:
            n_left = current_split.pos - start + n_missing
            n_right = end_non_missing - current_split.pos
        else:
            n_left = current_split.pos - start
            n_right = end_non_missing - current_split.pos + n_missing

        # Reject if min_samples_leaf is not guaranteed
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        # Evaluate split
        criterion.reset()
        criterion.update(current_split.pos)

        # Reject if min_weight_leaf is not satisfied
        if ((criterion.weighted_n_left < min_weight_leaf) or
                (criterion.weighted_n_right < min_weight_leaf)):
            continue

        # Reject if monotonicity constraints are not satisfied
        if (with_monotonic_cst and
            monotonic_cst[current_split.feature] != 0 and
            not criterion.check_monotonicity(
                monotonic_cst[current_split.feature],
                lower_bound,
                upper_bound,
            )):
            continue

        if importance_array is not None and len(importance_array) != criterion.n_unique_times:
            raise ValueError(f"importance_array length mismatch: got {len(importance_array)}, "
                            f"expected {criterion.n_unique_times}")

        if hasattr(criterion, 'importance_matrix') and criterion.importance_matrix is not None:
            # Получаем importance_array для текущего признака
            if hasattr(splitter, 'importance_matrix') and splitter.importance_matrix is not None:
                importance_array = splitter.importance_matrix[current_split.feature, :]
            else:
                importance_array = np.ones(criterion.n_unique_times)
            current_proxy_improvement = criterion.proxy_impurity_improvement(importance_array)
        else:
            current_proxy_improvement = criterion.proxy_impurity_improvement()

        #if hasattr(criterion, 'proxy_impurity_improvement'):
        #    if importance_array is not None:
        #        current_proxy_improvement = criterion.proxy_impurity_improvement(importance_array)
        #    else:
        #        current_proxy_improvement = criterion.proxy_impurity_improvement()

        #current_proxy_improvement = criterion.proxy_impurity_improvement()

        if current_proxy_improvement > best_proxy_improvement:
            current_split.n_missing = n_missing

            # if there are no missing values in the training data, during
            # test time, we send missing values to the branch that contains
            # the most samples during training time.
            if has_missing:
                current_split.missing_go_to_left = missing_go_to_left
            else:
                current_split.missing_go_to_left = n_left > n_right

            best_proxy_improvement = current_proxy_improvement
            best_split.copy_from(current_split)

    # Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if best_split.pos < end:
        if current_split.feature != best_split.feature:
            partitioner.partition_samples_final(
                best_split.pos,
                best_split.threshold,
                best_split.feature,
                best_split.n_missing
            )
        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        best_split.impurity_left, best_split.impurity_right = criterion.children_impurity()
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right,
            importance_array
        )

        shift_missing_values_to_left_if_required(best_split, samples, end)

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    features[:n_known_constants] = constant_features[:n_known_constants]

    # Copy newly found constant features
    constant_features[n_known_constants:n_known_constants + n_found_constants] = \
        features[n_known_constants:n_known_constants + n_found_constants]

    # Return values
    parent_record.n_constant_features = n_total_constants
    split.copy_from(best_split)
    return 0


class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""
    
    def __init__(self, criterion, max_features, min_samples_leaf,
                 min_weight_leaf, random_state, monotonic_cst):
        super().__init__(criterion, max_features, min_samples_leaf,
                        min_weight_leaf, random_state, monotonic_cst)
    
    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        rc = super().init(X, y, sample_weight, missing_values_in_feature_mask)
        if rc != 0:
            return rc
        
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )
        return 0
    
    def node_split(self, parent_record, split, importance_matrix=None):
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
            importance_matrix,
        )


class BestSparseSplitter(Splitter):
    """Splitter for finding the best split, using the sparse data."""
    
    def __init__(self, criterion, max_features, min_samples_leaf,
                 min_weight_leaf, random_state, monotonic_cst):
        super().__init__(criterion, max_features, min_samples_leaf,
                        min_weight_leaf, random_state, monotonic_cst)
    
    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        rc = super().init(X, y, sample_weight, missing_values_in_feature_mask)
        if rc != 0:
            return rc
        
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )
        return 0
    
    def node_split(self, parent_record, split, importance_matrix=None):
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
            importance_matrix,
        )


class RandomSplitter(Splitter):
    """Splitter for finding the best random split on dense data."""
    
    def __init__(self, criterion, max_features, min_samples_leaf,
                 min_weight_leaf, random_state, monotonic_cst):
        super().__init__(criterion, max_features, min_samples_leaf,
                        min_weight_leaf, random_state, monotonic_cst)
    
    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        rc = super().init(X, y, sample_weight, missing_values_in_feature_mask)
        if rc != 0:
            return rc
        
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )
        return 0
    
    def node_split(self, parent_record, split, importance_matrix=None):
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
            importance_matrix,
        )


class RandomSparseSplitter(Splitter):
    """Splitter for finding the best random split, using the sparse data."""
    
    def __init__(self, criterion, max_features, min_samples_leaf,
                 min_weight_leaf, random_state, monotonic_cst):
        super().__init__(criterion, max_features, min_samples_leaf,
                        min_weight_leaf, random_state, monotonic_cst)
    
    def init(self, X, y, sample_weight, missing_values_in_feature_mask):
        rc = super().init(X, y, sample_weight, missing_values_in_feature_mask)
        if rc != 0:
            return rc
        
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )
        return 0
    
    def node_split(self, parent_record, split, importance_matrix=None):
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
            importance_matrix,
        )
