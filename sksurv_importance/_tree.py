# sksurv_python/_tree.py
import numpy as np
from scipy.sparse import issparse, csr_matrix
import struct
import warnings
from collections import deque
import heapq

# Импортируем наши собственные классы сплиттеров
from ._splitter import (
    BestSplitter, BestSparseSplitter, RandomSplitter, RandomSparseSplitter,
    SplitRecord, ParentInfo
)

INFINITY = np.inf
EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
_TREE_LEAF = TREE_LEAF
_TREE_UNDEFINED = TREE_UNDEFINED

DTYPE = np.float32
DOUBLE = np.float64

INTPTR_MAX = np.iinfo(np.intp).max


class Node:
    """Node structure for tree."""
    
    __slots__ = ('left_child', 'right_child', 'feature', 'threshold', 
                 'impurity', 'n_node_samples', 'weighted_n_node_samples',
                 'missing_go_to_left')
    
    def __init__(self):
        self.left_child = _TREE_UNDEFINED
        self.right_child = _TREE_UNDEFINED
        self.feature = _TREE_UNDEFINED
        self.threshold = _TREE_UNDEFINED
        self.impurity = INFINITY
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.missing_go_to_left = False
    
    def __repr__(self):
        return (f"Node(left={self.left_child}, right={self.right_child}, "
                f"feature={self.feature}, threshold={self.threshold:.4f}, "
                f"impurity={self.impurity:.4f}, samples={self.n_node_samples})")


class TreeBuilder:
    """Interface for different tree building strategies."""
    
    def build(self, tree, X, y, sample_weight=None, missing_values_in_feature_mask=None):
        """Build a decision tree from the training set (X, y)."""
        raise NotImplementedError()
    
    def _check_input(self, X, y, sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()
            
            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)
            
            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based sparse matrices")
        
        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)
        
        if sample_weight is not None and not sample_weight.flags.contiguous:
            sample_weight = np.asarray(sample_weight, dtype=DOUBLE, order="C")
        
        return X, y, sample_weight


class StackRecord:
    """Record on stack for depth-first tree growing."""
    
    __slots__ = ('start', 'end', 'depth', 'parent', 'is_left', 'impurity',
                 'n_constant_features', 'lower_bound', 'upper_bound')
    
    def __init__(self, start=0, end=0, depth=0, parent=_TREE_UNDEFINED, 
                 is_left=False, impurity=INFINITY, n_constant_features=0,
                 lower_bound=-INFINITY, upper_bound=INFINITY):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""
    
    def __init__(self, splitter, min_samples_split, min_samples_leaf, 
                 min_weight_leaf, max_depth, min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
    
    def build(self, tree, X, y, sample_weight=None, missing_values_in_feature_mask=None, importance_matrix=None):
        """Build a decision tree from the training set (X, y)."""
        
        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)
        
        # Initial capacity
        init_capacity = 0
        if tree.max_depth <= 10:
            init_capacity = int(2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047
        
        tree._resize(init_capacity)
        
        # Parameters
        splitter = self.splitter
        max_depth = self.max_depth
        min_samples_leaf = self.min_samples_leaf
        min_weight_leaf = self.min_weight_leaf
        min_samples_split = self.min_samples_split
        min_impurity_decrease = self.min_impurity_decrease
        
        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)
        
        n_node_samples = splitter.n_samples
        
        split = SplitRecord()
        node_id = 0
        
        middle_value = 0.0
        left_child_min = 0.0
        left_child_max = 0.0
        right_child_min = 0.0
        right_child_max = 0.0
        is_leaf = False
        first = True
        max_depth_seen = -1
        rc = 0
        
        builder_stack = []
        parent_record = ParentInfo()
        
        # push root node onto stack
        builder_stack.append(StackRecord(
            start=0,
            end=n_node_samples,
            depth=0,
            parent=_TREE_UNDEFINED,
            is_left=False,
            impurity=INFINITY,
            n_constant_features=0,
            lower_bound=-INFINITY,
            upper_bound=INFINITY
        ))
        
        while builder_stack:
            stack_record = builder_stack.pop()
            
            start = stack_record.start
            end = stack_record.end
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left
            parent_record.impurity = stack_record.impurity
            parent_record.n_constant_features = stack_record.n_constant_features
            parent_record.lower_bound = stack_record.lower_bound
            parent_record.upper_bound = stack_record.upper_bound
            
            n_node_samples = end - start
            weighted_n_node_samples = np.zeros(1, dtype=np.float64)
            splitter.node_reset(start, end, weighted_n_node_samples)
            
            is_leaf = (depth >= max_depth or
                      n_node_samples < min_samples_split or
                      n_node_samples < 2 * min_samples_leaf or
                      weighted_n_node_samples[0] < 2 * min_weight_leaf)
            
            if first:
                parent_record.impurity = splitter.node_impurity()
                first = False
            
            # impurity == 0 with tolerance due to rounding errors
            is_leaf = is_leaf or parent_record.impurity <= EPSILON
            
            if not is_leaf:
                splitter.node_split(parent_record, split, importance_matrix)
                # If EPSILON=0 in the below comparison, float precision
                # issues stop splitting, producing trees that are
                # dissimilar to v0.18
                is_leaf = (is_leaf or split.pos >= end or
                          (split.improvement + EPSILON < min_impurity_decrease))
            
            node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                    split.threshold, parent_record.impurity,
                                    n_node_samples, weighted_n_node_samples[0],
                                    split.missing_go_to_left)
            
            if node_id == INTPTR_MAX:
                rc = -1
                break
            
            # Store value for all nodes
            dest_start = node_id * tree.value_stride
            dest_end = dest_start + tree.value_stride
            dest = tree.value[dest_start:dest_end]
            splitter.node_value(dest)
            if splitter.with_monotonic_cst:
                splitter.clip_node_value(dest, parent_record.lower_bound, parent_record.upper_bound)
            
            if not is_leaf:
                if (not splitter.with_monotonic_cst or
                    splitter.monotonic_cst[split.feature] == 0):
                    # Split on a feature with no monotonicity constraint
                    left_child_min = right_child_min = parent_record.lower_bound
                    left_child_max = right_child_max = parent_record.upper_bound
                elif splitter.monotonic_cst[split.feature] == 1:
                    # Split on a feature with monotonic increase constraint
                    left_child_min = parent_record.lower_bound
                    right_child_max = parent_record.upper_bound
                    
                    # Lower bound for right child and upper bound for left child
                    middle_value = splitter.criterion.middle_value()
                    right_child_min = middle_value
                    left_child_max = middle_value
                else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                    # Split on a feature with monotonic decrease constraint
                    right_child_min = parent_record.lower_bound
                    left_child_max = parent_record.upper_bound
                    
                    # Lower bound for left child and upper bound for right child
                    middle_value = splitter.criterion.middle_value()
                    left_child_min = middle_value
                    right_child_max = middle_value
                
                # Push right child on stack
                builder_stack.append(StackRecord(
                    start=split.pos,
                    end=end,
                    depth=depth + 1,
                    parent=node_id,
                    is_left=False,
                    impurity=split.impurity_right,
                    n_constant_features=parent_record.n_constant_features,
                    lower_bound=right_child_min,
                    upper_bound=right_child_max
                ))
                
                # Push left child on stack
                builder_stack.append(StackRecord(
                    start=start,
                    end=split.pos,
                    depth=depth + 1,
                    parent=node_id,
                    is_left=True,
                    impurity=split.impurity_left,
                    n_constant_features=parent_record.n_constant_features,
                    lower_bound=left_child_min,
                    upper_bound=left_child_max
                ))
            
            if depth > max_depth_seen:
                max_depth_seen = depth
        
        if rc >= 0:
            rc = tree._resize_c(tree.node_count)
        
        if rc >= 0:
            tree.max_depth = max_depth_seen
        
        if rc == -1:
            raise MemoryError()


class FrontierRecord:
    """Record for frontier in best-first tree building."""
    
    __slots__ = ('node_id', 'start', 'end', 'pos', 'depth', 'is_leaf',
                 'impurity', 'impurity_left', 'impurity_right', 'improvement',
                 'lower_bound', 'upper_bound', 'middle_value')
    
    def __init__(self):
        self.node_id = 0
        self.start = 0
        self.end = 0
        self.pos = 0
        self.depth = 0
        self.is_leaf = False
        self.impurity = INFINITY
        self.impurity_left = INFINITY
        self.impurity_right = INFINITY
        self.improvement = -INFINITY
        self.lower_bound = -INFINITY
        self.upper_bound = INFINITY
        self.middle_value = 0.0
    
    def __lt__(self, other):
        return self.improvement < other.improvement


class BestFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in best-first fashion."""
    
    def __init__(self, splitter, min_samples_split, min_samples_leaf, 
                 min_weight_leaf, max_depth, max_leaf_nodes, min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
    
    def build(self, tree, X, y, sample_weight=None, missing_values_in_feature_mask=None, importance_matrix=None):
        """Build a decision tree from the training set (X, y)."""
        
        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)
        
        # Parameters
        splitter = self.splitter
        max_leaf_nodes = self.max_leaf_nodes
        
        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)
        
        frontier = []
        record = FrontierRecord()
        split_node_left = FrontierRecord()
        split_node_right = FrontierRecord()
        left_child_min = 0.0
        left_child_max = 0.0
        right_child_min = 0.0
        right_child_max = 0.0
        
        n_node_samples = splitter.n_samples
        max_split_nodes = max_leaf_nodes - 1
        max_depth_seen = -1
        rc = 0
        
        parent_record = ParentInfo()
        
        # Initial capacity
        init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)
        
        # add root to frontier
        rc = self._add_split_node(
            splitter=splitter,
            tree=tree,
            start=0,
            end=n_node_samples,
            is_first=IS_FIRST,
            is_left=IS_LEFT,
            parent=None,
            depth=0,
            parent_record=parent_record,
            res=split_node_left,
            importance_matrix=importance_matrix
        )
        
        if rc >= 0:
            heapq.heappush(frontier, split_node_left)
        
        while frontier:
            record = heapq.heappop(frontier)
            
            node = tree.nodes[record.node_id]
            is_leaf = (record.is_leaf or max_split_nodes <= 0)
            
            if is_leaf:
                # Node is not expandable; set node as leaf
                node.left_child = _TREE_LEAF
                node.right_child = _TREE_LEAF
                node.feature = _TREE_UNDEFINED
                node.threshold = _TREE_UNDEFINED
            
            else:
                # Node is expandable
                if (not splitter.with_monotonic_cst or
                    splitter.monotonic_cst[node.feature] == 0):
                    # Split on a feature with no monotonicity constraint
                    left_child_min = right_child_min = record.lower_bound
                    left_child_max = right_child_max = record.upper_bound
                elif splitter.monotonic_cst[node.feature] == 1:
                    # Split on a feature with monotonic increase constraint
                    left_child_min = record.lower_bound
                    right_child_max = record.upper_bound
                    
                    # Lower bound for right child and upper bound for left child
                    right_child_min = record.middle_value
                    left_child_max = record.middle_value
                else:  # i.e. splitter.monotonic_cst[node.feature] == -1
                    # Split on a feature with monotonic decrease constraint
                    right_child_min = record.lower_bound
                    left_child_max = record.upper_bound
                    
                    # Lower bound for left child and upper bound for right child
                    left_child_min = record.middle_value
                    right_child_max = record.middle_value
                
                # Decrement number of split nodes available
                max_split_nodes -= 1
                
                # Compute left split node
                parent_record.lower_bound = left_child_min
                parent_record.upper_bound = left_child_max
                parent_record.impurity = record.impurity_left
                rc = self._add_split_node(
                    splitter=splitter,
                    tree=tree,
                    start=record.start,
                    end=record.pos,
                    is_first=IS_NOT_FIRST,
                    is_left=IS_LEFT,
                    parent=node,
                    depth=record.depth + 1,
                    parent_record=parent_record,
                    res=split_node_left,
                    importance_matrix=importance_matrix
                )
                
                if rc == -1:
                    break
                
                # tree.nodes may have changed
                node = tree.nodes[record.node_id]
                
                # Compute right split node
                parent_record.lower_bound = right_child_min
                parent_record.upper_bound = right_child_max
                parent_record.impurity = record.impurity_right
                rc = self._add_split_node(
                    splitter=splitter,
                    tree=tree,
                    start=record.pos,
                    end=record.end,
                    is_first=IS_NOT_FIRST,
                    is_left=IS_NOT_LEFT,
                    parent=node,
                    depth=record.depth + 1,
                    parent_record=parent_record,
                    res=split_node_right,
                    importance_matrix=importance_matrix
                )
                
                if rc == -1:
                    break
                
                # Add nodes to queue
                heapq.heappush(frontier, split_node_left)
                heapq.heappush(frontier, split_node_right)
            
            if record.depth > max_depth_seen:
                max_depth_seen = record.depth
        
        if rc >= 0:
            rc = tree._resize_c(tree.node_count)
        
        if rc >= 0:
            tree.max_depth = max_depth_seen
        
        if rc == -1:
            raise MemoryError()
    
    def _add_split_node(self, splitter, tree, start, end, is_first, is_left,
                       parent, depth, parent_record, res, importance_matrix=None):
        """Adds node w/ partition ``[start, end)`` to the frontier."""
        split = SplitRecord()
        node_id = 0
        n_node_samples = 0
        min_impurity_decrease = self.min_impurity_decrease
        weighted_n_node_samples = np.zeros(1, dtype=np.float64)
        is_leaf = False
        
        splitter.node_reset(start, end, weighted_n_node_samples)
        
        # reset n_constant_features for this specific split before beginning split search
        parent_record.n_constant_features = 0
        
        if is_first:
            parent_record.impurity = splitter.node_impurity()
        
        n_node_samples = end - start
        is_leaf = (depth >= self.max_depth or
                  n_node_samples < self.min_samples_split or
                  n_node_samples < 2 * self.min_samples_leaf or
                  weighted_n_node_samples[0] < 2 * self.min_weight_leaf or
                  parent_record.impurity <= EPSILON)  # impurity == 0 with tolerance
        
        if not is_leaf:
            splitter.node_split(parent_record, split, importance_matrix)
            # If EPSILON=0 in the below comparison, float precision issues stop
            # splitting early, producing trees that are dissimilar to v0.18
            is_leaf = (is_leaf or split.pos >= end or
                      split.improvement + EPSILON < min_impurity_decrease)
        
        parent_idx = -1 if parent is None else parent
        node_id = tree._add_node(parent_idx, is_left, is_leaf,
                                split.feature, split.threshold, parent_record.impurity,
                                n_node_samples, weighted_n_node_samples[0],
                                split.missing_go_to_left)
        
        if node_id == INTPTR_MAX:
            return -1
        
        # compute values also for split nodes (might become leafs later).
        dest_start = node_id * tree.value_stride
        dest_end = dest_start + tree.value_stride
        dest = tree.value[dest_start:dest_end]
        splitter.node_value(dest)
        if splitter.with_monotonic_cst:
            splitter.clip_node_value(dest, parent_record.lower_bound, parent_record.upper_bound)
        
        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = parent_record.impurity
        res.lower_bound = parent_record.lower_bound
        res.upper_bound = parent_record.upper_bound
        
        res.middle_value = splitter.criterion.middle_value()
        
        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = False
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right
        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = True
            res.improvement = 0.0
            res.impurity_left = parent_record.impurity
            res.impurity_right = parent_record.impurity
        
        return 0


class Tree:
    """Array-based representation of a binary decision tree."""
    
    def __init__(self, n_features, n_classes, n_outputs):
        """Constructor."""
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = np.asarray(n_classes, dtype=np.intp)
        
        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes
        
        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = None
        self.nodes = None
        
        # Initialize arrays
        self._resize(3)  # default initial capacity
    
    @property
    def children_left(self):
        """Array of left children for each node."""
        if self.node_count == 0:
            return np.array([], dtype=np.intp)
        return np.array([node.left_child for node in self.nodes[:self.node_count]], dtype=np.intp)
    
    @property
    def children_right(self):
        """Array of right children for each node."""
        if self.node_count == 0:
            return np.array([], dtype=np.intp)
        return np.array([node.right_child for node in self.nodes[:self.node_count]], dtype=np.intp)
    
    @property
    def n_leaves(self):
        """Number of leaves in the tree."""
        if self.node_count == 0:
            return 0
        left_leaves = self.children_left == -1
        right_leaves = self.children_right == -1
        return np.sum(np.logical_and(left_leaves, right_leaves))
    
    @property
    def feature(self):
        """Array of features for each node."""
        if self.node_count == 0:
            return np.array([], dtype=np.intp)
        return np.array([node.feature for node in self.nodes[:self.node_count]], dtype=np.intp)
    
    @property
    def threshold(self):
        """Array of thresholds for each node."""
        if self.node_count == 0:
            return np.array([], dtype=np.float64)
        return np.array([node.threshold for node in self.nodes[:self.node_count]], dtype=np.float64)
    
    @property
    def impurity(self):
        """Array of impurities for each node."""
        if self.node_count == 0:
            return np.array([], dtype=np.float64)
        return np.array([node.impurity for node in self.nodes[:self.node_count]], dtype=np.float64)
    
    @property
    def n_node_samples(self):
        """Array of sample counts for each node."""
        if self.node_count == 0:
            return np.array([], dtype=np.intp)
        return np.array([node.n_node_samples for node in self.nodes[:self.node_count]], dtype=np.intp)
    
    @property
    def weighted_n_node_samples(self):
        """Array of weighted sample counts for each node."""
        if self.node_count == 0:
            return np.array([], dtype=np.float64)
        return np.array([node.weighted_n_node_samples for node in self.nodes[:self.node_count]], dtype=np.float64)
    
    @property
    def missing_go_to_left(self):
        """Array indicating missing values go to left child."""
        if self.node_count == 0:
            return np.array([], dtype=np.bool_)
        return np.array([node.missing_go_to_left for node in self.nodes[:self.node_count]], dtype=np.bool_)
    
    @property
    def value_array(self):
        """3D array of node values."""
        if self.value is None or self.node_count == 0:
            return np.zeros((0, self.n_outputs, self.max_n_classes), dtype=np.float64)
        return self.value[:self.node_count * self.value_stride].reshape(
            (self.node_count, self.n_outputs, self.max_n_classes))
    
    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                      self.n_classes,
                      self.n_outputs), self.__getstate__())
    
    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        d["max_depth"] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d
    
    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.node_count = d["node_count"]
        
        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which cannot be imported')
        
        node_ndarray = d['nodes']
        value_ndarray = d['values']
        
        value_shape = (node_ndarray.shape[0], self.n_outputs, self.max_n_classes)
        
        node_ndarray = self._check_node_ndarray(node_ndarray)
        value_ndarray = self._check_value_ndarray(value_ndarray, expected_shape=value_shape)
        
        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        
        # Copy node data
        node_dtype = node_ndarray.dtype
        for i in range(min(self.capacity, len(node_ndarray))):
            node = Node()
            node.left_child = node_ndarray[i]['left_child']
            node.right_child = node_ndarray[i]['right_child']
            node.feature = node_ndarray[i]['feature']
            node.threshold = node_ndarray[i]['threshold']
            node.impurity = node_ndarray[i]['impurity']
            node.n_node_samples = node_ndarray[i]['n_node_samples']
            node.weighted_n_node_samples = node_ndarray[i]['weighted_n_node_samples']
            node.missing_go_to_left = node_ndarray[i]['missing_go_to_left']
            self.nodes[i] = node
        
        # Copy value data
        if value_ndarray is not None:
            self.value[:value_ndarray.size] = value_ndarray.ravel()
    
    def _resize(self, capacity):
        """Resize all inner arrays to `capacity`."""
        if self._resize_c(capacity) != 0:
            raise MemoryError()
    
    def _resize_c(self, capacity=INTPTR_MAX):
        """Guts of _resize"""
        if capacity == self.capacity and self.nodes is not None:
            return 0
        
        if capacity == INTPTR_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity
        
        # Resize nodes array
        if self.nodes is None:
            self.nodes = [Node() for _ in range(capacity)]
        else:
            old_nodes = self.nodes
            self.nodes = [Node() for _ in range(capacity)]
            min_count = min(len(old_nodes), self.node_count)
            for i in range(min_count):
                self.nodes[i] = old_nodes[i]
        
        # Resize value array
        new_value_size = capacity * self.value_stride
        if self.value is None:
            self.value = np.zeros(new_value_size, dtype=np.float64)
        else:
            old_value = self.value
            self.value = np.zeros(new_value_size, dtype=np.float64)
            min_size = min(len(old_value), self.node_count * self.value_stride)
            self.value[:min_size] = old_value[:min_size]
        
        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity
        
        self.capacity = capacity
        return 0
    
    def _add_node(self, parent, is_left, is_leaf, feature, threshold, impurity,
                 n_node_samples, weighted_n_node_samples, missing_go_to_left):
        """Add a node to the tree."""
        node_id = self.node_count
        
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return INTPTR_MAX
        
        node = self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        node.missing_go_to_left = missing_go_to_left
        
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id
        
        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold
        
        self.node_count += 1
        return node_id
    
    def predict(self, X):
        """Predict target for X."""
        out = self.value_array.take(self.apply(X), axis=0, mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out
    
    def apply(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)
    
    def _apply_dense(self, X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        
        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s" % type(X))
        
        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
        
        # Extract input
        X_ndarray = X
        n_samples = X.shape[0]
        
        # Initialize output
        out = np.zeros(n_samples, dtype=np.intp)
        
        for i in range(n_samples):
            node_idx = 0
            
            while True:
                node = self.nodes[node_idx]
                
                # Check if leaf
                if node.left_child == _TREE_LEAF:
                    out[i] = node_idx
                    break
                
                # Get feature value
                X_i_node_feature = X_ndarray[i, node.feature]
                
                if np.isnan(X_i_node_feature):
                    if node.missing_go_to_left:
                        node_idx = node.left_child
                    else:
                        node_idx = node.right_child
                elif X_i_node_feature <= node.threshold:
                    node_idx = node.left_child
                else:
                    node_idx = node.right_child
        
        return out
    
    def _apply_sparse_csr(self, X):
        """Finds the terminal region (=leaf node) for each sample in sparse X."""
        # Check input
        if not (issparse(X) and X.format == 'csr'):
            raise ValueError("X should be in csr_matrix format, got %s" % type(X))
        
        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
        
        # Extract input
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Initialize output
        out = np.zeros(n_samples, dtype=np.intp)
        
        for i in range(n_samples):
            node_idx = 0
            
            while True:
                node = self.nodes[node_idx]
                
                # Check if leaf
                if node.left_child == _TREE_LEAF:
                    out[i] = node_idx
                    break
                
                # Get feature value
                feature_value = 0.0
                for k in range(X_indptr[i], X_indptr[i + 1]):
                    if X_indices[k] == node.feature:
                        feature_value = X_data[k]
                        break
                
                if feature_value <= node.threshold:
                    node_idx = node.left_child
                else:
                    node_idx = node.right_child
        
        return out
    
    def decision_path(self, X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)
    
    def _decision_path_dense(self, X):
        """Finds the decision path (=node) for each sample in X."""
        
        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s" % type(X))
        
        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
        
        # Extract input
        X_ndarray = X
        n_samples = X.shape[0]
        
        # Initialize output
        indptr = np.zeros(n_samples + 1, dtype=np.intp)
        indices = np.zeros(n_samples * (1 + self.max_depth), dtype=np.intp)
        
        for i in range(n_samples):
            node_idx = 0
            indptr[i + 1] = indptr[i]
            
            while True:
                node = self.nodes[node_idx]
                indices[indptr[i + 1]] = node_idx
                indptr[i + 1] += 1
                
                # Check if leaf
                if node.left_child == _TREE_LEAF:
                    break
                
                # Get feature value
                X_i_node_feature = X_ndarray[i, node.feature]
                
                if np.isnan(X_i_node_feature):
                    if node.missing_go_to_left:
                        node_idx = node.left_child
                    else:
                        node_idx = node.right_child
                elif X_i_node_feature <= node.threshold:
                    node_idx = node.left_child
                else:
                    node_idx = node.right_child
        
        indices = indices[:indptr[n_samples]]
        data = np.ones(shape=len(indices), dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                        shape=(n_samples, self.node_count))
        
        return out
    
    def _decision_path_sparse_csr(self, X):
        """Finds the decision path (=node) for each sample in X."""
        
        # Check input
        if not (issparse(X) and X.format == "csr"):
            raise ValueError("X should be in csr_matrix format, got %s" % type(X))
        
        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
        
        # Extract input
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Initialize output
        indptr = np.zeros(n_samples + 1, dtype=np.intp)
        indices = np.zeros(n_samples * (1 + self.max_depth), dtype=np.intp)
        
        for i in range(n_samples):
            node_idx = 0
            indptr[i + 1] = indptr[i]
            
            while True:
                node = self.nodes[node_idx]
                indices[indptr[i + 1]] = node_idx
                indptr[i + 1] += 1
                
                # Check if leaf
                if node.left_child == _TREE_LEAF:
                    break
                
                # Get feature value
                feature_value = 0.0
                for k in range(X_indptr[i], X_indptr[i + 1]):
                    if X_indices[k] == node.feature:
                        feature_value = X_data[k]
                        break
                
                if feature_value <= node.threshold:
                    node_idx = node.left_child
                else:
                    node_idx = node.right_child
        
        indices = indices[:indptr[n_samples]]
        data = np.ones(shape=len(indices), dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                        shape=(n_samples, self.node_count))
        
        return out
    
    def compute_node_depths(self):
        """Compute the depth of each node in a tree."""
        depths = np.zeros(self.node_count, dtype=np.int64)
        children_left = self.children_left
        children_right = self.children_right
        
        depths[0] = 1  # init root node
        for node_id in range(self.node_count):
            if children_left[node_id] != _TREE_LEAF:
                depth = depths[node_id] + 1
                depths[children_left[node_id]] = depth
                depths[children_right[node_id]] = depth
        
        return depths
    
    def compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        importances = np.zeros(self.n_features, dtype=np.float64)
        
        for node in self.nodes[:self.node_count]:
            if node.left_child != _TREE_LEAF:
                left = self.nodes[node.left_child]
                right = self.nodes[node.right_child]
                
                importances[node.feature] += (
                    node.weighted_n_node_samples * node.impurity -
                    left.weighted_n_node_samples * left.impurity -
                    right.weighted_n_node_samples * right.impurity)
        
        if self.node_count > 0:
            root_weight = self.nodes[0].weighted_n_node_samples
            if root_weight > 0:
                importances /= root_weight
        
        if normalize:
            normalizer = np.sum(importances)
            
            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer
        
        return importances
    
    def _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array."""
        return self.value_array
    
    def _get_node_ndarray(self):
        """Wraps nodes as a NumPy structured array."""
        dtype = np.dtype([
            ('left_child', np.intp),
            ('right_child', np.intp),
            ('feature', np.intp),
            ('threshold', np.float64),
            ('impurity', np.float64),
            ('n_node_samples', np.intp),
            ('weighted_n_node_samples', np.float64),
            ('missing_go_to_left', np.bool_)
        ])
        
        arr = np.zeros(self.node_count, dtype=dtype)
        for i, node in enumerate(self.nodes[:self.node_count]):
            arr[i] = (node.left_child, node.right_child, node.feature,
                     node.threshold, node.impurity, node.n_node_samples,
                     node.weighted_n_node_samples, node.missing_go_to_left)
        
        return arr
    
    def _check_node_ndarray(self, node_ndarray):
        """Check node array from pickle."""
        if node_ndarray.ndim != 1:
            raise ValueError(
                "Wrong dimensions for node array from the pickle: "
                f"expected 1, got {node_ndarray.ndim}"
            )
        
        if not node_ndarray.flags.c_contiguous:
            raise ValueError(
                "node array from the pickle should be a C-contiguous array"
            )
        
        return node_ndarray
    
    def _check_value_ndarray(self, value_ndarray, expected_shape):
        """Check value array from pickle."""
        if value_ndarray.shape != expected_shape:
            raise ValueError(
                "Wrong shape for value array from the pickle: "
                f"expected {expected_shape}, got {value_ndarray.shape}"
            )
        
        if not value_ndarray.flags.c_contiguous:
            raise ValueError(
                "value array from the pickle should be a C-contiguous array"
            )
        
        return value_ndarray


# =============================================================================
# Build Pruned Tree
# =============================================================================

class _CCPPruneController:
    """Base class used by build_pruned_tree_ccp and ccp_pruning_path
    to control pruning.
    """
    
    def stop_pruning(self, effective_alpha):
        """Return True to stop pruning and False to continue pruning"""
        return False
    
    def save_metrics(self, effective_alpha, subtree_impurities):
        """Save metrics when pruning"""
        pass
    
    def after_pruning(self, in_subtree):
        """Called after pruning"""
        pass


class _AlphaPruner(_CCPPruneController):
    """Use alpha to control when to stop pruning."""
    
    def __init__(self, ccp_alpha):
        self.ccp_alpha = ccp_alpha
        self.capacity = 0
    
    def stop_pruning(self, effective_alpha):
        # The subtree on the previous iteration has the greatest ccp_alpha
        # less than or equal to self.ccp_alpha
        return self.ccp_alpha < effective_alpha
    
    def after_pruning(self, in_subtree):
        """Updates the number of leaves in subtree"""
        self.capacity = np.sum(in_subtree)


class _PathFinder(_CCPPruneController):
    """Record metrics used to return the cost complexity path."""
    
    def __init__(self, node_count):
        self.ccp_alphas = np.zeros(node_count, dtype=np.float64)
        self.impurities = np.zeros(node_count, dtype=np.float64)
        self.count = 0
    
    def save_metrics(self, effective_alpha, subtree_impurities):
        self.ccp_alphas[self.count] = effective_alpha
        self.impurities[self.count] = subtree_impurities
        self.count += 1


def _cost_complexity_prune(leaves_in_subtree, orig_tree, controller):
    """Perform cost complexity pruning."""
    
    n_nodes = orig_tree.node_count
    weighted_n_node_samples = orig_tree.weighted_n_node_samples
    total_sum_weights = weighted_n_node_samples[0]
    impurity = orig_tree.impurity
    
    # weighted impurity of each node
    r_node = np.empty(n_nodes, dtype=np.float64)
    
    child_l = orig_tree.children_left
    child_r = orig_tree.children_right
    parent = np.zeros(n_nodes, dtype=np.intp)
    
    for i in range(n_nodes):
        r_node[i] = (weighted_n_node_samples[i] * impurity[i] / total_sum_weights)
    
    # Find parent node ids and leaves
    stack = [(0, _TREE_UNDEFINED)]  # (node_idx, parent)
    
    while stack:
        node_idx, parent_idx = stack.pop()
        parent[node_idx] = parent_idx
        
        if child_l[node_idx] == _TREE_LEAF:
            leaves_in_subtree[node_idx] = 1
        else:
            stack.append((child_l[node_idx], node_idx))
            stack.append((child_r[node_idx], node_idx))
    
    # computes number of leaves in all branches and the overall impurity of
    # the branch. The overall impurity is the sum of r_node in its leaves.
    n_leaves = np.zeros(n_nodes, dtype=np.intp)
    r_branch = np.zeros(n_nodes, dtype=np.float64)
    
    for leaf_idx in range(leaves_in_subtree.shape[0]):
        if not leaves_in_subtree[leaf_idx]:
            continue
        
        r_branch[leaf_idx] = r_node[leaf_idx]
        
        # bubble up values to ancestor nodes
        current_r = r_node[leaf_idx]
        current_idx = leaf_idx
        
        while current_idx != 0:
            parent_idx = parent[current_idx]
            r_branch[parent_idx] += current_r
            n_leaves[parent_idx] += 1
            current_idx = parent_idx
    
    candidate_nodes = ~leaves_in_subtree.astype(bool)
    
    # save metrics before pruning
    controller.save_metrics(0.0, r_branch[0])
    
    # while root node is not a leaf
    while candidate_nodes[0]:
        # computes ccp_alpha for subtrees and finds the minimal alpha
        effective_alpha = np.inf
        pruned_branch_node_idx = -1
        
        for i in range(n_nodes):
            if not candidate_nodes[i]:
                continue
            
            subtree_alpha = (r_node[i] - r_branch[i]) / (n_leaves[i] - 1)
            if subtree_alpha < effective_alpha:
                effective_alpha = subtree_alpha
                pruned_branch_node_idx = i
        
        if controller.stop_pruning(effective_alpha):
            break
        
        # Mark branch for pruning
        node_stack = [pruned_branch_node_idx]
        
        while node_stack:
            node_idx = node_stack.pop()
            
            if not candidate_nodes[node_idx] and node_idx != pruned_branch_node_idx:
                continue
            
            candidate_nodes[node_idx] = False
            leaves_in_subtree[node_idx] = False
            
            if child_l[node_idx] != _TREE_LEAF:
                node_stack.append(child_l[node_idx])
                node_stack.append(child_r[node_idx])
        
        leaves_in_subtree[pruned_branch_node_idx] = True
        
        # updates number of leaves
        n_pruned_leaves = n_leaves[pruned_branch_node_idx] - 1
        n_leaves[pruned_branch_node_idx] = 0
        
        # computes the increase in r_branch to bubble up
        r_diff = r_node[pruned_branch_node_idx] - r_branch[pruned_branch_node_idx]
        r_branch[pruned_branch_node_idx] = r_node[pruned_branch_node_idx]
        
        # bubble up values to ancestors
        node_idx = parent[pruned_branch_node_idx]
        while node_idx != _TREE_UNDEFINED:
            n_leaves[node_idx] -= n_pruned_leaves
            r_branch[node_idx] += r_diff
            node_idx = parent[node_idx]
        
        controller.save_metrics(effective_alpha, r_branch[0])
    
    controller.after_pruning(candidate_nodes)


def _build_pruned_tree_ccp(tree, orig_tree, ccp_alpha):
    """Build a pruned tree from the original tree using cost complexity pruning."""
    
    n_nodes = orig_tree.node_count
    leaves_in_subtree = np.zeros(n_nodes, dtype=np.uint8)
    
    pruning_controller = _AlphaPruner(ccp_alpha=ccp_alpha)
    
    _cost_complexity_prune(leaves_in_subtree, orig_tree, pruning_controller)
    
    _build_pruned_tree(tree, orig_tree, leaves_in_subtree, pruning_controller.capacity)


def ccp_pruning_path(orig_tree):
    """Computes the cost complexity pruning path."""
    leaves_in_subtree = np.zeros(orig_tree.node_count, dtype=np.uint8)
    
    path_finder = _PathFinder(orig_tree.node_count)
    
    _cost_complexity_prune(leaves_in_subtree, orig_tree, path_finder)
    
    total_items = path_finder.count
    ccp_alphas = np.empty(total_items, dtype=np.float64)
    impurities = np.empty(total_items, dtype=np.float64)
    
    for count in range(total_items):
        ccp_alphas[count] = path_finder.ccp_alphas[count]
        impurities[count] = path_finder.impurities[count]
    
    return {
        'ccp_alphas': ccp_alphas,
        'impurities': impurities,
    }


def _build_pruned_tree(tree, orig_tree, leaves_in_subtree, capacity):
    """Build a pruned tree."""
    tree._resize(capacity)
    
    # Build tree using stack
    stack = [(0, 0, _TREE_UNDEFINED, False)]  # (orig_node_id, depth, parent, is_left)
    max_depth_seen = -1
    
    while stack:
        orig_node_id, depth, parent, is_left = stack.pop()
        
        node = orig_tree.nodes[orig_node_id]
        is_leaf = bool(leaves_in_subtree[orig_node_id])
        
        # protect against an infinite loop as a runtime error
        if (not is_leaf and node.left_child == _TREE_LEAF
                and node.right_child == _TREE_LEAF):
            raise ValueError(
                "Node has reached a leaf in the original tree, but is not "
                "marked as a leaf in the leaves_in_subtree mask."
            )
        
        new_node_id = tree._add_node(
            parent, is_left, is_leaf, node.feature, node.threshold,
            node.impurity, node.n_node_samples,
            node.weighted_n_node_samples, node.missing_go_to_left)
        
        if new_node_id == INTPTR_MAX:
            raise MemoryError("pruning tree")
        
        # copy value from original tree to new tree
        orig_value_start = orig_node_id * tree.value_stride
        new_value_start = new_node_id * tree.value_stride
        tree.value[new_value_start:new_value_start + tree.value_stride] = \
            orig_tree.value[orig_value_start:orig_value_start + tree.value_stride]
        
        if not is_leaf:
            # Push children on stack (right first, then left for correct order)
            stack.append((node.right_child, depth + 1, new_node_id, False))
            stack.append((node.left_child, depth + 1, new_node_id, True))
        
        if depth > max_depth_seen:
            max_depth_seen = depth
    
    tree.max_depth = max_depth_seen


def _build_pruned_tree_py(tree, orig_tree, leaves_in_subtree):
    """Build a pruned tree."""
    if leaves_in_subtree.shape[0] != orig_tree.node_count:
        raise ValueError(
            f"The length of leaves_in_subtree {len(leaves_in_subtree)} must be "
            f"equal to the number of nodes in the original tree {orig_tree.node_count}."
        )
    
    _build_pruned_tree(tree, orig_tree, leaves_in_subtree, orig_tree.node_count)
