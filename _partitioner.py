"""Partition samples in the construction of a tree.

This module contains the algorithms for moving sample indices to
the left and right child node given a split determined by the
splitting algorithm in `_splitter.pyx`.

Partitioning is done in a way that is efficient for both dense data,
and sparse data stored in a Compressed Sparse Column (CSC) format.
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from math import log2
from scipy.sparse import issparse, csc_matrix


# Constant to switch between algorithm non zero value extract algorithm
# in SparsePartitioner
EXTRACT_NNZ_SWITCH = 0.1

# Allow for 32 bit float comparisons
INFINITY_32t = np.inf

# Feature threshold for considering values equal
FEATURE_THRESHOLD = 1e-7


class DensePartitioner:
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    def __init__(
        self,
        X,
        samples,
        feature_values,
        missing_values_in_feature_mask,
    ):
        self.X = X
        self.samples = samples
        self.feature_values = feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask
        self.start = 0
        self.end = 0
        self.n_missing = 0

    def init_node_split(self, start, end):
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.n_missing = 0

    def sort_samples_and_feature_values(self, current_feature):
        """Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """
        X = self.X
        samples = self.samples
        feature_values = self.feature_values
        missing_values_in_feature_mask = self.missing_values_in_feature_mask
        
        n_missing = 0
        
        # Sort samples along that feature; by copying the values into an array and
        # sorting the array in a manner which utilizes the cache more effectively.
        if (missing_values_in_feature_mask is not None and 
            missing_values_in_feature_mask[current_feature]):
            
            i = self.start
            current_end = self.end - 1
            
            # Missing values are placed at the end and do not participate in the sorting.
            while i <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values at its left.
                if np.isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if np.isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1

                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            # When there are no missing values, we only need to copy the data into
            # feature_values
            for i in range(self.start, self.end):
                feature_values[i] = X[samples[i], current_feature]

        n_to_sort = self.end - self.start - n_missing
        if n_to_sort > 0:
            self._sort(feature_values, samples, self.start, n_to_sort)
        
        self.n_missing = n_missing

    def find_min_max(self, current_feature):
        """Find the minimum and maximum value for current_feature.

        Missing values are stored at the end of feature_values. The number of missing
        values observed in feature_values is stored in self.n_missing.
        """
        X = self.X
        samples = self.samples
        feature_values = self.feature_values
        missing_values_in_feature_mask = self.missing_values_in_feature_mask
        
        min_feature_value = INFINITY_32t
        max_feature_value = -INFINITY_32t
        n_missing = 0

        # We are copying the values into an array and finding min/max of the array in
        # a manner which utilizes the cache more effectively. We need to also count
        # the number of missing-values there are.
        if (missing_values_in_feature_mask is not None and 
            missing_values_in_feature_mask[current_feature]):
            
            p = self.start
            current_end = self.end - 1
            
            # Missing values are placed at the end and do not participate in the
            # min/max calculation.
            while p <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values towards its left.
                if np.isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if np.isnan(X[samples[p], current_feature]):
                    samples[p], samples[current_end] = samples[current_end], samples[p]
                    n_missing += 1
                    current_end -= 1

                current_feature_value = X[samples[p], current_feature]
                feature_values[p] = current_feature_value
                
                if current_feature_value < min_feature_value:
                    min_feature_value = current_feature_value
                elif current_feature_value > max_feature_value:
                    max_feature_value = current_feature_value
                
                p += 1
        else:
            min_feature_value = X[samples[self.start], current_feature]
            max_feature_value = min_feature_value
            feature_values[self.start] = min_feature_value
            
            for p in range(self.start + 1, self.end):
                current_feature_value = X[samples[p], current_feature]
                feature_values[p] = current_feature_value

                if current_feature_value < min_feature_value:
                    min_feature_value = current_feature_value
                elif current_feature_value > max_feature_value:
                    max_feature_value = current_feature_value

        self.n_missing = n_missing
        return min_feature_value, max_feature_value

    def next_p(self, p_prev, p):
        """Compute the next p_prev and p for iterating over feature values.

        The missing values are not included when iterating through the feature values.
        """
        feature_values = self.feature_values
        end_non_missing = self.end - self.n_missing
        
        # Используем изменяемые объекты для p и p_prev
        p_val = p
        p_prev_val = p_prev
        
        while (p_val + 1 < end_non_missing and
               feature_values[p_val + 1] <= feature_values[p_val] + FEATURE_THRESHOLD):
            p_val += 1

        p_prev_val = p_val

        # By adding 1, we have
        # (feature_values[p] >= end) or (feature_values[p] > feature_values[p - 1])
        p_val += 1
        
        return p_prev_val, p_val

    def partition_samples(self, current_threshold):
        """Partition samples for feature_values at the current_threshold."""
        p = self.start
        partition_end = self.end - self.n_missing
        samples = self.samples
        feature_values = self.feature_values

        while p < partition_end:
            if feature_values[p] <= current_threshold:
                p += 1
            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )
                samples[p], samples[partition_end] = samples[partition_end], samples[p]

        return partition_end

    def partition_samples_final(
        self,
        best_pos,
        best_threshold,
        best_feature,
        best_n_missing,
    ):
        """Partition samples for X at the best_threshold and best_feature.

        If missing values are present, this method partitions `samples`
        so that the `best_n_missing` missing values' indices are in the
        right-most end of `samples`, that is `samples[end_non_missing:end]`.
        """
        # Local invariance: start <= p <= partition_end <= end
        start = self.start
        p = start
        end = self.end - 1
        partition_end = end - best_n_missing
        samples = self.samples
        X = self.X

        if best_n_missing != 0:
            # Move samples with missing values to the end while partitioning the
            # non-missing samples
            while p < partition_end:
                # Keep samples with missing values at the end
                if np.isnan(X[samples[end], best_feature]):
                    end -= 1
                    continue

                # Swap sample with missing values with the sample at the end
                current_value = X[samples[p], best_feature]
                if np.isnan(current_value):
                    samples[p], samples[end] = samples[end], samples[p]
                    end -= 1

                    # The swapped sample at the end is always a non-missing value, so
                    # we can continue the algorithm without checking for missingness.
                    current_value = X[samples[p], best_feature]

                # Partition the non-missing samples
                if current_value <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1
        else:
            # Partitioning routine when there are no missing values
            while p < partition_end:
                if X[samples[p], best_feature] <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1

    def _sort(self, feature_values, samples, start, n):
        """Sort n-element arrays feature_values and samples simultaneously."""
        if n == 0:
            return
        
        # Copy the slices to sort
        fv_slice = feature_values[start:start + n].copy()
        s_slice = samples[start:start + n].copy()
        
        # Get sorted indices
        sorted_idx = np.argsort(fv_slice)
        
        # Apply sorting to both arrays
        feature_values[start:start + n] = fv_slice[sorted_idx]
        samples[start:start + n] = s_slice[sorted_idx]


class SparsePartitioner:
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    def __init__(
        self,
        X,
        samples,
        n_samples,
        feature_values,
        missing_values_in_feature_mask,
    ):
        if not (issparse(X) and X.format == "csc"):
            raise ValueError("X should be in csc format")

        self.samples = samples
        self.feature_values = feature_values

        # Initialize X
        n_total_samples = X.shape[0]

        self.X_data = X.data
        self.X_indices = X.indices
        self.X_indptr = X.indptr
        self.n_total_samples = n_total_samples

        # Initialize auxiliary array used to perform split
        self.index_to_samples = np.full(n_total_samples, fill_value=-1, dtype=np.intp)
        self.sorted_samples = np.empty(n_samples, dtype=np.intp)

        for p in range(n_samples):
            self.index_to_samples[samples[p]] = p

        self.missing_values_in_feature_mask = missing_values_in_feature_mask
        
        # State variables
        self.start = 0
        self.end = 0
        self.is_samples_sorted = 0
        self.n_missing = 0
        self.end_negative = 0
        self.start_positive = 0

    def init_node_split(self, start, end):
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.is_samples_sorted = 0
        self.n_missing = 0

    def sort_samples_and_feature_values(self, current_feature):
        """Simultaneously sort based on the feature_values."""
        feature_values = self.feature_values
        index_to_samples = self.index_to_samples
        samples = self.samples

        self.extract_nnz(current_feature)
        
        # Sort the positive and negative parts of `feature_values`
        n_negative = self.end_negative - self.start
        if n_negative > 0:
            self._sort(feature_values, samples, self.start, n_negative)
        
        n_positive = self.end - self.start_positive
        if n_positive > 0:
            self._sort(feature_values, samples, self.start_positive, n_positive)

        # Update index_to_samples to take into account the sort
        for p in range(self.start, self.end_negative):
            index_to_samples[samples[p]] = p
        for p in range(self.start_positive, self.end):
            index_to_samples[samples[p]] = p

        # Add one or two zeros in feature_values, if there is any
        if self.end_negative < self.start_positive:
            self.start_positive -= 1
            feature_values[self.start_positive] = 0.

            if self.end_negative != self.start_positive:
                feature_values[self.end_negative] = 0.
                self.end_negative += 1

        # XXX: When sparse supports missing values, this should be set to the
        # number of missing values for current_feature
        self.n_missing = 0

    def find_min_max(self, current_feature):
        """Find the minimum and maximum value for current_feature."""
        self.extract_nnz(current_feature)

        if self.end_negative != self.start_positive:
            # There is a zero
            min_feature_value = 0
            max_feature_value = 0
        else:
            min_feature_value = self.feature_values[self.start]
            max_feature_value = min_feature_value

        # Find min, max in feature_values[start:end_negative]
        for p in range(self.start, self.end_negative):
            current_feature_value = self.feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        # Update min, max given feature_values[start_positive:end]
        for p in range(self.start_positive, self.end):
            current_feature_value = self.feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        return min_feature_value, max_feature_value

    def next_p(self, p_prev, p):
        """Compute the next p_prev and p for iterating over feature values."""
        # Используем изменяемые объекты для p и p_prev
        p_val = p
        p_prev_val = p_prev
        
        if p_val + 1 != self.end_negative:
            p_next = p_val + 1
        else:
            p_next = self.start_positive

        while (p_next < self.end and
               self.feature_values[p_next] <= self.feature_values[p_val] + FEATURE_THRESHOLD):
            p_val = p_next
            if p_val + 1 != self.end_negative:
                p_next = p_val + 1
            else:
                p_next = self.start_positive

        p_prev_val = p_val
        p_val = p_next
        
        return p_prev_val, p_val

    def partition_samples(self, current_threshold):
        """Partition samples for feature_values at the current_threshold."""
        return self._partition(current_threshold, self.start_positive)

    def partition_samples_final(
        self,
        best_pos,
        best_threshold,
        best_feature,
        n_missing,
    ):
        """Partition samples for X at the best_threshold and best_feature."""
        self.extract_nnz(best_feature)
        self._partition(best_threshold, best_pos)

    def _partition(self, threshold, zero_pos):
        """Partition samples[start:end] based on threshold."""
        index_to_samples = self.index_to_samples
        feature_values = self.feature_values
        samples = self.samples

        if threshold < 0.:
            p = self.start
            partition_end = self.end_negative
        elif threshold > 0.:
            p = self.start_positive
            partition_end = self.end
        else:
            # Data are already split
            return zero_pos

        while p < partition_end:
            if feature_values[p] <= threshold:
                p += 1
            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )
                self._sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    def extract_nnz(self, feature):
        """Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        feature_values[start:end_negative[0]] and positive values
        feature_values[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.
        """
        samples = self.samples
        feature_values = self.feature_values
        indptr_start = self.X_indptr[feature]
        indptr_end = self.X_indptr[feature + 1]
        n_indices = indptr_end - indptr_start
        n_samples = self.end - self.start
        index_to_samples = self.index_to_samples
        sorted_samples = self.sorted_samples
        X_indices = self.X_indices
        X_data = self.X_data

        # Use binary search if n_samples * log(n_indices) <
        # n_indices and index_to_samples approach otherwise.
        if ((1 - self.is_samples_sorted) * n_samples * log2(n_samples) +
                n_samples * log2(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
            self._extract_nnz_binary_search(
                X_indices, X_data, indptr_start, indptr_end,
                samples, self.start, self.end, index_to_samples,
                feature_values, sorted_samples
            )
        else:
            self._extract_nnz_index_to_samples(
                X_indices, X_data, indptr_start, indptr_end,
                samples, self.start, self.end, index_to_samples,
                feature_values
            )

    def _extract_nnz_index_to_samples(self, X_indices, X_data,
                                      indptr_start, indptr_end,
                                      samples, start, end,
                                      index_to_samples, feature_values):
        """Extract and partition values for a feature using index_to_samples."""
        end_negative_ = start
        start_positive_ = end

        for k in range(indptr_start, indptr_end):
            idx = X_indices[k]
            if start <= index_to_samples[idx] < end:
                if X_data[k] > 0:
                    start_positive_ -= 1
                    feature_values[start_positive_] = X_data[k]
                    index = index_to_samples[idx]
                    self._sparse_swap(index_to_samples, samples, index, start_positive_)
                elif X_data[k] < 0:
                    feature_values[end_negative_] = X_data[k]
                    index = index_to_samples[idx]
                    self._sparse_swap(index_to_samples, samples, index, end_negative_)
                    end_negative_ += 1

        self.end_negative = end_negative_
        self.start_positive = start_positive_

    def _extract_nnz_binary_search(self, X_indices, X_data,
                                   indptr_start, indptr_end,
                                   samples, start, end,
                                   index_to_samples, feature_values,
                                   sorted_samples):
        """Extract and partition values for a given feature using binary search."""
        n_samples = end - start

        if not self.is_samples_sorted:
            sorted_samples[start:end] = samples[start:end]
            sorted_samples[start:end].sort()
            self.is_samples_sorted = 1

        while (indptr_start < indptr_end and
               sorted_samples[start] > X_indices[indptr_start]):
            indptr_start += 1

        while (indptr_start < indptr_end and
               sorted_samples[end - 1] < X_indices[indptr_end - 1]):
            indptr_end -= 1

        p = start
        end_negative_ = start
        start_positive_ = end

        while p < end and indptr_start < indptr_end:
            # Find index of sorted_samples[p] in X_indices using binary search
            target = sorted_samples[p]
            k = self._binary_search(X_indices, indptr_start, indptr_end, target)
            
            if k != -1:
                # If k != -1, we have found a non zero value
                if X_data[k] > 0:
                    start_positive_ -= 1
                    feature_values[start_positive_] = X_data[k]
                    index = index_to_samples[X_indices[k]]
                    self._sparse_swap(index_to_samples, samples, index, start_positive_)
                elif X_data[k] < 0:
                    feature_values[end_negative_] = X_data[k]
                    index = index_to_samples[X_indices[k]]
                    self._sparse_swap(index_to_samples, samples, index, end_negative_)
                    end_negative_ += 1
            p += 1

        self.end_negative = end_negative_
        self.start_positive = start_positive_

    def _binary_search(self, sorted_array, start, end, value):
        """Return the index of value in the sorted array using binary search."""
        while start < end:
            pivot = start + (end - start) // 2
            
            if sorted_array[pivot] == value:
                return pivot
            
            if sorted_array[pivot] < value:
                start = pivot + 1
            else:
                end = pivot
        return -1

    def _sparse_swap(self, index_to_samples, samples, pos_1, pos_2):
        """Swap sample pos_1 and pos_2 preserving sparse invariant."""
        samples[pos_1], samples[pos_2] = samples[pos_2], samples[pos_1]
        index_to_samples[samples[pos_1]] = pos_1
        index_to_samples[samples[pos_2]] = pos_2

    def _sort(self, feature_values, samples, start, n):
        """Sort n-element arrays feature_values and samples simultaneously."""
        if n == 0:
            return
        
        # Copy the slices to sort
        fv_slice = feature_values[start:start + n].copy()
        s_slice = samples[start:start + n].copy()
        
        # Get sorted indices
        sorted_idx = np.argsort(fv_slice)
        
        # Apply sorting to both arrays
        feature_values[start:start + n] = fv_slice[sorted_idx]
        samples[start:start + n] = s_slice[sorted_idx]


def shift_missing_values_to_left_if_required(best, samples, end):
    """Shift missing value sample indices to the left of the split if required.

    Note: this should always be called at the very end because it will
    move samples around, thereby affecting the criterion.
    This affects the computation of the children impurity, which affects
    the computation of the next node.
    """
    # The partitioner partitions the data such that the missing values are in
    # samples[-n_missing:] for the criterion to consume. If the missing values
    # are going to the right node, then the missing values are already in the
    # correct position. If the missing values go left, then we move the missing
    # values to samples[best.pos:best.pos+n_missing] and update `best.pos`.
    if best.n_missing > 0 and best.missing_go_to_left:
        for p in range(best.n_missing):
            i = best.pos + p
            current_end = end - 1 - p
            samples[i], samples[current_end] = samples[current_end], samples[i]
        best.pos += best.n_missing


def _py_sort(feature_values, samples, n):
    """Used for testing sort."""
    if n == 0:
        return
    
    # Copy arrays for sorting
    fv_copy = feature_values[:n].copy()
    s_copy = samples[:n].copy()
    
    # Get sorted indices
    sorted_idx = np.argsort(fv_copy)
    
    # Apply sorting
    feature_values[:n] = fv_copy[sorted_idx]
    samples[:n] = s_copy[sorted_idx]
