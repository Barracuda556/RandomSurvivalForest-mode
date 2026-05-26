# _utils.py
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from math import log as math_log

# =============================================================================
# Helper functions
# =============================================================================

def safe_realloc(p, nelems):
    """Safe reallocation that checks for overflow."""
    if nelems < 0:
        raise MemoryError(f"negative number of elements: {nelems}")
    return nelems


def _realloc_test():
    pass


def sizet_ptr_to_ndarray(data, size):
    """Return copied data as 1D numpy array of intp's."""
    if data is None or size == 0:
        return np.empty(0, dtype=np.intp)
    return np.array(data[:size], dtype=np.intp)


# КОНСТАНТА из C кода
RAND_R_MAX = np.uint32(2**31 - 1)  # Максимальное значение для rand_r


def our_rand_r(state_ptr):
    """Python implementation of C's rand_r function using numpy.
    
    Это линейный конгруэнтный генератор, как в стандартной библиотеке C.
    Формула: state = state * 1103515245 + 12345
    
    Используем numpy uint32 для точной эмуляции 32-битной арифметики C.
    """
    # Преобразуем состояние в numpy uint32 для автоматической 32-битной арифметики
    state = np.uint32(state_ptr[0])
    
    # Константы как numpy uint32
    multiplier = np.uint32(1103515245)
    increment = np.uint32(12345)
    
    # Умножение и сложение с 32-битным переполнением
    # numpy автоматически обрабатывает переполнение для uint32
    # Подавляем предупреждение о переполнении, так как это ожидаемое поведение
    with np.errstate(over='ignore'):
        state = state * multiplier + increment
    
    # Сохраняем новое состояние как обычное Python int
    state_ptr[0] = int(state)
    
    # Возвращаем только положительные значения (31 бит)
    # Это соответствует: return state & 0x7FFFFFFF в C
    return int(state & np.uint32(0x7FFFFFFF))


def rand_int(low, high, random_state_ptr):
    """Generate a random integer in [low; high).
    
    Parameters
    ----------
    low : int
        Lower bound (inclusive)
    high : int
        Upper bound (exclusive)
    random_state_ptr : list or array with one element
        Pointer to random state that gets updated
    """
    if high <= low:
        return low
    
    # Используем our_rand_r для генерации случайного числа
    random_val = our_rand_r(random_state_ptr)
    return low + random_val % (high - low)


def rand_uniform(low, high, random_state_ptr):
    """Generate a random float64_t in [low; high).
    
    Parameters
    ----------
    low : float
        Lower bound (inclusive)
    high : float
        Upper bound (exclusive)
    random_state_ptr : list or array with one element
        Pointer to random state that gets updated
    """
    if high == low:
        return low
    
    # Генерируем случайное число и обновляем состояние
    random_val = our_rand_r(random_state_ptr)
    
    # Преобразуем в float и масштабируем в диапазон [low, high)
    # Используем RAND_R_MAX = 2**31 - 1 для соответствия C rand_r
    return ((high - low) * float(random_val) / float(RAND_R_MAX)) + low


def log(x):
    """Base-2 logarithm."""
    if x <= 0:
        return -np.inf
    return math_log(x) / math_log(2.0)


# =============================================================================
# WeightedPQueue data structure
# =============================================================================

class WeightedPQueueRecord:
    __slots__ = ('data', 'weight')
    
    def __init__(self, data=0.0, weight=0.0):
        self.data = data
        self.weight = weight
    
    def __repr__(self):
        return f"WeightedPQueueRecord(data={self.data:.4f}, weight={self.weight:.4f})"


class WeightedPQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.array_ptr = 0
        self.array_ = []

    def reset(self):
        self.array_ptr = 0
        self.array_ = []
        return 0

    def is_empty(self):
        return self.array_ptr <= 0

    def size(self):
        return self.array_ptr

    def push(self, data, weight):
        record = WeightedPQueueRecord(data, weight)
        
        if self.array_ptr >= self.capacity:
            self.capacity *= 2
        
        # Вставляем и поддерживаем отсортированный порядок
        if not self.array_ or data >= self.array_[-1].data:
            self.array_.append(record)
        else:
            # Найти позицию для вставки
            pos = 0
            while pos < len(self.array_) and self.array_[pos].data < data:
                pos += 1
            self.array_.insert(pos, record)
        
        self.array_ptr += 1
        return 0

    def remove(self, data, weight):
        if self.array_ptr <= 0:
            return -1
        
        for i, record in enumerate(self.array_):
            if abs(record.data - data) < 1e-12 and abs(record.weight - weight) < 1e-12:
                del self.array_[i]
                self.array_ptr -= 1
                return 0
        return -1

    def pop(self):
        if self.array_ptr <= 0:
            return None
        
        record = self.array_[0]
        del self.array_[0]
        self.array_ptr -= 1
        return record.data, record.weight

    def peek(self):
        if self.array_ptr <= 0:
            return None
        record = self.array_[0]
        return record.data, record.weight

    def get_weight_from_index(self, index):
        if index < 0 or index >= self.array_ptr:
            raise IndexError(f"Index {index} out of bounds [0, {self.array_ptr})")
        return self.array_[index].weight

    def get_value_from_index(self, index):
        if index < 0 or index >= self.array_ptr:
            raise IndexError(f"Index {index} out of bounds [0, {self.array_ptr})")
        return self.array_[index].data


# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

class WeightedMedianCalculator:
    def __init__(self, initial_capacity):
        self.initial_capacity = initial_capacity
        self.samples = WeightedPQueue(initial_capacity)
        self.total_weight = 0.0
        self.k = 0
        self.sum_w_0_k = 0.0

    def size(self):
        return self.samples.size()

    def reset(self):
        self.samples.reset()
        self.total_weight = 0.0
        self.k = 0
        self.sum_w_0_k = 0.0
        return 0

    def push(self, data, weight):
        original_median = 0.0
        if self.size() != 0:
            original_median = self.get_median()
        
        return_value = self.samples.push(data, weight)
        self.update_median_parameters_post_push(data, weight, original_median)
        return return_value

    def update_median_parameters_post_push(self, data, weight, original_median):
        if self.size() == 1:
            self.k = 1
            self.total_weight = weight
            self.sum_w_0_k = self.total_weight
            return 0

        self.total_weight += weight

        if data < original_median:
            self.k += 1
            self.sum_w_0_k += weight

            while (self.k > 1 and 
                   ((self.sum_w_0_k - self.samples.get_weight_from_index(self.k-1))
                    >= self.total_weight / 2.0)):
                self.k -= 1
                self.sum_w_0_k -= self.samples.get_weight_from_index(self.k)
            return 0

        if data >= original_median:
            while (self.k < self.samples.size() and
                   (self.sum_w_0_k < self.total_weight / 2.0)):
                self.k += 1
                self.sum_w_0_k += self.samples.get_weight_from_index(self.k-1)
            return 0

    def remove(self, data, weight):
        original_median = 0.0
        if self.size() != 0:
            original_median = self.get_median()

        return_value = self.samples.remove(data, weight)
        self.update_median_parameters_post_remove(data, weight, original_median)
        return return_value

    def pop(self):
        if self.size() == 0:
            return None, None, -1

        original_median = self.get_median()
        
        if self.samples.size() == 0:
            return None, None, -1

        result = self.samples.pop()
        if result is None:
            return None, None, -1
            
        data, weight = result
        self.update_median_parameters_post_remove(data, weight, original_median)
        return data, weight, 0

    def update_median_parameters_post_remove(self, data, weight, original_median):
        if self.samples.size() == 0:
            self.k = 0
            self.total_weight = 0.0
            self.sum_w_0_k = 0.0
            return 0

        if self.samples.size() == 1:
            self.k = 1
            self.total_weight -= weight
            self.sum_w_0_k = self.total_weight
            return 0

        self.total_weight -= weight

        if data < original_median:
            self.k -= 1
            self.sum_w_0_k -= weight

            while (self.k < self.samples.size() and
                   (self.sum_w_0_k < self.total_weight / 2.0)):
                self.k += 1
                self.sum_w_0_k += self.samples.get_weight_from_index(self.k-1)
            return 0

        if data >= original_median:
            while (self.k > 1 and 
                   ((self.sum_w_0_k - self.samples.get_weight_from_index(self.k-1))
                    >= self.total_weight / 2.0)):
                self.k -= 1
                self.sum_w_0_k -= self.samples.get_weight_from_index(self.k)
            return 0

    def get_median(self):
        if self.samples.size() == 0:
            return 0.0
            
        if abs(self.sum_w_0_k - (self.total_weight / 2.0)) < 1e-12:
            return (self.samples.get_value_from_index(self.k) +
                    self.samples.get_value_from_index(self.k-1)) / 2.0
        elif self.sum_w_0_k > (self.total_weight / 2.0):
            return self.samples.get_value_from_index(self.k-1)
        else:
            return self.samples.get_value_from_index(min(self.k, self.samples.size() - 1))


def _any_isnan_axis0(X):
    """Same as np.any(np.isnan(X), axis=0)"""
    if isinstance(X, np.ndarray):
        return np.any(np.isnan(X), axis=0)
    else:
        # Для memory views или других типов
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        isnan_out = np.zeros(n_features, dtype=np.bool_)
        for j in range(n_features):
            if len(X.shape) > 1:
                for i in range(X.shape[0]):
                    if np.isnan(X[i, j]):
                        isnan_out[j] = True
                        break
            else:
                if np.isnan(X[j]):
                    isnan_out[j] = True
        return isnan_out


class RandomState:
    """Random state that emulates C's rand_r behavior using numpy."""
    def __init__(self, seed=None):
        if seed is None:
            # Используем numpy для генерации случайного числа
            seed = np.random.randint(0, int(RAND_R_MAX))
        elif isinstance(seed, np.random.RandomState):
            # Если передали numpy RandomState, берем его seed
            seed = seed.randint(0, int(RAND_R_MAX))
        elif isinstance(seed, np.integer):
            # Если передали numpy integer
            seed = int(seed)
        
        # Состояние хранится как одноэлементный список для передачи по "указателю"
        # Инициализируем как numpy uint32 для согласованности
        self.state = [np.uint32(seed)]
    
    def randint(self, low, high=None):
        if high is None:
            high = low
            low = 0
        if high <= low:
            return low
        return rand_int(low, high, self.state)
    
    def uniform(self, low=0.0, high=1.0):
        if high == low:
            return low
        return rand_uniform(low, high, self.state)
    
    def __mod__(self, other):
        return int(self.state[0]) % other
    
    @property
    def state_value(self):
        return int(self.state[0])
    
    @state_value.setter
    def state_value(self, value):
        self.state[0] = np.uint32(value)
