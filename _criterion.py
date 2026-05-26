# _criterion.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
import numpy as np
import math


def get_unique_times(time, event):
    """Get unique times and event indicators."""
    order = np.argsort(time, kind='mergesort')
    time_sorted = time[order]
    event_sorted = event[order]
    
    unique_times = []
    has_event = []
    
    last_value = None
    for t, e in zip(time_sorted, event_sorted):
        if t != last_value:
            unique_times.append(t)
            has_event.append(e)
            last_value = t
        elif e:
            has_event[-1] = True
    
    return np.array(unique_times), np.array(has_event, dtype=np.bool_)


class RisksetCounter:
    """Counter for riskset statistics."""
    
    def __init__(self, unique_times):
        self.unique_times = unique_times
        self.n_unique_times = len(unique_times)
        self.n_events = np.zeros(self.n_unique_times, dtype=np.float64)
        self.n_at_risk = np.zeros(self.n_unique_times, dtype=np.float64)
        self.data = None
        self.sample_weight = None
    
    def reset(self):
        self.n_events.fill(0.0)
        self.n_at_risk.fill(0.0)
    
    def set_data(self, data, sample_weight):
        self.data = data
        self.sample_weight = sample_weight
    
    def update(self, samples, start, end):
        """Update statistics for samples[start:end]."""
        self.reset()
        
        unique_times = self.unique_times
        y = self.data
        sample_weight = self.sample_weight
        n_times = self.n_unique_times
        
        for i in range(start, end):
            idx = samples[i]
            time, event = y[idx, 0], y[idx, 1]
            
            w = 1.0 if sample_weight is None else sample_weight[idx]
            
            # i-th sample is in all risk sets with time <= i-th time
            ti = 0
            while ti < n_times and unique_times[ti] < time:
                self.n_at_risk[ti] += w
                ti += 1
            
            # Если ti < n_times, то unique_times[ti] >= time
            # В оригинале комментарий говорит unique_times[ti] == time, но это не всегда верно
            # Однако логика такая: добавляем в рисковое множество на момент ti
            if ti < n_times:  # unique_times[ti] >= time
                self.n_at_risk[ti] += w
                # События добавляем только если это точно время события (не цензурирование)
                if event != 0.0:
                    self.n_events[ti] += w
    
    def at(self, index):
        return self.n_at_risk[index], self.n_events[index]


class LogrankCriterion:
    """Log-rank criterion for survival tree splitting."""
    
    def __init__(self, n_outputs, n_samples, unique_times, is_event_time):
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.unique_times = unique_times
        self.is_event_time = is_event_time
        self.n_unique_times = len(unique_times)
        
        # Initialize riskset counter
        self.riskset_total = RisksetCounter(unique_times)
        
        # Buffers
        self.y = None
        self.sample_weight = None
        self.sample_indices = None
        
        # Для хранения статистик для левой ветви
        self.weighted_delta_n_at_risk_left = np.zeros(self.n_unique_times, dtype=np.float64)
        self.weighted_n_events_left = np.zeros(self.n_unique_times, dtype=np.float64)
        
        # Для кэширования индексов времени (как в Cython)
        self.samples_time_idx = np.zeros(n_samples, dtype=np.intp)
        
        # Node state
        self.start = 0
        self.end = 0
        self.pos = 0
        self.n_node_samples = 0
        self.weighted_n_samples = 0.0
        self.weighted_n_node_samples = 0.0
        # Используем приватные атрибуты для свойств
        self._weighted_n_left = 0.0
        self._weighted_n_right = 0.0
        
        # For missing values
        self._missing_go_to_left = False
        self._n_missing = 0
    
    def init(self, y, sample_weight, weighted_n_samples, sample_indices, start, end):
        """Initialize the criterion at node samples[start:end]."""
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        
        # Initialize riskset counter with data
        self.riskset_total.set_data(y, sample_weight)
        self.riskset_total.update(sample_indices, start, end)
        
        # Compute weighted number of node samples и кэшируем индексы времени
        self.weighted_n_node_samples = 0.0
        
        # Pre-compute time indices for all samples (как в Cython)
        unique_times = self.unique_times
        n_times = self.n_unique_times
        
        for i in range(start, end):
            idx = sample_indices[i]
            time = y[idx, 0]
            
            # Бинарный поиск для нахождения индекса времени
            # В Cython это argbinsearch
            lo, hi = 0, n_times
            while lo < hi:
                mid = (lo + hi) // 2
                if unique_times[mid] < time:
                    lo = mid + 1
                else:
                    hi = mid
            
            self.samples_time_idx[idx] = lo
            
            # Добавляем вес
            if sample_weight is not None:
                self.weighted_n_node_samples += sample_weight[idx]
            else:
                self.weighted_n_node_samples += 1.0
        
        # Reset to pos=start
        self.reset()
        
        return 0
    
    def init_missing(self, n_missing):
        """Initialize handling of missing values."""
        self._n_missing = n_missing
        return 0
    
    def init_sum_missing(self):
        """Initialize sum for missing values."""
        return 0
    
    def reset(self):
        """Reset the criterion at pos=start."""
        self._weighted_n_left = 0.0
        self._weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        
        # Reset left statistics
        self.weighted_delta_n_at_risk_left.fill(0.0)
        self.weighted_n_events_left.fill(0.0)
        
        return 0
    
    def reverse_reset(self):
        """Reset the criterion at pos=end."""
        self._weighted_n_right = 0.0
        self._weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0
    
    def update(self, new_pos):
        """Update statistics by moving samples[pos:new_pos] to the left.
        
        ВАЖНО: В Cython update всегда работает от self.start до new_pos,
        а не от self.pos до new_pos!
        """
        samples = self.sample_indices
        y = self.y
        sample_weight = self.sample_weight
        
        # Reset left statistics (как в Cython: memset)
        self.weighted_delta_n_at_risk_left.fill(0.0)
        self.weighted_n_events_left.fill(0.0)
        
        # Update statistics for samples moved to left (ВСЕГДА от start до new_pos)
        self._weighted_n_left = 0.0
        
        for i in range(self.start, new_pos):
            idx = samples[i]
            time_idx = self.samples_time_idx[idx]
            event = y[idx, 1]
            
            w = 1.0 if sample_weight is None else sample_weight[idx]
            
            # В Cython: weighted_delta_n_at_risk_left[time_idx] += w
            # Это показывает изменение в количестве людей в рисковом множестве
            # в момент времени time_idx
            self.weighted_delta_n_at_risk_left[time_idx] += w
            
            # Если это событие (не цензурирование)
            # В оригинале проверка на равенство времени не делается
            if event != 0.0:
                self.weighted_n_events_left[time_idx] += w
            
            self._weighted_n_left += w
        
        self._weighted_n_right = self.weighted_n_node_samples - self._weighted_n_left
        self.pos = new_pos
        
        return 0
    
    def proxy_impurity_improvement(self):
        """Compute a proxy of the impurity reduction.
        
        Возвращает абсолютное значение log-rank статистики.
        """
        weighted_at_risk = self._weighted_n_left
        numer = 0.0
        denom = 0.0
        
        for i in range(self.n_unique_times):
            # События в левой ветви
            events_left = self.weighted_n_events_left[i]
            
            # Всего в рисковом множестве и событий
            total_at_risk, total_events = self.riskset_total.at(i)
            
            if total_at_risk == 0:
                break  # достигли конца
            
            # Доля левой ветви в рисковом множестве
            ratio = weighted_at_risk / total_at_risk
            
            # Числитель: наблюдаемые - ожидаемые события
            numer += events_left - total_events * ratio
            
            # Знаменатель: дисперсия
            if total_at_risk > 1.0:
                v = (total_at_risk - total_events) / (total_at_risk - 1.0) * total_events
                denom += ratio * (1.0 - ratio) * v
            
            # Обновляем количество в рисковом множестве для следующего времени
            weighted_at_risk -= self.weighted_delta_n_at_risk_left[i]
        
        if denom != 0.0:
            # Абсолютное значение log-rank статистики
            return abs(numer / math.sqrt(denom))
        else:  # all samples are censored
            # indicates that this node cannot be split
            return -np.inf
    
    def impurity_improvement(self, impurity_parent, impurity_left, impurity_right, importance_array=None):
        """Compute the improvement in impurity."""
        return self.proxy_impurity_improvement()
    
    def children_impurity(self):
        """Evaluate the impurity in children nodes."""
        return np.inf, np.inf
    
    def node_impurity(self):
        """Evaluate the impurity of the current node."""
        return np.inf
    
    def node_value(self, dest):
        """Compute the node value of samples[start:end] into dest.
        
        В Cython есть два режима:
        1. n_outputs == 1: низкий расход памяти, сохраняем только для времен с событиями
        2. n_outputs > 1: полный режим, сохраняем и CHR и survival для всех времен
        """
        if self.n_outputs == 1:
            # Low memory mode
            # В оригинале: dest[0] накапливает cumulative_chf только в моменты событий
            dest[0] = dest_j0 = 0.0
            
            for i in range(self.n_unique_times):
                n_at_risk, n_events = self.riskset_total.at(i)
                if n_at_risk != 0:
                    ratio = n_events / n_at_risk
                    dest_j0 += ratio
                
                if self.is_event_time[i]:
                    dest[0] += dest_j0
        else:
            # Full mode
            n_at_risk, n_events = self.riskset_total.at(0)
            ratio = n_events / n_at_risk if n_at_risk != 0 else 0.0
            dest[0] = ratio  # Nelson-Aalen estimator
            dest[1] = 1.0 - ratio  # Kaplan-Meier estimator
            
            j = 2
            for i in range(1, self.n_unique_times):
                n_at_risk, n_events = self.riskset_total.at(i)
                dest[j] = dest[j - 2]
                dest[j + 1] = dest[j - 1]
                if n_at_risk != 0:
                    ratio = n_events / n_at_risk
                    dest[j] += ratio
                    dest[j + 1] *= 1.0 - ratio
                j += 2
    
    def middle_value(self):
        """Return middle value for monotonic constraints.
        
        Это для монотонных ограничений.
        """
        return 0.0
    
    def clip_node_value(self, dest, lower_bound, upper_bound):
        """Clip the node value between bounds for monotonic constraints."""
        np.clip(dest, lower_bound, upper_bound, out=dest)
    
    def check_monotonicity(self, monotonic_cst, lower_bound, upper_bound):
        """Check if monotonicity constraints are satisfied."""
        # Для survival trees монотонные ограничения обычно не используются
        return True
    
    # Свойства для совместимости
    @property
    def weighted_n_left(self):
        return self._weighted_n_left
    
    @weighted_n_left.setter
    def weighted_n_left(self, value):
        self._weighted_n_left = value
    
    @property
    def weighted_n_right(self):
        return self._weighted_n_right
    
    @weighted_n_right.setter
    def weighted_n_right(self, value):
        self._weighted_n_right = value
    
    @property
    def weighted_n_node_samples(self):
        return self._weighted_n_node_samples
    
    @weighted_n_node_samples.setter
    def weighted_n_node_samples(self, value):
        self._weighted_n_node_samples = value
    
    @property
    def n_missing(self):
        return self._n_missing
    
    @n_missing.setter 
    def n_missing(self, value):
        self._n_missing = value
    
    @property
    def missing_go_to_left(self):
        return self._missing_go_to_left
    
    @missing_go_to_left.setter
    def missing_go_to_left(self, value):
        self._missing_go_to_left = value

class WeightLogrankCriterion:
    """Weight log-rank criterion for survival tree splitting."""
    
    def __init__(self, n_outputs, n_samples, unique_times, is_event_time, importance_matrix=None):
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.unique_times = unique_times
        self.is_event_time = is_event_time
        self.n_unique_times = len(unique_times)
        self.importance_matrix = importance_matrix
        #self.importance_array = importance_array
        
        # Initialize riskset counter
        self.riskset_total = RisksetCounter(unique_times)
        
        # Buffers
        self.y = None
        self.sample_weight = None
        self.sample_indices = None
        
        # Для хранения статистик для левой ветви
        self.weighted_delta_n_at_risk_left = np.zeros(self.n_unique_times, dtype=np.float64)
        self.weighted_n_events_left = np.zeros(self.n_unique_times, dtype=np.float64)
        
        # Для кэширования индексов времени (как в Cython)
        self.samples_time_idx = np.zeros(n_samples, dtype=np.intp)
        
        # Node state
        self.start = 0
        self.end = 0
        self.pos = 0
        self.n_node_samples = 0
        self.weighted_n_samples = 0.0
        self.weighted_n_node_samples = 0.0
        # Используем приватные атрибуты для свойств
        self._weighted_n_left = 0.0
        self._weighted_n_right = 0.0
        
        # For missing values
        self._missing_go_to_left = False
        self._n_missing = 0
    
    def init(self, y, sample_weight, weighted_n_samples, sample_indices, start, end):
        """Initialize the criterion at node samples[start:end]."""
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        
        # Initialize riskset counter with data
        self.riskset_total.set_data(y, sample_weight)
        self.riskset_total.update(sample_indices, start, end)
        
        # Compute weighted number of node samples и кэшируем индексы времени
        self.weighted_n_node_samples = 0.0
        
        # Pre-compute time indices for all samples (как в Cython)
        unique_times = self.unique_times
        n_times = self.n_unique_times
        
        for i in range(start, end):
            idx = sample_indices[i]
            time = y[idx, 0]
            
            # Бинарный поиск для нахождения индекса времени
            # В Cython это argbinsearch
            lo, hi = 0, n_times
            while lo < hi:
                mid = (lo + hi) // 2
                if unique_times[mid] < time:
                    lo = mid + 1
                else:
                    hi = mid
            
            self.samples_time_idx[idx] = lo
            
            # Добавляем вес
            if sample_weight is not None:
                self.weighted_n_node_samples += sample_weight[idx]
            else:
                self.weighted_n_node_samples += 1.0
        
        # Reset to pos=start
        self.reset()
        
        return 0
    
    def init_missing(self, n_missing):
        """Initialize handling of missing values."""
        self._n_missing = n_missing
        return 0
    
    def init_sum_missing(self):
        """Initialize sum for missing values."""
        return 0
    
    def reset(self):
        """Reset the criterion at pos=start."""
        self._weighted_n_left = 0.0
        self._weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        
        # Reset left statistics
        self.weighted_delta_n_at_risk_left.fill(0.0)
        self.weighted_n_events_left.fill(0.0)
        
        return 0
    
    def reverse_reset(self):
        """Reset the criterion at pos=end."""
        self._weighted_n_right = 0.0
        self._weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0
    
    def update(self, new_pos):
        """Update statistics by moving samples[pos:new_pos] to the left.
        
        ВАЖНО: В Cython update всегда работает от self.start до new_pos,
        а не от self.pos до new_pos!
        """
        samples = self.sample_indices
        y = self.y
        sample_weight = self.sample_weight
        
        # Reset left statistics (как в Cython: memset)
        self.weighted_delta_n_at_risk_left.fill(0.0)
        self.weighted_n_events_left.fill(0.0)
        
        # Update statistics for samples moved to left (ВСЕГДА от start до new_pos)
        self._weighted_n_left = 0.0
        
        for i in range(self.start, new_pos):
            idx = samples[i]
            time_idx = self.samples_time_idx[idx]
            event = y[idx, 1]
            
            w = 1.0 if sample_weight is None else sample_weight[idx]
            
            # В Cython: weighted_delta_n_at_risk_left[time_idx] += w
            # Это показывает изменение в количестве людей в рисковом множестве
            # в момент времени time_idx
            self.weighted_delta_n_at_risk_left[time_idx] += w
            
            # Если это событие (не цензурирование)
            # В оригинале проверка на равенство времени не делается
            if event != 0.0:
                self.weighted_n_events_left[time_idx] += w
            
            self._weighted_n_left += w
        
        self._weighted_n_right = self.weighted_n_node_samples - self._weighted_n_left
        self.pos = new_pos
        
        return 0
    
    def proxy_impurity_improvement(self, importance_array=None):
        """Compute a proxy of the impurity reduction.
        
        Возвращает абсолютное значение log-rank статистики.
        """
        # Use importance_matrix if importance_array is None
        if importance_array is None and hasattr(self, 'importance_matrix') and self.importance_matrix is not None:
            # Важно: получаем вектор для текущего признака из матрицы
            # Это предполагает, что splitter передает правильный importance_array
            # Но в данном случае мы можем использовать и self.importance_matrix, если splitter не передал
            # Для простоты берем entire importance matrix if splitter hasn't specified array
            # Но splitter будет передавать specific feature array
            pass
        elif importance_array is None:
            importance_array = np.ones(self.n_unique_times)

        if importance_array is not None:
            weight_importance = importance_array
        else:
            weight_importance = np.ones(self.n_unique_times)

        # Добавить проверку размерности
        if len(weight_importance) != self.n_unique_times:
            raise ValueError(f"importance_array length {len(weight_importance)} "
                            f"does not match n_unique_times {self.n_unique_times}")
        
        weighted_at_risk = self._weighted_n_left # Y_1j
        numer = 0.0
        denom = 0.0
        
        for i in range(self.n_unique_times):
            weight_val = weight_importance[i]
            weight = 1.0 + weight_val
            
            # События в левой ветви
            events_left = self.weighted_n_events_left[i]
            
            # Всего в рисковом множестве и событий
            total_at_risk, total_events = self.riskset_total.at(i)
            
            if total_at_risk == 0:
                break  # достигли конца
            
            # Доля левой ветви в рисковом множестве
            ratio = weighted_at_risk / total_at_risk
            
            # Числитель: наблюдаемые - ожидаемые события
            numer += (events_left - total_events * ratio) * weight
            
            # Знаменатель: дисперсия
            if total_at_risk > 1.0:
                v = (total_at_risk - total_events) / (total_at_risk - 1.0) * total_events
                denom += ratio * (1.0 - ratio) * v * (weight ** 2)
            
            # Обновляем количество в рисковом множестве для следующего времени
            weighted_at_risk -= self.weighted_delta_n_at_risk_left[i]
        
        if denom != 0.0:
            # Абсолютное значение log-rank статистики
            return abs(numer / math.sqrt(denom))
        else:  # all samples are censored
            # indicates that this node cannot be split
            return -np.inf
    
    def impurity_improvement(self, impurity_parent, impurity_left, impurity_right, importance_array=None):
        """Compute the improvement in impurity."""
        return self.proxy_impurity_improvement(importance_array)
    
    def children_impurity(self):
        """Evaluate the impurity in children nodes."""
        return np.inf, np.inf
    
    def node_impurity(self):
        """Evaluate the impurity of the current node."""
        return np.inf
    
    def node_value(self, dest):
        """Compute the node value of samples[start:end] into dest.
        
        В Cython есть два режима:
        1. n_outputs == 1: низкий расход памяти, сохраняем только для времен с событиями
        2. n_outputs > 1: полный режим, сохраняем и CHR и survival для всех времен
        """
        if self.n_outputs == 1:
            # Low memory mode
            # В оригинале: dest[0] накапливает cumulative_chf только в моменты событий
            dest[0] = dest_j0 = 0.0
            
            for i in range(self.n_unique_times):
                n_at_risk, n_events = self.riskset_total.at(i)
                if n_at_risk != 0:
                    ratio = n_events / n_at_risk
                    dest_j0 += ratio
                
                if self.is_event_time[i]:
                    dest[0] += dest_j0
        else:
            # Full mode
            n_at_risk, n_events = self.riskset_total.at(0)
            ratio = n_events / n_at_risk if n_at_risk != 0 else 0.0
            dest[0] = ratio  # Nelson-Aalen estimator
            dest[1] = 1.0 - ratio  # Kaplan-Meier estimator
            
            j = 2
            for i in range(1, self.n_unique_times):
                n_at_risk, n_events = self.riskset_total.at(i)
                dest[j] = dest[j - 2]
                dest[j + 1] = dest[j - 1]
                if n_at_risk != 0:
                    ratio = n_events / n_at_risk
                    dest[j] += ratio
                    dest[j + 1] *= 1.0 - ratio
                j += 2
    
    def middle_value(self):
        """Return middle value for monotonic constraints.
        
        Это для монотонных ограничений.
        """
        return 0.0
    
    def clip_node_value(self, dest, lower_bound, upper_bound):
        """Clip the node value between bounds for monotonic constraints."""
        np.clip(dest, lower_bound, upper_bound, out=dest)
    
    def check_monotonicity(self, monotonic_cst, lower_bound, upper_bound):
        """Check if monotonicity constraints are satisfied."""
        # Для survival trees монотонные ограничения обычно не используются
        return True
    
    # Свойства для совместимости
    @property
    def weighted_n_left(self):
        return self._weighted_n_left
    
    @weighted_n_left.setter
    def weighted_n_left(self, value):
        self._weighted_n_left = value
    
    @property
    def weighted_n_right(self):
        return self._weighted_n_right
    
    @weighted_n_right.setter
    def weighted_n_right(self, value):
        self._weighted_n_right = value
    
    @property
    def weighted_n_node_samples(self):
        return self._weighted_n_node_samples
    
    @weighted_n_node_samples.setter
    def weighted_n_node_samples(self, value):
        self._weighted_n_node_samples = value
    
    @property
    def n_missing(self):
        return self._n_missing
    
    @n_missing.setter 
    def n_missing(self, value):
        self._n_missing = value
    
    @property
    def missing_go_to_left(self):
        return self._missing_go_to_left
    
    @missing_go_to_left.setter
    def missing_go_to_left(self, value):
        self._missing_go_to_left = value

# Для совместимости с существующим кодом
INFINITY = np.inf
NAN = np.nan
