# RandomSurvivalForest with Time-Dependent Feature Importance

This is an enhanced version of Random Survival Forest that supports **time-varying feature importance** using splines.

## New Parameter: `spline_importance`

**Type**: `dict[str, scipy.interpolate.UnivariateSpline]` or `None`

- If `None` (default) — uses standard Logrank criterion.
- If dict is provided — each tree builds its own importance matrix based on its bootstrap sample and unique times.

### Example

```python
from scipy.interpolate import UnivariateSpline
from sksurv_importance import RandomSurvivalForest

# Example spline dictionary
spline_importance = {
    'age': UnivariateSpline(time_points, importance_age_values, k=3),
    'bmi': UnivariateSpline(time_points, importance_bmi_values, k=3),
    # add all features
}

rsf = RandomSurvivalForest(
    n_estimators=100,
    spline_importance=spline_importance,
    max_features='sqrt',
    random_state=42
)

rsf.fit(X, y)
```

## How it works

- For each tree:
  - Computes its own `unique_times_` from bootstrap sample
  - Builds importance matrix `(n_relevant_features, n_unique_times)`
  - Uses `WeightLogrankCriterion` with weights from splines

## Files structure

- `sksurv_importance/forest.py`
- `sksurv_importance/tree.py`
- `sksurv_importance/_criterion.py`
- `sksurv_importance/_splitter.py`

---

**Note**: Splines must be callable: `spline(time)` returns importance at that time.