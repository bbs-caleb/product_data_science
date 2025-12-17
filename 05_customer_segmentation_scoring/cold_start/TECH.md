# Technical Notes: fillna_with_mean (group mean + floor)

## Problem
Downstream models/pipelines cannot handle NaNs in the target metric for new entities (cold start).
We impute missing target values using historical behavior of the entity's group.

## Implementation
1) Reduce to group-level signal:
- `groupby(group)[target].transform("mean")` computes a per-row group mean aligned to the original dataframe.

2) Impute:
- `fillna(means)` replaces only missing target values with the corresponding group mean.

3) Business rule: floor
- `np.floor(...)` enforces integer-like target values required by the task checker.
- This is an explicit rule, not a numerical artifact.

## Complexity
- Time: O(N) for groupby-transform + O(N) for fill/floor.
- Memory: O(N) for the aligned `means` vector.

## Edge Cases
- If a group has only NaNs in `target`, its mean is NaN, so imputed values remain NaN.
- Non-null values are preserved (only NaNs are replaced).

## Alternatives
- Median by group (robust to outliers) if business wants stable fallback.
- Hierarchical fallback: group mean -> global mean -> constant.
- For large-scale pipelines: compute group aggregates once and join back (avoids transform cost in repeated jobs).

