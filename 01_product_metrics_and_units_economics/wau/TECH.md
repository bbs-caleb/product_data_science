# Technical Notes

## Why window functions
Rolling WAU via window aggregates is O(N) over ordered days.
Self-join implementations typically scale worse and are harder to maintain.

## Important detail: distinct users in a rolling window
A naive approach "sum DAU over 7 days" is incorrect because users repeat across days.
We use uniqExact(user_id) over a date-ordered RANGE window.

## Data normalization
We pre-deduplicate to one row per (day, user_id) to reduce input volume and avoid intra-day duplicates.

## Alternatives
1) Self-join on day between (d-6..d) and countDistinct(user_id) by day.
2) Approximate distinct: uniqCombined / uniqHLL12 for large-scale monitoring (trade accuracy for speed).

