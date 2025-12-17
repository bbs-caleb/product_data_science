# Cold Start Imputation: Group Mean (Floor)

## Context
New entities (e.g., new products/users) have missing target values (NaN).
Downstream models/pipelines require fully populated targets.

## Goal
Implement a deterministic imputation:
- fill NaNs in `target` with the per-`group` mean of `target`
- apply floor() to the imputed value (integer-like business rule)

## Contract
- Input dataframe is not mutated (function returns a copy)
- Non-null values in `target` are preserved
- If a group's mean is NaN (all values missing), NaNs remain NaN
