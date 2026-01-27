# WAU (Weekly Active Users)

## Context
Metric for weekly product activity based on user submissions.
Source: `default.churn_submits` (ClickHouse).
One row = one submission attempt by a user for a task.

## Goal
Compute WAU for the full period using a rolling 7-day window with a 1-day step.
The current day must be included in the window.

Example:
For day = 2022-09-07, window is 2022-09-01 .. 2022-09-07.

## Output
SQL output columns (strict order):
1) day
2) wau

## Extended analysis (portfolio)
Additionally compute:
- dau (daily active users)
- sticky_factor = dau / wau

## Deliverables
- query.sql: WAU rolling window query
- TECH.md: implementation details and alternatives
- notebooks/analysis.ipynb: DAU vs WAU plot and sticky factor analysis
