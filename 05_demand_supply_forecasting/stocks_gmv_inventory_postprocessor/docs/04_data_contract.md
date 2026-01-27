
# 04. Data Contract (Report Schema)

## Input schema (minimum)
`sample_input.csv` mimics a typical SKU-day table.

Required columns:
- date (YYYY-MM-DD)
- sku (int)
- category (str)
- price (float)
- stock (int)
- forecast_gmv (float)   # model output
- actual_units (int)     # factual sales units for evaluation (optional for planning-only mode)

## Derived fields
- forecast_units: floor(forecast_gmv / price)
- feasible_units: min(forecast_units, stock)
- feasible_gmv: feasible_units * price
- actual_gmv: actual_units * price
- lost_sales_units: max(forecast_units - stock, 0)
- lost_sales_gmv: lost_sales_units * price
- is_stock_constrained: forecast_units > stock

## Notes
- If `actual_units` is missing, the pipeline still produces planning outputs (feasible plan + losses),
  but will skip forecast quality metrics.
