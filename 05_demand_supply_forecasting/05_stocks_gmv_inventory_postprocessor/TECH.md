
# TECH.md

## Why this solution is operational (not a homework)
The key difference is **governance**:
- we produce an executable plan (feasible_units/gmv), not just a forecast
- we quantify the business impact of availability (lost sales)
- we separate model quality from supply constraints (non-OOS evaluation)
- we generate alerts and route actions by ownership

## Key implementation choices
1) Integer units with floor
Retail sales units are integers. Floor is conservative and aligns with planning risk control:
over-forecasting units may lead to unrealistic plans and wrong replenishment decisions.

2) Supply constraint application
`feasible_units = min(forecast_units, stock)`
This converts demand forecast to an executable plan under current availability.

3) Lost sales estimation
We estimate censored demand by:
`lost_units = max(forecast_units - stock, 0)`
This is a practical proxy used for operational triage. For higher maturity,
we can replace it with demand models that infer latent demand using availability signals.

4) Non-OOS evaluation for forecast quality
Quality metrics are computed on subset where `forecast_units <= stock`.
This avoids punishing the forecast for supply shortfalls.

## Extensibility
- plug real sources (SAP BW, DWH) via SQL templates in `sql/`
- publish outputs to Power BI / Redash
- add promo calendars and vendor lead times
