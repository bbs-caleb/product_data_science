
# 02. Operating Model

## Cadence
- Daily (or weekly) refresh for regular forecasting
- Special refresh for promo windows (SR/LR)

## Inputs (minimum viable)
- Forecast: by SKU-day (units or GMV)
- Price: by SKU-day
- Availability: stock on hand (or available-to-sell) by SKU-day
- Actual sales: by SKU-day (units/GMV)

## Steps
1. Data validation (schema + basic sanity checks)
2. Convert forecast to units (integer, floor)
3. Apply stock constraint:
   feasible_units = min(forecast_units, stock)
4. Compute feasible GMV:
   feasible_gmv = feasible_units * price
5. Lost sales estimation (proxy):
   lost_units = max(forecast_units - stock, 0)
   lost_gmv = lost_units * price
6. Forecast quality split:
   - raw: compare forecast vs actual
   - non-OOS: compare forecast vs actual only where not constrained
7. Alerts + action routing
8. Publish tables to BI (Power BI / Redash) and share weekly summary

## RACI (simplified)
- Planning Analyst: owns report, monitors losses and bias
- Supply/Logistics: owns actions on OOS and redistribution
- Commercial/Marketing: owns promo adjustments/substitutions
- DS/Forecast Owner: owns model improvements when non-OOS quality degrades
- DWH/IT: owns data quality incidents
