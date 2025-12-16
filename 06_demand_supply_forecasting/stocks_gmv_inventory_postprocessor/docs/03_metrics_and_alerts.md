
# 03. Metrics and Alerts

## Core metrics (management-level)
- OOS_rate: share of SKU-days where forecast_units > stock
- LostSales_GMV: sum(lost_sales_gmv) for the period
- WAPE_non_OOS: weighted absolute percentage error on non-OOS subset
- Bias_non_OOS: signed error on non-OOS subset (systematic under/over-forecast)

### Why "non-OOS" matters
If the shelf is empty, the model cannot be “right” against sales. We isolate the segment where supply didn't cap sales.

## Alert rules (practical defaults)
All thresholds are configurable in `config/thresholds.yaml`.

1) Supply alert: OOS spike
- Condition: OOS_rate_week > 1.3 * OOS_rate_4w_avg AND LostSales_GMV_week > threshold
- Action: supply team review (expedite / redistribute / order)

2) Model alert: quality deterioration on non-OOS
- Condition: WAPE_non_OOS_week > WAPE_non_OOS_4w_avg + delta AND volume_weighted_units > threshold
- Action: forecast owner review (parameters, promo factors, price elasticity)

3) Data alert: invalid values
- Condition: price <= 0 OR stock < 0 OR missing keys OR extreme outliers
- Action: DWH / source system fix; temporary fallback rules applied

## Output fields
See `docs/04_data_contract.md`.
