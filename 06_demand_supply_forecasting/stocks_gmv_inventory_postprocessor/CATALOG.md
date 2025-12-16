
# Catalog (Repository Map)

This folder contains a practical retail demand planning case: governance of a demand forecast under inventory constraints.

## Main outputs
- report/governance_daily.csv: SKU-day governance table (forecast, feasible plan, actuals, lost sales, flags)
- report/management_weekly_summary.csv: weekly management summary (quality, OOS, losses)
- report/alerts.csv: alerts with recommended actions

## Key code
- src/core/feasible_plan.py: convert forecast GMV -> integer units; apply stock constraint; compute feasible GMV
- src/core/oos_impact.py: compute lost sales (units/GMV) and OOS flags
- src/core/metrics.py: WAPE/Bias + “non-OOS” evaluation
- src/alerts/rules.py: alert rules and action routing
- src/pipeline/run_demo.py: demo pipeline entrypoint

## Documentation
- docs/01_business_context.md
- docs/02_operating_model.md
- docs/03_metrics_and_alerts.md
- docs/04_data_contract.md
- docs/05_decision_playbook.md

## SQL templates (ClickHouse)
- sql/01_extract_base.sql
- sql/02_management_summary.sql
