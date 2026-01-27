
# 05. Decision Playbook (What to do with the numbers)

## When LostSales is high
1) Check if it is concentrated:
- Top 10 SKUs contribute > 60% of lost_sales_gmv → focus actions there.
2) Identify root:
- If OOS_rate high but non-OOS WAPE stable → supply issue
- If non-OOS WAPE deteriorates → model issue
- If data alerts fire → data issue

## Recommended actions by role
Supply/Logistics:
- expedite inbound shipment
- redistribute stock between DC/stores
- adjust safety stock for promo SKUs

Commercial/Marketing:
- limit promo volume, change promo depth
- substitute SKU in promo mechanics
- switch traffic to alternatives

Forecast owner (DS / planning):
- re-estimate price elasticity / promo uplift factors
- update calendar features and lagged effects
- review outliers and cleaning rules

## Weekly leadership summary template
- Total LostSales_GMV, top categories, top SKUs
- OOS_rate vs last 4 weeks
- WAPE_non_OOS and Bias_non_OOS (is the model drifting?)
- Action list: owner, due date, expected impact
