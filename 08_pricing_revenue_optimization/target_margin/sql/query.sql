/* Extract for price scenario planning.

Expected columns:
sku, category, cost, price, qty, scenario_id

Notes:
- qty is forecasted units for the planning horizon under each price candidate.
- scenario_id can be 'regular' / 'promo' / 'competitor_response'.
*/

SELECT
    sku,
    category,
    cost,
    price,
    qty,
    scenario_id
FROM your_schema.your_price_candidates_forecast
WHERE qty IS NOT NULL
;

