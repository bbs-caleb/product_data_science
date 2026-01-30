# Technical Notes

## Goal
Select exactly one price candidate per SKU to maximize GMV subject to category weighted margin >= target.

## Metrics
GMV_i = price_i * qty_i
margin_i = (price_i - cost_i) / price_i
weighted_margin = sum(GMV_i * margin_i) / sum(GMV_i)

## Algorithm (practical, explainable)
1) Start from GMV-max candidate per SKU (best sales plan by revenue).
2) If weighted_margin >= target -> done.
3) If not, iteratively "repair" the plan:
   - For each SKU compute alternative candidates that increase margin.
   - Estimate "GMV loss per margin gain" for switching candidate:
     delta_gmv / delta_weighted_margin (proxy).
   - Apply the cheapest repairs first until target is met or infeasible.

This is not a perfect global optimum (NP-hard-like with discrete choices),
but it is fast, explainable to business, and produces an action list.

## Infeasibility detection
Compute best achievable weighted margin by selecting, for each SKU,
the candidate with maximum margin (or maximum GP), then check target.
If still below target -> infeasible scenario (requires external actions).

