import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = ["sku", "category", "cost", "price", "qty"]


@dataclass
class RunConfig:
    input_path: str
    outdir: str
    target_margin: float
    category: Optional[str]
    scenario: Optional[str]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_and_validate(cfg: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cfg.input_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if cfg.category is not None:
        df = df[df["category"].astype(str) == str(cfg.category)].copy()

    if cfg.scenario is not None:
        if "scenario_id" not in df.columns:
            raise ValueError("scenario_id column is required when --scenario is used")
        df = df[df["scenario_id"].astype(str) == str(cfg.scenario)].copy()

    # Basic type normalization
    df["sku"] = df["sku"].astype(str)
    df["category"] = df["category"].astype(str)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    # Data quality checks
    dq_rows = []

    def flag(mask, reason):
        nonlocal dq_rows
        bad = df[mask].copy()
        if not bad.empty:
            bad["dq_reason"] = reason
            dq_rows.append(bad)

    flag(df["cost"].isna(), "cost_is_nan")
    flag(df["price"].isna(), "price_is_nan")
    flag(df["qty"].isna(), "qty_is_nan")
    flag(df["price"] <= 0, "price_le_0")
    flag(df["qty"] < 0, "qty_lt_0")
    flag(df["cost"] < 0, "cost_lt_0")

    dq = pd.concat(dq_rows, ignore_index=True) if dq_rows else pd.DataFrame(columns=list(df.columns) + ["dq_reason"])

    # Drop invalid rows from main set
    valid_mask = (
        df["cost"].notna()
        & df["price"].notna()
        & df["qty"].notna()
        & (df["price"] > 0)
        & (df["qty"] >= 0)
        & (df["cost"] >= 0)
    )
    df = df[valid_mask].copy()

    # Deduplicate exact duplicates (same sku/price/cost/qty)
    df = df.drop_duplicates(subset=["sku", "category", "cost", "price", "qty"])

    return df, dq


def compute_candidate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gmv"] = out["price"] * out["qty"]
    # margin can be negative; keep as float
    out["margin"] = (out["price"] - out["cost"]) / out["price"]
    # gross profit for optional reporting
    out["gp"] = (out["price"] - out["cost"]) * out["qty"]
    return out


def weighted_margin(df_selected: pd.DataFrame) -> float:
    total_gmv = df_selected["gmv"].sum()
    if total_gmv <= 0:
        return 0.0
    return float((df_selected["gmv"] * df_selected["margin"]).sum() / total_gmv)


def initial_plan_max_gmv(df: pd.DataFrame) -> pd.DataFrame:
    # Choose candidate with max GMV per SKU (tie-breaker: higher margin)
    idx = (
        df.sort_values(["sku", "gmv", "margin"], ascending=[True, False, False])
          .groupby("sku", as_index=False)
          .head(1)
          .index
    )
    return df.loc[idx].copy()


def best_possible_margin_plan(df: pd.DataFrame) -> pd.DataFrame:
    # Choose candidate with max margin per SKU (tie-breaker: higher GMV)
    idx = (
        df.sort_values(["sku", "margin", "gmv"], ascending=[True, False, False])
          .groupby("sku", as_index=False)
          .head(1)
          .index
    )
    return df.loc[idx].copy()


def repair_to_target(df_all: pd.DataFrame, df_plan: pd.DataFrame, target: float) -> Tuple[pd.DataFrame, dict]:
    """
    Greedy repair:
    - if current weighted_margin < target, switch some SKUs to higher-margin candidates
    - prioritize switches with minimal GMV loss per margin improvement (proxy)
    """
    meta = {
        "status": "ok",
        "iterations": 0,
        "repairs_applied": 0,
    }

    current = df_plan.set_index("sku")[["price", "cost", "qty", "gmv", "margin", "gp", "category"]].copy()
    cur_wm = weighted_margin(current.reset_index())
    if cur_wm >= target:
        meta["status"] = "already_feasible"
        return current.reset_index(), meta

    # Precompute per SKU candidate list
    # For each SKU compute deltas vs current choice
    repairs = []
    for sku, grp in df_all.groupby("sku", sort=False):
        if sku not in current.index:
            continue
        cur_row = current.loc[sku]
        cur_price = cur_row["price"]

        # candidates excluding current selected row (by price+cost+qty match can vary; compare by GMV+margin)
        grp2 = grp.copy()
        # Build delta vs current selected point
        grp2["delta_gmv"] = grp2["gmv"] - float(cur_row["gmv"])
        grp2["delta_gp"] = grp2["gp"] - float(cur_row["gp"])
        grp2["delta_margin"] = grp2["margin"] - float(cur_row["margin"])

        # We need margin to go up
        grp2 = grp2[grp2["delta_margin"] > 0].copy()
        if grp2.empty:
            continue

        # Proxy: prefer small GMV loss for margin gain; allow GMV gain too
        # score = GMV_loss_per_margin_gain (lower is better). If delta_gmv >=0, make it very attractive.
        gmv_loss = (-grp2["delta_gmv"]).clip(lower=0)  # only penalize losses
        grp2["repair_score"] = np.where(
            grp2["delta_gmv"] >= 0,
            -1e9,
            gmv_loss / grp2["delta_margin"]
        )

        # keep best candidate for repair per SKU
        best = grp2.sort_values("repair_score", ascending=True).head(1).iloc[0]
        repairs.append({
            "sku": sku,
            "new_price": float(best["price"]),
            "new_cost": float(best["cost"]),
            "new_qty": float(best["qty"]),
            "new_gmv": float(best["gmv"]),
            "new_margin": float(best["margin"]),
            "new_gp": float(best["gp"]),
            "delta_gmv": float(best["delta_gmv"]),
            "delta_margin": float(best["delta_margin"]),
            "repair_score": float(best["repair_score"]),
        })

    if not repairs:
        meta["status"] = "infeasible_no_repairs"
        return current.reset_index(), meta

    repairs_df = pd.DataFrame(repairs).sort_values("repair_score", ascending=True)

    # Apply repairs until target reached or we run out
    # Note: weighted margin depends on GMV weights; we do iterative recompute for correctness.
    plan = current.reset_index().copy()
    for _, r in repairs_df.iterrows():
        meta["iterations"] += 1

        # apply switch for SKU
        plan.loc[plan["sku"] == r["sku"], ["price", "cost", "qty", "gmv", "margin", "gp"]] = [
            r["new_price"], r["new_cost"], r["new_qty"], r["new_gmv"], r["new_margin"], r["new_gp"]
        ]
        meta["repairs_applied"] += 1

        cur_wm = weighted_margin(plan)
        if cur_wm >= target:
            meta["status"] = "feasible_after_repairs"
            return plan, meta

    meta["status"] = "still_infeasible_after_repairs"
    return plan, meta


def build_reports(df_all: pd.DataFrame, df_plan: pd.DataFrame, target: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    wm = weighted_margin(df_plan)
    total_gmv = float(df_plan["gmv"].sum())
    total_gp = float(df_plan["gp"].sum())
    buffer_pp = (wm - target) * 100.0

    category = df_plan["category"].iloc[0] if not df_plan.empty else "UNKNOWN"

    summary = pd.DataFrame([{
        "category": category,
        "target_margin": target,
        "weighted_margin": wm,
        "margin_buffer_pp": buffer_pp,
        "total_gmv": total_gmv,
        "total_gp": total_gp,
        "n_skus": int(df_plan["sku"].nunique()),
    }])

    # Contribution analysis: who supports/destroys weighted margin
    contrib = df_plan.copy()
    if total_gmv > 0:
        contrib["gmv_penetration"] = contrib["gmv"] / total_gmv
    else:
        contrib["gmv_penetration"] = 0.0
    contrib["weighted_margin_contribution"] = contrib["gmv_penetration"] * contrib["margin"]

    top_support = contrib.sort_values("weighted_margin_contribution", ascending=False).head(20)
    top_damage = contrib.sort_values("weighted_margin_contribution", ascending=True).head(20)

    # Action list: focus on negative margin + high penetration
    action = contrib.copy()
    action = action[(action["margin"] < 0) & (action["gmv_penetration"] >= 0.02)].copy()
    action["issue"] = "negative_margin_high_penetration"
    action["recommended_action"] = "Review promo depth / negotiate cost / adjust price floor / replace SKU in promo set"
    action = action.sort_values("gmv_penetration", ascending=False)

    # Alerts text
    alerts = []
    if df_plan.empty:
        alerts.append("ALERT: empty_plan | No valid rows after data quality filtering.")
    else:
        if wm < target:
            alerts.append(f"ALERT: infeasible_or_not_met | weighted_margin={wm:.4f} < target={target:.4f}.")
        if buffer_pp < 1.0 and wm >= target:
            alerts.append(f"ALERT: low_margin_buffer | buffer={buffer_pp:.2f}pp. Risk of missing target due to forecast error / cost change.")
        loss_leader_risk = contrib[(contrib["margin"] < 0) & (contrib["gmv_penetration"] >= 0.05)]
        if not loss_leader_risk.empty:
            alerts.append("ALERT: loss_leader_penetration | Negative margin SKUs have high GMV penetration. Check pricing strategy and supplier conditions.")

    # Add a compact block with top contributors (for management)
    alerts.append("INFO: top_support_skus | " + ", ".join(top_support["sku"].astype(str).head(10).tolist()))
    alerts.append("INFO: top_damage_skus | " + ", ".join(top_damage["sku"].astype(str).head(10).tolist()))
    alerts_text = "\n".join(alerts) + "\n"

    return summary, top_support, action, alerts_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV (sku/category/cost/price/qty ...)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--target-margin", required=True, type=float, help="Target weighted margin (e.g. 0.12)")
    parser.add_argument("--category", default=None, help="Optional category filter")
    parser.add_argument("--scenario", default=None, help="Optional scenario_id filter")

    args = parser.parse_args()
    cfg = RunConfig(
        input_path=args.input,
        outdir=args.outdir,
        target_margin=float(args.target_margin),
        category=args.category,
        scenario=args.scenario,
    )

    ensure_dir(cfg.outdir)

    df_raw, dq = load_and_validate(cfg)
    dq.to_csv(os.path.join(cfg.outdir, "data_quality_report.csv"), index=False)

    df = compute_candidate_metrics(df_raw)

    # Feasibility check
    plan_best_margin = best_possible_margin_plan(df)
    wm_best = weighted_margin(plan_best_margin)

    # Initial plan (maximize GMV)
    plan_init = initial_plan_max_gmv(df)
    wm_init = weighted_margin(plan_init)

    # Repair if needed
    plan_final, meta = repair_to_target(df, plan_init, cfg.target_margin)
    wm_final = weighted_margin(plan_final)

    # Reports
    summary, top_support, action_list, alerts_text = build_reports(df, plan_final, cfg.target_margin)

    # Save outputs
    plan_final.sort_values(["category", "sku"]).to_csv(os.path.join(cfg.outdir, "recommended_prices.csv"), index=False)
    summary.to_csv(os.path.join(cfg.outdir, "category_summary.csv"), index=False)
    top_support.to_csv(os.path.join(cfg.outdir, "top_support_skus.csv"), index=False)
    action_list.to_csv(os.path.join(cfg.outdir, "action_list.csv"), index=False)

    with open(os.path.join(cfg.outdir, "alerts.md"), "w", encoding="utf-8") as f:
        f.write("# Alerts\n\n")
        f.write(alerts_text)
        f.write("\n")
        f.write("## Run metadata\n")
        f.write(f"- feasibility_best_possible_margin: {wm_best:.6f}\n")
        f.write(f"- initial_plan_weighted_margin: {wm_init:.6f}\n")
        f.write(f"- final_plan_weighted_margin: {wm_final:.6f}\n")
        f.write(f"- status: {meta.get('status')}\n")
        f.write(f"- repairs_applied: {meta.get('repairs_applied')}\n")

    print("Done.")
    print(f"Status: {meta.get('status')}")
    print(f"Initial weighted margin: {wm_init:.6f}")
    print(f"Final weighted margin:   {wm_final:.6f}")
    print(f"Best possible margin:    {wm_best:.6f}")


if __name__ == "__main__":
    main()

