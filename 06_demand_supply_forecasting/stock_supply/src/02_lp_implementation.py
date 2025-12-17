"""Greedy and LP Restock Algorithm Implementations."""
from typing import Tuple
import math
import json
import pulp
import pandas as pd


class GreedyRestock:
    """
    A class to allocate a budget for restocking products using a greedy algorithm.
    """

    def __init__(self, json_path: str):
        """
        Initializes the GreedyRestock class by loading products from a JSON file.

        Args:
            json_path (str): The file path to the JSON file containing product data.

        Raises:
            FileNotFoundError: If the provided JSON file does not exist.
            ValueError: If the JSON data cannot be decoded or if the product list is empty.
            RuntimeError: If an unexpected error occurs while reading the JSON file.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.products = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"JSON file not found: {json_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON format in file: {json_path}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error reading JSON file: {json_path}"
            ) from exc

        if not self.products:
            raise ValueError("Product list is empty")

        self._validate()

    def _validate(self):
        """
        Validates the structure and data of the products loaded from the JSON file.
        """
        required_keys = {"sku_id", "price", "cost", "current_stock",
                         "storage_time", "percentiles"}
        required_weeks = {"1w", "2w", "3w", "4w"}
        required_percentiles = {"5", "10", "25", "50", "75", "90", "95"}

        for product in self.products:
            missing_keys = required_keys - set(product.keys())
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")

            if product["price"] <= 0:
                raise ValueError("Price must be positive")
            if product["cost"] <= 0:
                raise ValueError("Cost must be positive")
            if product["price"] < product["cost"]:
                raise ValueError("Price must be >= cost")
            if product["current_stock"] < 0:
                raise ValueError("Current stock must be non-negative")
            if product["storage_time"] <= 0:
                raise ValueError("Storage time must be positive")

            percentiles = product["percentiles"]
            missing_weeks = required_weeks - set(percentiles.keys())
            if missing_weeks:
                raise ValueError(f"Missing weeks: {missing_weeks}")

            for week in required_weeks:
                missing_pcts = required_percentiles - set(percentiles[week].keys())
                if missing_pcts:
                    raise ValueError(f"Missing percentiles for {week}: {missing_pcts}")

    def allocate(
        self, budget: int, weeks: int, optimization_goal: str = "profit"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Allocates a budget for restocking products based on a greedy algorithm.
        """
        if budget <= 0:
            raise ValueError("Budget must be a positive integer")
        if weeks < 1 or weeks > 4:
            raise ValueError("Weeks must be between 1 and 4")
        if optimization_goal not in ("profit", "revenue"):
            raise ValueError("Optimization goal must be 'profit' or 'revenue'")

        week_key = f"{weeks}w"
        rows = []

        for product in self.products:
            sku_id = product["sku_id"]
            price = product["price"]
            cost = product["cost"]
            current_stock = product["current_stock"]
            storage_time = product["storage_time"]
            percentiles = product["percentiles"][week_key]

            storage_score = math.log10(storage_time)

            for pct_str, demand in percentiles.items():
                percentile = int(pct_str)
                demand_to_buy = max(0, demand - current_stock)

                if demand_to_buy <= 0:
                    continue

                oos_risk = 1 - percentile / 100.0
                tot_profit = (price - cost) * demand_to_buy
                tot_revenue = price * demand_to_buy
                tot_cost = cost * demand_to_buy

                profit_score = ((price - cost) / cost) * oos_risk * storage_score
                revenue_score = (price / cost) * demand_to_buy * oos_risk * storage_score

                rows.append({
                    "sku_id": sku_id,
                    "percentile": percentile,
                    "demand_to_buy": demand_to_buy,
                    "price": price,
                    "cost": cost,
                    "tot_profit": tot_profit,
                    "tot_revenue": tot_revenue,
                    "tot_cost": tot_cost,
                    "storage_score": storage_score,
                    "oos_risk": oos_risk,
                    "profit_score": profit_score,
                    "revenue_score": revenue_score,
                })

        sorted_df = pd.DataFrame(rows)

        if len(sorted_df) == 0:
            sorted_df = pd.DataFrame(columns=[
                "sku_id", "percentile", "demand_to_buy", "price", "cost",
                "tot_profit", "tot_revenue", "tot_cost", "storage_score",
                "oos_risk", "profit_score", "revenue_score"
            ])
            allocation_df = pd.DataFrame(columns=[
                "sku_id", "percentile", "price", "cost", "allocated_qty",
                "allocated_budget", "expected_profit", "expected_revenue"
            ])
            return sorted_df, allocation_df

        score_col = "profit_score" if optimization_goal == "profit" else "revenue_score"
        sorted_df = sorted_df.sort_values(
            by=[score_col, "sku_id"],
            ascending=[False, True]
        ).reset_index(drop=True)

        remaining_budget = budget
        allocated_skus = set()
        allocation_rows = []

        for _, row in sorted_df.iterrows():
            sku_id = row["sku_id"]

            if sku_id in allocated_skus:
                continue

            demand_to_buy = row["demand_to_buy"]
            cost = row["cost"]
            price = row["price"]
            percentile = row["percentile"]

            max_affordable = remaining_budget // cost
            if max_affordable <= 0:
                continue

            allocated_qty = min(demand_to_buy, max_affordable)
            allocated_budget = allocated_qty * cost
            expected_profit = allocated_qty * (price - cost)
            expected_revenue = allocated_qty * price

            allocation_rows.append({
                "sku_id": sku_id,
                "percentile": percentile,
                "price": price,
                "cost": cost,
                "allocated_qty": allocated_qty,
                "allocated_budget": allocated_budget,
                "expected_profit": expected_profit,
                "expected_revenue": expected_revenue,
            })

            remaining_budget -= allocated_budget
            allocated_skus.add(sku_id)

            if remaining_budget <= 0:
                break

        allocation_df = pd.DataFrame(allocation_rows)

        if len(allocation_df) > 0:
            allocation_df = allocation_df.sort_values(
                by=["allocated_budget", "sku_id"],
                ascending=[False, True]
            ).reset_index(drop=True)

        return sorted_df, allocation_df


class LPRestock(GreedyRestock):
    """
    A class to allocate budget using linear programming optimization.
    """

    def allocate(
        self, budget: int, weeks: int, optimization_goal: str = "profit"
    ) -> Tuple[pulp.LpProblem, pd.DataFrame]:
        """
        Allocates budget for restocking SKUs using linear programming.
        """
        if budget <= 0:
            raise ValueError("Budget must be a positive integer")
        if weeks < 1 or weeks > 4:
            raise ValueError("Weeks must be between 1 and 4")
        if optimization_goal not in ("profit", "revenue"):
            raise ValueError("Optimization goal must be 'profit' or 'revenue'")

        week_key = f"{weeks}w"

        # Collect all SKU-percentile combinations with demand > 0
        sku_data = []
        for product in self.products:
            sku_id = product["sku_id"]
            price = product["price"]
            cost = product["cost"]
            current_stock = product["current_stock"]
            storage_time = product["storage_time"]
            percentiles = product["percentiles"][week_key]

            storage_score = math.log10(storage_time)

            for pct_str, demand in percentiles.items():
                percentile = int(pct_str)
                demand_to_buy = max(0, demand - current_stock)

                if demand_to_buy <= 0:
                    continue

                oos_risk = 1 - percentile / 100.0
                profit_score = ((price - cost) / cost) * oos_risk * storage_score
                revenue_score = (price / cost) * demand_to_buy * oos_risk * storage_score

                sku_data.append({
                    "sku_id": sku_id,
                    "percentile": percentile,
                    "demand_to_buy": demand_to_buy,
                    "price": price,
                    "cost": cost,
                    "profit_score": round(profit_score, 4),
                    "revenue_score": round(revenue_score, 4),
                })

        # Create LP problem
        prob = pulp.LpProblem("Restock_Optimization", pulp.LpMaximize)

        # Create decision variables and binary variables
        decision_vars = {}
        binary_vars = {}

        for item in sku_data:
            var_name = f"sku_{item['sku_id']}_{item['percentile']}"
            bin_name = f"bin_{var_name}"

            decision_vars[var_name] = pulp.LpVariable(
                var_name,
                lowBound=0,
                upBound=item["demand_to_buy"],
                cat="Integer"
            )
            binary_vars[var_name] = pulp.LpVariable(
                bin_name,
                cat="Binary"
            )

        # Set objective function
        score_key = "profit_score" if optimization_goal == "profit" else "revenue_score"
        objective_terms = []
        for item in sku_data:
            var_name = f"sku_{item['sku_id']}_{item['percentile']}"
            coefficient = item[score_key]
            objective_terms.append(coefficient * decision_vars[var_name])

        prob += pulp.lpSum(objective_terms), "Objective"

        # Budget constraint
        budget_terms = []
        for item in sku_data:
            var_name = f"sku_{item['sku_id']}_{item['percentile']}"
            budget_terms.append(item["cost"] * decision_vars[var_name])

        prob += pulp.lpSum(budget_terms) <= budget, "budget_constraint"

        # Single selection constraint per SKU
        sku_vars = {}
        for item in sku_data:
            sku_id = item["sku_id"]
            var_name = f"sku_{sku_id}_{item['percentile']}"
            if sku_id not in sku_vars:
                sku_vars[sku_id] = []
            sku_vars[sku_id].append(var_name)

        for sku_id, var_names in sku_vars.items():
            constraint_name = f"sku_{sku_id}_single_selection_constraint"
            prob += (
                pulp.lpSum([binary_vars[vn] for vn in var_names]) == 1,
                constraint_name
            )

        # Demand constraints
        for item in sku_data:
            var_name = f"sku_{item['sku_id']}_{item['percentile']}"
            demand = item["demand_to_buy"]
            constraint_name = f"{var_name}_demand_constraint"
            prob += (
                decision_vars[var_name] <= demand * binary_vars[var_name],
                constraint_name
            )

        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Build allocation DataFrame
        allocation_rows = []
        for item in sku_data:
            var_name = f"sku_{item['sku_id']}_{item['percentile']}"
            qty = int(pulp.value(decision_vars[var_name]) or 0)

            if qty > 0:
                allocation_rows.append({
                    "sku_id": item["sku_id"],
                    "percentile": item["percentile"],
                    "price": item["price"],
                    "cost": item["cost"],
                    "allocated_qty": qty,
                    "allocated_budget": qty * item["cost"],
                    "expected_profit": qty * (item["price"] - item["cost"]),
                    "expected_revenue": qty * item["price"],
                })

        allocation_df = pd.DataFrame(allocation_rows)

        if len(allocation_df) > 0:
            allocation_df = allocation_df.sort_values(
                by=["allocated_budget", "sku_id"],
                ascending=[False, True]
            ).reset_index(drop=True)

        return prob, allocation_df
