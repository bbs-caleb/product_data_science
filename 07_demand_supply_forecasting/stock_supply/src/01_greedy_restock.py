"""Greedy Restock Algorithm Implementation."""
from typing import Tuple
import math
import json
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
            # Check required keys
            missing_keys = required_keys - set(product.keys())
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")

            # Validate numeric fields
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

            # Validate percentiles structure
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

        Args:
            budget (int): The total budget available for restocking.
            weeks (int): The number of weeks to consider for demand forecasting (1 to 4).
            optimization_goal (str): The goal of the optimization, either "profit" or "revenue".

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: sorted_df and allocation_df
        """
        # Validate inputs
        if budget <= 0:
            raise ValueError("Budget must be a positive integer")
        if weeks < 1 or weeks > 4:
            raise ValueError("Weeks must be between 1 and 4")
        if optimization_goal not in ("profit", "revenue"):
            raise ValueError("Optimization goal must be 'profit' or 'revenue'")

        week_key = f"{weeks}w"
        rows = []

        # Calculate scores for each product and percentile
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

                # Skip if nothing to buy
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

        # Create sorted DataFrame
        sorted_df = pd.DataFrame(rows)

        if len(sorted_df) == 0:
            # Return empty DataFrames with correct columns
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

        # Sort by score (descending) and sku_id (ascending)
        score_col = "profit_score" if optimization_goal == "profit" else "revenue_score"
        sorted_df = sorted_df.sort_values(
            by=[score_col, "sku_id"],
            ascending=[False, True]
        ).reset_index(drop=True)

        # Greedy allocation
        # For each SKU, keep only the row with max score
        remaining_budget = budget
        allocated_skus = set()
        allocation_rows = []

        for _, row in sorted_df.iterrows():
            sku_id = row["sku_id"]

            # Skip if already allocated for this SKU
            if sku_id in allocated_skus:
                continue

            demand_to_buy = row["demand_to_buy"]
            cost = row["cost"]
            price = row["price"]
            percentile = row["percentile"]

            # Calculate how many we can afford
            max_affordable = remaining_budget // cost
            if max_affordable <= 0:
                continue

            # Allocate min of what we need and what we can afford
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

        # Create allocation DataFrame
        allocation_df = pd.DataFrame(allocation_rows)

        if len(allocation_df) > 0:
            # Sort by allocated_budget (descending) and sku_id (ascending)
            allocation_df = allocation_df.sort_values(
                by=["allocated_budget", "sku_id"],
                ascending=[False, True]
            ).reset_index(drop=True)

        return sorted_df, allocation_df
