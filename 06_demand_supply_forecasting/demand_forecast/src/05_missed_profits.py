from typing import Tuple

import numpy as np
import pandas as pd


def week_missed_profits(
    df: pd.DataFrame,
    sales_col: str,
    forecast_col: str,
    date_col: str = "day",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Calculates the missed profits every week for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed profits for.
        (Must contain columns "sku_id", "date", "price", "sales" and "forecast")
    sales_col : str
        The column with the actual sales.
    forecast_col : str
        The column with the forecasted sales.
    price_col : str, optional
        The column with the price, by default "price".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the missed profits.
        (Contains columns "day", "revenue", "missed_profits")
    """
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Calculate missed sales (units): max(forecast - sales, 0)
    # This represents how many units we could have sold but didn't
    data["missed_sales"] = np.maximum(
        data[forecast_col] - data[sales_col], 0
    )

    # Calculate missed profits in money
    data["missed_profits"] = data["missed_sales"] * data[price_col]

    # Calculate actual revenue
    data["revenue"] = data[sales_col] * data[price_col]

    # Group by week
    # Convert date column to datetime if needed
    data[date_col] = pd.to_datetime(data[date_col])

    # Group by week (using the first day of each week)
    result = data.groupby(pd.Grouper(key=date_col, freq="W")).agg({
        "revenue": "sum",
        "missed_profits": "sum"
    }).reset_index()

    # Rename date column to "day"
    result = result.rename(columns={date_col: "day"})

    return result


def missed_profits_ci(
    df: pd.DataFrame,
    missed_profits_col: str,
    confidence_level: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]:
    """
    Estimates the missed profits for the given DataFrame.
    Calculates average missed_profits per week and estimates
    the 95% confidence interval.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed_profits for.

    missed_profits_col : str
        The column with the missed_profits.

    confidence_level : float, optional
        The confidence level for the confidence interval, by default 0.95.

    n_bootstraps : int, optional
        The number of bootstrap samples to use for the confidence interval,
        by default 1000.

    Returns
    -------
    Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]
        Returns a tuple of tuples, where the first tuple is the absolute average
        missed profits with its CI, and the second is the relative average missed
        profits with its CI.

    Example:
    -------
    ((1200000, (1100000, 1300000)), (0.5, (0.4, 0.6)))
    """
    # Extract the missed profits and revenue values
    missed_profits_values = df[missed_profits_col].values
    revenue_values = df["revenue"].values

    n_samples = len(missed_profits_values)

    # Calculate the point estimates (average values)
    avg_missed_profits = np.mean(missed_profits_values)
    avg_revenue = np.mean(revenue_values)
    relative_missed_profits = avg_missed_profits / avg_revenue

    # Bootstrap
    bootstrap_abs_means = []
    bootstrap_rel_means = []

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_missed = missed_profits_values[indices]
        sample_revenue = revenue_values[indices]

        # Calculate mean for this bootstrap sample
        bootstrap_abs_means.append(np.mean(sample_missed))
        bootstrap_rel_means.append(np.mean(sample_missed) / np.mean(sample_revenue))

    # Convert to numpy arrays
    bootstrap_abs_means = np.array(bootstrap_abs_means)
    bootstrap_rel_means = np.array(bootstrap_rel_means)

    # Calculate quantiles for confidence interval
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    # Absolute CI
    abs_ci_lower = np.quantile(bootstrap_abs_means, lower_quantile)
    abs_ci_upper = np.quantile(bootstrap_abs_means, upper_quantile)

    # Relative CI
    rel_ci_lower = np.quantile(bootstrap_rel_means, lower_quantile)
    rel_ci_upper = np.quantile(bootstrap_rel_means, upper_quantile)

    # Return results as specified format
    absolute_result = (avg_missed_profits, (abs_ci_lower, abs_ci_upper))
    relative_result = (relative_missed_profits, (rel_ci_lower, rel_ci_upper))

    return (absolute_result, relative_result)
