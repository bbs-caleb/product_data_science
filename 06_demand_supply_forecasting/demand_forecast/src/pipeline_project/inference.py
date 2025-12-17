from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import fire
import pandas as pd
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(
    return_values=["orders"],
    task_type=TaskTypes.data_processing,
)
def fetch_orders(orders_url: str) -> pd.DataFrame:
    import requests
    from urllib.parse import urlencode
    import pandas as pd
    from clearml import StorageManager

    print(f"Downloading orders data from {orders_url}...")

    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    full_url = base_url + urlencode(dict(public_key=orders_url))
    response = requests.get(full_url)
    download_url = response.json()["href"]

    local_path = StorageManager.get_local_copy(remote_url=download_url)
    df_orders = pd.read_csv(local_path)
    df_orders["timestamp"] = pd.to_datetime(df_orders["timestamp"])

    print(f"Orders data downloaded. orders.csv shape: {df_orders.shape}")

    return df_orders


@PipelineDecorator.component(
    return_values=["sales"],
    task_type=TaskTypes.data_processing,
)
def extract_sales(df_orders: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    print("Extracting sales data...")

    df_orders["timestamp"] = pd.to_datetime(df_orders["timestamp"])
    df_sales = df_orders.copy()

    df_sales["day"] = df_sales["timestamp"].dt.floor("D")

    df_sales = (
        df_sales.groupby(["day", "sku_id", "sku", "price"])["qty"].sum().reset_index()
    )

    all_sku_ids = df_sales["sku_id"].unique()
    all_dates = pd.date_range(
        df_sales["day"].min(),
        df_sales["day"].max(),
        freq="D",
    )

    all_dates_sku_df = pd.DataFrame(
        {
            "day": np.repeat(all_dates, len(all_sku_ids)),
            "sku_id": np.tile(all_sku_ids, len(all_dates)),
        }
    )

    df_sales = pd.merge(all_dates_sku_df, df_sales, how="left", on=["day", "sku_id"])
    df_sales["qty"] = df_sales["qty"].fillna(0).astype(int)

    df = df_orders[["sku_id", "sku", "price"]].drop_duplicates().reset_index(drop=True)
    df_sales = pd.merge(
        df_sales, df[["sku_id", "sku", "price"]], how="left", on="sku_id"
    )
    df_sales["sku"] = df_sales["sku_x"].fillna(df_sales["sku_y"])
    df_sales["price"] = df_sales["price_x"].fillna(df_sales["price_y"])
    df_sales.drop(columns=["sku_x", "sku_y", "price_x", "price_y"], inplace=True)

    df_sales = df_sales[["day", "sku_id", "sku", "price", "qty"]]

    df_sales.sort_values(by=["sku_id", "day"], inplace=True)
    df_sales.reset_index(drop=True, inplace=True)

    print(f"Sales data extracted. sales.csv shape: {df_sales.shape}")

    return df_sales


@PipelineDecorator.component(
    return_values=["features"],
    task_type=TaskTypes.data_processing,
)
def extract_features(
    df_sales: pd.DataFrame,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    print("Extracting features...")

    df_features = df_sales.copy()

    # Add features only (no targets for inference)
    for feature_name, (col, window, agg_type, q_value) in features.items():
        if agg_type == "avg":
            df_features[feature_name] = (
                df_features.groupby("sku_id")[col]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
        elif agg_type == "quantile":
            quantile = q_value / 100.0
            df_features[feature_name] = (
                df_features.groupby("sku_id")[col]
                .transform(
                    lambda x: x.rolling(window=window, min_periods=1).quantile(quantile)
                )
            )

    df_features.sort_values(["sku_id", "day"], inplace=True)

    print(f"Features extracted. features.csv shape: {df_features.shape}")

    return df_features


@PipelineDecorator.component(
    return_values=["predictions"],
    task_type=TaskTypes.inference,
)
def predict(
    model_path: str,
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    import pickle

    print("Making predictions...")

    # Ensure day column is datetime
    df_features["day"] = pd.to_datetime(df_features["day"])

    # Get only the last day for each SKU
    last_day = df_features["day"].max()
    df_last_day = df_features[df_features["day"] == last_day].copy()

    print(f"Predicting for date: {last_day}")
    print(f"Number of SKUs: {len(df_last_day)}")

    # Load model
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)

    fitted_models = model_info["fitted_models"]
    features = model_info["features"]
    quantiles = model_info["quantiles"]
    horizons = model_info["horizons"]

    # Drop rows with NaN in features
    df_clean = df_last_day.dropna(subset=features)

    print(f"Data shape after dropping NaN: {df_clean.shape}")

    # Create predictions dataframe
    predictions = df_clean[["sku_id", "day"]].copy()

    # Initialize prediction columns with zeros
    for horizon in horizons:
        for quantile in quantiles:
            col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
            predictions[col_name] = 0.0

    # Make predictions for each SKU
    sku_ids = df_clean["sku_id"].unique()

    for sku_id in sku_ids:
        if sku_id not in fitted_models:
            continue

        sku_mask = df_clean["sku_id"] == sku_id
        sku_data = df_clean[sku_mask]
        X = sku_data[features].values

        for quantile in quantiles:
            for horizon in horizons:
                model_key = (quantile, horizon)
                if model_key in fitted_models[sku_id]:
                    model = fitted_models[sku_id][model_key]
                    preds = model.predict(X)
                    col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
                    predictions.loc[sku_mask, col_name] = preds

    # Sort by sku_id
    predictions = predictions.sort_values("sku_id").reset_index(drop=True)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions:\n{predictions}")

    return predictions


@PipelineDecorator.pipeline(
    name="Inference Pipeline",
    project="Stock Management System Task",
    version="1.0.0",
)
def run_pipeline(
    orders_url: str,
    model_path: str,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
) -> None:
    # Step 1: Fetch orders data
    orders_df = fetch_orders(orders_url)

    # Step 2: Extract sales data
    df_sales = extract_sales(orders_df)

    # Step 3: Extract features (no targets for inference)
    df_features = extract_features(df_sales, features)

    # Step 4: Make predictions
    predictions = predict(model_path, df_features)


def main(
    orders_url: str = "https://disk.yandex.ru/d/OK5gyMuEfhJA0g",
    model_path: str = "model.pkl",
    debug: bool = False,
) -> None:
    """Main function

    Args:
        orders_url (str): URL to the orders data on Yandex Disk (last 21 days)
        model_path (str): Local path of production model
        debug (bool, optional): Run the pipeline in debug mode.
            In debug mode no Tasks are created, so it is running faster.
            Defaults to False.
    """

    if debug:
        PipelineDecorator.debug_pipeline()
    else:
        PipelineDecorator.run_locally()

    features = {
        "qty_7d_avg": ("qty", 7, "avg", None),
        "qty_7d_q10": ("qty", 7, "quantile", 10),
        "qty_7d_q50": ("qty", 7, "quantile", 50),
        "qty_7d_q90": ("qty", 7, "quantile", 90),
        "qty_14d_avg": ("qty", 14, "avg", None),
        "qty_14d_q10": ("qty", 14, "quantile", 10),
        "qty_14d_q50": ("qty", 14, "quantile", 50),
        "qty_14d_q90": ("qty", 14, "quantile", 90),
        "qty_21d_avg": ("qty", 21, "avg", None),
        "qty_21d_q10": ("qty", 21, "quantile", 10),
        "qty_21d_q50": ("qty", 21, "quantile", 50),
        "qty_21d_q90": ("qty", 21, "quantile", 90),
    }

    run_pipeline(
        orders_url=orders_url,
        model_path=model_path,
        features=features,
    )


if __name__ == "__main__":
    fire.Fire(main)
