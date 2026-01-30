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
    targets: Dict[str, Tuple[str, int]],
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    print("Extracting features...")

    df_features = df_sales.copy()

    # Add features
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

    # Add targets
    for target_name, (col, horizon) in targets.items():
        df_features[target_name] = (
            df_features.groupby("sku_id")[col]
            .transform(
                lambda x: x.shift(-1).rolling(window=horizon, min_periods=horizon).sum()
            )
        )

    df_features.sort_values(["sku_id", "day"], inplace=True)

    print(f"Features extracted. features.csv shape: {df_features.shape}")

    return df_features


@PipelineDecorator.component(
    return_values=["df_train", "df_test"],
    cache=True,
    task_type=TaskTypes.data_processing,
    name="split_train_test",
)
def split_train_test(
    df_features: pd.DataFrame,
    test_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd

    print("Splitting train and test data...")

    # Ensure day column is datetime
    df_features["day"] = pd.to_datetime(df_features["day"])

    max_date = df_features["day"].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days - 1)

    df_test = df_features[df_features["day"] >= cutoff_date].copy()
    df_train = df_features[df_features["day"] < cutoff_date].copy()

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    print("Train and test data splitted.")

    return df_train, df_test


@PipelineDecorator.component(
    return_values=["model"],
    cache=True,
    task_type=TaskTypes.training,
)
def fit_model(
    df_features: pd.DataFrame,
    features: List[str],
    quantiles: List[float],
    horizons: List[int],
):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import QuantileRegressor

    print("Training production model...")

    # Drop NaN values before training
    target_cols = [f"next_{h}d" for h in horizons]
    cols_to_check = features + target_cols
    df_clean = df_features.dropna(subset=cols_to_check)

    print(f"Data shape after dropping NaN: {df_clean.shape}")

    # Train models
    fitted_models = {}
    sku_ids = df_clean["sku_id"].unique()

    for sku_id in sku_ids:
        sku_data = df_clean[df_clean["sku_id"] == sku_id].copy()

        if len(sku_data) == 0:
            continue

        fitted_models[sku_id] = {}
        X = sku_data[features].values

        for quantile in quantiles:
            for horizon in horizons:
                target_col = f"next_{horizon}d"
                y = sku_data[target_col].values

                model = QuantileRegressor(
                    quantile=quantile,
                    alpha=0.0,
                    solver="highs"
                )
                model.fit(X, y)
                fitted_models[sku_id][(quantile, horizon)] = model

    # Return model info as dict
    model_info = {
        "fitted_models": fitted_models,
        "features": features,
        "quantiles": quantiles,
        "horizons": horizons,
    }

    print("Production model trained.")

    return model_info


@PipelineDecorator.component(
    return_values=["eval_model"],
    cache=True,
    task_type=TaskTypes.training,
)
def fit_eval_model(
    df_train: pd.DataFrame,
    features: List[str],
    quantiles: List[float],
    horizons: List[int],
):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import QuantileRegressor

    print("Training evaluation model...")

    # Drop NaN values before training
    target_cols = [f"next_{h}d" for h in horizons]
    cols_to_check = features + target_cols
    df_clean = df_train.dropna(subset=cols_to_check)

    print(f"Train data shape after dropping NaN: {df_clean.shape}")

    # Train models
    fitted_models = {}
    sku_ids = df_clean["sku_id"].unique()

    for sku_id in sku_ids:
        sku_data = df_clean[df_clean["sku_id"] == sku_id].copy()

        if len(sku_data) == 0:
            continue

        fitted_models[sku_id] = {}
        X = sku_data[features].values

        for quantile in quantiles:
            for horizon in horizons:
                target_col = f"next_{horizon}d"
                y = sku_data[target_col].values

                model = QuantileRegressor(
                    quantile=quantile,
                    alpha=0.0,
                    solver="highs"
                )
                model.fit(X, y)
                fitted_models[sku_id][(quantile, horizon)] = model

    # Return model info as dict
    model_info = {
        "fitted_models": fitted_models,
        "features": features,
        "quantiles": quantiles,
        "horizons": horizons,
    }

    print("Evaluation model trained.")

    return model_info


@PipelineDecorator.component(
    return_values=["losses", "df_pred"],
    task_type=TaskTypes.qc,
)
def evaluate(
    eval_model,
    df_test: pd.DataFrame,
    quantiles: List[float],
    horizons: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd
    import numpy as np

    print("Evaluating model...")

    fitted_models = eval_model["fitted_models"]
    features = eval_model["features"]

    # Ensure day column is datetime
    df_test["day"] = pd.to_datetime(df_test["day"])

    # Drop NaN from test data for evaluation
    target_cols = [f"next_{h}d" for h in horizons]
    cols_to_check = features + target_cols
    df_test_clean = df_test.dropna(subset=cols_to_check)

    print(f"Test data shape after dropping NaN: {df_test_clean.shape}")

    # Make predictions
    predictions = df_test_clean[["sku_id", "day"]].copy()

    for horizon in horizons:
        for quantile in quantiles:
            col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
            predictions[col_name] = 0.0

    sku_ids = df_test_clean["sku_id"].unique()

    for sku_id in sku_ids:
        if sku_id not in fitted_models:
            continue

        sku_mask = df_test_clean["sku_id"] == sku_id
        sku_data = df_test_clean[sku_mask]
        X = sku_data[features].values

        for quantile in quantiles:
            for horizon in horizons:
                model_key = (quantile, horizon)
                if model_key in fitted_models[sku_id]:
                    model = fitted_models[sku_id][model_key]
                    preds = model.predict(X)
                    col_name = f"pred_{horizon}d_q{int(quantile * 100)}"
                    predictions.loc[sku_mask, col_name] = preds

    df_pred = predictions

    # Calculate losses
    def quantile_loss(y_true, y_pred, quantile):
        diff = y_true - y_pred
        return np.mean(
            quantile * np.maximum(diff, 0) + (1 - quantile) * np.maximum(-diff, 0)
        )

    losses_dict = {}
    for quantile in quantiles:
        for horizon in horizons:
            true = df_test_clean[f"next_{horizon}d"].values
            pred = df_pred[f"pred_{horizon}d_q{int(quantile*100)}"].values
            loss = quantile_loss(true, pred, quantile)
            losses_dict[(quantile, horizon)] = loss

    losses = pd.DataFrame(losses_dict, index=["loss"]).T.reset_index()
    losses.columns = ["quantile", "horizon", "avg_quantile_loss"]

    print(f"Losses:\n{losses}")
    print("Model evaluated.")

    return losses, df_pred


@PipelineDecorator.component(
    task_type=TaskTypes.custom,
)
def deploy_model(
    model,
    model_path: str,
    losses: pd.DataFrame,
    df_pred: pd.DataFrame,
) -> None:
    import pickle

    print("Check model quality...")

    print(f"Losses:\n{losses}")

    pred_cols_q10 = [col for col in df_pred.columns if "q10" in col]
    pred_cols_q90 = [col for col in df_pred.columns if "q90" in col]

    if pred_cols_q10 and pred_cols_q90:
        avg_q10 = df_pred[pred_cols_q10].mean().mean()
        avg_q90 = df_pred[pred_cols_q90].mean().mean()
        print(f"Average q10 prediction: {avg_q10:.2f}")
        print(f"Average q90 prediction: {avg_q90:.2f}")

    print("Quality checked. Saving production model...")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Production model saved to {model_path}!")


@PipelineDecorator.pipeline(
    name="Training Pipeline",
    project="Stock Management System Task",
    version="1.0.0",
)
def run_pipeline(
    orders_url: str,
    test_days: int,
    model_path: str,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
    targets: Dict[str, Tuple[str, int]],
    quantiles: List[float],
    horizons: List[int],
) -> None:
    orders_df = fetch_orders(orders_url)

    df_sales = extract_sales(orders_df)

    df_features = extract_features(df_sales, features, targets)

    model_features = ["price", "qty"] + list(features.keys())

    df_train, df_test = split_train_test(df_features, test_days)

    model = fit_model(df_features, model_features, quantiles, horizons)

    eval_model = fit_eval_model(df_train, model_features, quantiles, horizons)

    losses, df_pred = evaluate(eval_model, df_test, quantiles, horizons)

    deploy_model(model, model_path, losses, df_pred)


def main(
    orders_url: str = "https://disk.yandex.ru/d/NUDMAdBMe9sbLw",
    model_path: str = "model.pkl",
    debug: bool = False,
) -> None:
    if debug:
        PipelineDecorator.debug_pipeline()
    else:
        PipelineDecorator.run_locally()

    test_days = 30

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

    targets = {
        "next_7d": ("qty", 7),
        "next_14d": ("qty", 14),
        "next_21d": ("qty", 21),
    }

    quantiles = [0.1, 0.5, 0.9]

    horizons = [7, 14, 21]

    run_pipeline(
        orders_url=orders_url,
        test_days=test_days,
        model_path=model_path,
        features=features,
        targets=targets,
        quantiles=quantiles,
        horizons=horizons,
    )


if __name__ == "__main__":
    fire.Fire(main)
