"""Main script for model validation."""
import pandas as pd
from solution import bias
from solution import best_model
from solution import GroupTimeSeriesSplit
from solution import mape
from solution import smape
from solution import wape


def main():
    # Data loading
    df_path = "../datasets/data_train_sql.csv"
    df = pd.read_csv(df_path, parse_dates=["monday"])

    y = df.pop("y")

    # monday as groups for validation (Time-Series split by weeks)
    # Нужно использовать monday как группу, потому что:
    # 1. Мы делаем Time-Series валидацию - train должен быть раньше test
    # 2. Все данные за одну неделю должны быть вместе (либо train, либо test)
    # 3. product_name не подходит - товары не упорядочены по времени
    df.drop("product_name", axis=1, inplace=True)
    groups = df.pop("monday")

    X = df

    # Validation loop
    cv = GroupTimeSeriesSplit(
        n_splits=5,
        max_train_size=None,
        test_size=None,
        gap=0,
    )

    print("Cross-validation results:")
    print("-" * 60)

    all_wape = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        # Split train/test
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model = best_model()
        model.fit(X_train, y_train)

        # Predict and print metrics
        y_pred = model.predict(X_test)

        fold_mape = mape(y_test.values, y_pred)
        fold_smape = smape(y_test.values, y_pred)
        fold_wape = wape(y_test.values, y_pred)
        fold_bias = bias(y_test.values, y_pred)

        all_wape.append(fold_wape)

        print(f"Fold {fold + 1}:")
        print(f"  MAPE:  {fold_mape:.2f}%")
        print(f"  sMAPE: {fold_smape:.2f}%")
        print(f"  WAPE:  {fold_wape:.2f}%")
        print(f"  Bias:  {fold_bias:.2f}%")
        print()

    print("-" * 60)
    print(f"Average WAPE: {sum(all_wape) / len(all_wape):.2f}%")


if __name__ == "__main__":
    main()
