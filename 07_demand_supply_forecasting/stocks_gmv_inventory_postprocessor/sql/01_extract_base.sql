
-- 01_extract_base.sql (ClickHouse template)
-- Purpose: build a SKU-day dataset with forecast, price, stock and actual sales.
-- Replace table names and date filters for your environment.

SELECT
    date,
    sku,
    category,
    price,
    stock,
    forecast_gmv,
    actual_units
FROM mart_forecast_sku_day
WHERE date BETWEEN {start_date} AND {end_date};
