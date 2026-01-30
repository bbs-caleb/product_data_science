
-- 02_management_summary.sql (ClickHouse template)
-- Purpose: weekly management summary from the governance table (published to BI).

SELECT
    toStartOfWeek(date) AS week,
    category,
    sum(lost_sales_gmv) AS lost_sales_gmv,
    avg(is_stock_constrained) AS oos_rate,
    sum(actual_units) AS actual_units
FROM governance_daily
GROUP BY
    week,
    category
ORDER BY
    week DESC,
    lost_sales_gmv DESC;
