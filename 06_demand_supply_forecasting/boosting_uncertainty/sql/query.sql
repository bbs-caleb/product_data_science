/* features.sql (ClickHouse)

Строит витрину product-week с лагами/роллингами по продукту и по всем продуктам.
Таблица-источник: default.data_sales_train (dt, product_name, user_id, price)

Если нужно ограничить период — раскомментируйте WHERE и подставьте параметры в вашем окружении.
*/

WITH
base AS (
    SELECT
        product_name,
        toMonday(toDate(dt)) AS monday,
        max(price) AS max_price,
        count() AS y
    FROM default.data_sales_train
    /* WHERE toDate(dt) BETWEEN toDate({{start_date}}) AND toDate({{end_date}}) */
    GROUP BY
        product_name,
        monday
),

product_feats AS (
    SELECT
        product_name,
        monday,
        max_price,
        y,

        /* lags per product */
        lagInFrame(y, 1) OVER w_prod AS y_lag_1,
        lagInFrame(y, 2) OVER w_prod AS y_lag_2,
        lagInFrame(y, 3) OVER w_prod AS y_lag_3,
        lagInFrame(y, 4) OVER w_prod AS y_lag_4,
        lagInFrame(y, 5) OVER w_prod AS y_lag_5,
        lagInFrame(y, 6) OVER w_prod AS y_lag_6,

        /* rolling stats per product: last 3 weeks (excluding current) */
        avg(y) OVER w_prod_3 AS y_avg_3,
        max(y) OVER w_prod_3 AS y_max_3,
        min(y) OVER w_prod_3 AS y_min_3,

        /* rolling stats per product: last 6 weeks (excluding current) */
        avg(y) OVER w_prod_6 AS y_avg_6,
        max(y) OVER w_prod_6 AS y_max_6,
        min(y) OVER w_prod_6 AS y_min_6

    FROM base
    WINDOW
        w_prod AS (
            PARTITION BY product_name
            ORDER BY monday
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        w_prod_3 AS (
            PARTITION BY product_name
            ORDER BY monday
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ),
        w_prod_6 AS (
            PARTITION BY product_name
            ORDER BY monday
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        )
),

all_weekly AS (
    SELECT
        monday,
        sum(y) AS y_all
    FROM base
    GROUP BY monday
),

all_feats AS (
    SELECT
        monday,

        /* lags for all products (total weekly sales) */
        lagInFrame(y_all, 1) OVER w_all AS y_all_lag_1,
        lagInFrame(y_all, 2) OVER w_all AS y_all_lag_2,
        lagInFrame(y_all, 3) OVER w_all AS y_all_lag_3,
        lagInFrame(y_all, 4) OVER w_all AS y_all_lag_4,
        lagInFrame(y_all, 5) OVER w_all AS y_all_lag_5,
        lagInFrame(y_all, 6) OVER w_all AS y_all_lag_6,

        /* rolling stats for all products: last 3 weeks (excluding current) */
        avg(y_all) OVER w_all_3 AS y_all_avg_3,
        max(y_all) OVER w_all_3 AS y_all_max_3,
        min(y_all) OVER w_all_3 AS y_all_min_3,

        /* rolling stats for all products: last 6 weeks (excluding current) */
        avg(y_all) OVER w_all_6 AS y_all_avg_6,
        max(y_all) OVER w_all_6 AS y_all_max_6,
        min(y_all) OVER w_all_6 AS y_all_min_6

    FROM all_weekly
    WINDOW
        w_all AS (
            PARTITION BY 1
            ORDER BY monday
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        w_all_3 AS (
            PARTITION BY 1
            ORDER BY monday
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ),
        w_all_6 AS (
            PARTITION BY 1
            ORDER BY monday
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        )
)

SELECT
    pf.product_name,
    pf.monday,
    pf.max_price,
    pf.y,

    pf.y_lag_1, pf.y_lag_2, pf.y_lag_3, pf.y_lag_4, pf.y_lag_5, pf.y_lag_6,
    pf.y_avg_3, pf.y_max_3, pf.y_min_3,
    pf.y_avg_6, pf.y_max_6, pf.y_min_6,

    af.y_all_lag_1, af.y_all_lag_2, af.y_all_lag_3, af.y_all_lag_4, af.y_all_lag_5, af.y_all_lag_6,
    af.y_all_avg_3, af.y_all_max_3, af.y_all_min_3,
    af.y_all_avg_6, af.y_all_max_6, af.y_all_min_6
FROM product_feats AS pf
LEFT JOIN all_feats AS af USING (monday)
ORDER BY
    pf.product_name,
    pf.monday;
