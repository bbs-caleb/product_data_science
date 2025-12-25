SELECT
    formatDateTime(day, '%Y-%m-%d') AS day,
    sku_id,
    sku,
    price,
    qty
FROM (
    SELECT
        d.day AS day,
        s.sku_id AS sku_id,
        s.sku AS sku,
        s.price AS price,
        COALESCE(sales.total_qty, 0) AS qty
    FROM (
        SELECT DISTINCT toDate(timestamp) AS day
        FROM default.demand_orders
    ) AS d
    CROSS JOIN (
        SELECT DISTINCT
            sku_id,
            sku,
            price
        FROM default.demand_orders
    ) AS s
    LEFT JOIN (
        SELECT
            toDate(o.timestamp) AS day,
            o.sku_id AS sku_id,
            SUM(o.qty) AS total_qty
        FROM default.demand_orders AS o
        WHERE o.order_id IN (
            SELECT order_id
            FROM default.demand_orders_status
            WHERE status NOT IN ('Return', 'Lost', 'Canceled')
        )
        GROUP BY toDate(o.timestamp), o.sku_id
    ) AS sales
        ON d.day = sales.day AND s.sku_id = sales.sku_id
)
ORDER BY sku_id, day
