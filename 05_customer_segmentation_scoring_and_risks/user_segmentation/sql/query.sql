WITH filtered_payments AS (
    SELECT amount
    FROM new_payments
    WHERE status = 'Confirmed'
      AND mode IN ('MasterCard', 'МИР', 'Visa')
),
max_amount AS (
    SELECT CAST(MAX(amount) AS INTEGER) AS max_amt
    FROM filtered_payments
),
segments AS (
    SELECT 
        CASE 
            WHEN amount <= 20000 THEN '0-20000'
            WHEN amount <= 40000 THEN '20000-40000'
            WHEN amount <= 60000 THEN '40000-60000'
            WHEN amount <= 80000 THEN '60000-80000'
            WHEN amount <= 100000 THEN '80000-100000'
            ELSE '100000-' || (SELECT max_amt FROM max_amount)
        END AS purchase_range,
        CASE 
            WHEN amount <= 20000 THEN 1
            WHEN amount <= 40000 THEN 2
            WHEN amount <= 60000 THEN 3
            WHEN amount <= 80000 THEN 4
            WHEN amount <= 100000 THEN 5
            ELSE 6
        END AS segment_order
    FROM filtered_payments
)
SELECT 
    purchase_range,
    COUNT(*) AS num_of_users
FROM segments
GROUP BY purchase_range, segment_order
ORDER BY segment_order;
