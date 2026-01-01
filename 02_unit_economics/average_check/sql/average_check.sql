SELECT 
    toStartOfMonth(toDate(buy_date)) AS month,
    avg(check_amount) AS avg_check,
    toFloat64(quantileExact(0.5)(check_amount)) AS median_check
FROM default.view_checks
GROUP BY month
ORDER BY month
