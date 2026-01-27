SELECT 
    DATE_TRUNC('month', date)::date AS time,
    SUM(amount) / COUNT(DISTINCT email_id) AS arppu,
    SUM(amount) / COUNT(*) AS aov
FROM new_payments
WHERE status = 'Confirmed'
GROUP BY DATE_TRUNC('month', date)
ORDER BY time;
