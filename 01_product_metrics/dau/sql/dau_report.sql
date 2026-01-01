SELECT
    toDate(timestamp) AS day,
    count(DISTINCT user_id) AS dau
FROM default.churn_submits
GROUP BY day
ORDER BY day
