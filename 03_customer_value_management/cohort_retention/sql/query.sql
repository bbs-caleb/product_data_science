WITH cohorts AS (
    SELECT 
        user_id,
        EXTRACT(MONTH FROM reg_date)::int AS cohort_id
    FROM cr_users
    WHERE reg_date >= '2024-01-01' AND reg_date < '2025-01-01'
),
cohort_sizes AS (
    SELECT 
        cohort_id,
        COUNT(*) AS total_users
    FROM cohorts
    GROUP BY cohort_id
),
max_month AS (
    SELECT MAX(EXTRACT(MONTH FROM action_date))::int AS max_m
    FROM cr_user_actions
    WHERE action_date >= '2024-01-01' AND action_date < '2025-01-01'
),
cohort_months AS (
    SELECT 
        cs.cohort_id,
        m.month_id
    FROM cohort_sizes cs
    CROSS JOIN LATERAL generate_series(cs.cohort_id, (SELECT max_m FROM max_month)) AS m(month_id)
),
monthly_activity AS (
    SELECT 
        c.user_id,
        c.cohort_id,
        EXTRACT(MONTH FROM a.action_date)::int AS month_id
    FROM cohorts c
    JOIN cr_user_actions a ON c.user_id = a.user_id
    WHERE a.action_date >= '2024-01-01' AND a.action_date < '2025-01-01'
    GROUP BY c.user_id, c.cohort_id, EXTRACT(MONTH FROM a.action_date)
    HAVING 
        SUM(CASE WHEN a.action_type = 'open_app' THEN 1 ELSE 0 END) > 0
        AND SUM(CASE WHEN a.action_type IN ('manage_notes', 'manage_plants', 'set_reminder', 'view_articles', 'share_plants') THEN 1 ELSE 0 END) > 0
),
active_counts AS (
    SELECT 
        cohort_id,
        month_id,
        COUNT(DISTINCT user_id) AS active_users
    FROM monthly_activity
    GROUP BY cohort_id, month_id
)
SELECT 
    cm.cohort_id,
    TRIM(TO_CHAR(TO_DATE(cm.cohort_id::text, 'MM'), 'Month')) || ' users' AS cohort_name,
    cm.month_id,
    COALESCE(ac.active_users, 0) AS active_users,
    ROUND(COALESCE(ac.active_users, 0) * 100.0 / cs.total_users, 1) AS retention_rate
FROM cohort_months cm
JOIN cohort_sizes cs ON cm.cohort_id = cs.cohort_id
LEFT JOIN active_counts ac ON cm.cohort_id = ac.cohort_id AND cm.month_id = ac.month_id
ORDER BY cm.cohort_id, cm.month_id;
