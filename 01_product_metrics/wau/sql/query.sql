/* WAU (7-day rolling, inclusive), plus DAU and Sticky Factor */

WITH daily_users AS (
    -- Deduplicate to (day, user_id): multiple submits per day should count as 1 active user
    SELECT
        toDate(timestamp) AS day,
        user_id
    FROM default.churn_submits
    GROUP BY day, user_id
),

wau_by_day AS (
    SELECT
        day,
        -- For a fixed day, WAU is identical across all rows of that day (window depends only on day).
        -- any() safely collapses N identical values to 1 row/day (max/min would be equivalent).
        any(wau) AS wau
    FROM (
        SELECT
            day,
            -- 7-day rolling window: [day-6 .. day] in calendar days
            uniqExact(user_id) OVER (
                ORDER BY day
                RANGE BETWEEN 6 PRECEDING AND CURRENT ROW
            ) AS wau
        FROM daily_users
    )
    GROUP BY day
),

dau_by_day AS (
    SELECT
        day,
        uniqExact(user_id) AS dau
    FROM daily_users
    GROUP BY day
)

SELECT
    d.day AS day,
    w.wau AS wau,
    d.dau AS dau,
    if(w.wau = 0, 0.0, d.dau / w.wau) AS sticky_factor
FROM dau_by_day d
JOIN wau_by_day w USING(day)
ORDER BY day;
