-- базовая сетка - все комбинации user_id × day
with users as (select distinct user_id
               from default.churn_submits),
     days as (select distinct toDate(timestamp) as day
from default.churn_submits
    ), user_days as (
select user_id, day
from users
    cross join days
    ),

-- агрегация активности по дням
    activity as (
select
    user_id, toDate(timestamp) as day, count (*) as n_submits, count (distinct task_id) as n_tasks, sum (is_solved) as n_solved
from default.churn_submits
group by user_id, day
    ),

-- заполняем пропуски нулями
    full_activity as (
select
    ud.user_id, ud.day, coalesce (a.n_submits, 0) as n_submits, coalesce (a.n_tasks, 0) as n_tasks, coalesce (a.n_solved, 0) as n_solved
from user_days ud
    left join activity a
on ud.user_id = a.user_id and ud.day = a.day
    )

select
    day, user_id,

    -- days_offline -  разница между текущим днем и последним днем с активностью
    -- если активности не было - null
    day - max (if(n_submits > 0, day, null))
    over (partition by user_id order by day rows between unbounded preceding and current row)
    as days_offline,

    -- avg_submits_14d: сумма сабмитов за 14 дней / 14 (всегда делим на 14)
    sum (n_submits)
    over (partition by user_id order by day rows between 13 preceding and current row) / 14.0
    as avg_submits_14d,

    -- success_rate_14d: доля успешных сабмитов за 14 дней
    -- если сабмитов не было - 0
    coalesce (
    sum (n_solved) over (partition by user_id order by day rows between 13 preceding and current row) * 1.0 /
    nullif (sum (n_submits) over (partition by user_id order by day rows between 13 preceding and current row), 0), 0
    ) as success_rate_14d,

    -- solved_total: накопленная сумма успешных сабмитов
    sum (n_solved)
    over (partition by user_id order by day rows between unbounded preceding and current row)
    as solved_total,

    -- target_14d: 1 если в следующие 14 дней не было активности, иначе 0
    if(
    sum (n_submits) over (partition by user_id order by day rows between 1 following and 14 following) = 0, 1, 0
    ) as target_14d

from full_activity
order by user_id, day
